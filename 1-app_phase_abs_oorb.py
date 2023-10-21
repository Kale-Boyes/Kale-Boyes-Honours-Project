# Authors: Nicolas Erasmus and Kale Boyes.
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.time import Time
from astropy.table import Table, vstack, QTable
import os
from scipy.optimize import curve_fit
import glob
from astropy.io import ascii
from sbpy.data import Orbit, Ephem
from astroquery.jplhorizons import Horizons



def get_phase_parameters(phase_angles,reduced_mag,reduced_mag_err):  
    if len(reduced_mag[np.where(phase_angles<10)]) > 0: 
        mag_around_phase_zero = np.median(reduced_mag[np.where(phase_angles<10)])
    else:
        mag_around_phase_zero = reduced_mag[np.where(phase_angles.min())]
    
    popt, pcov = curve_fit(f=phase_g_fit, xdata=phase_angles, ydata=reduced_mag, sigma = reduced_mag_err ,p0=[0.2,mag_around_phase_zero],bounds=([-0.5,mag_around_phase_zero-0.5],[1.5,mag_around_phase_zero+0.5]))
    
    return popt[0], np.sqrt(np.diag(pcov))[0], popt[1], np.sqrt(np.diag(pcov))[1]


def scrubber1(data, predicted_v_mags):
    ### data cleanup parameters ###
    mag5sig_cut = 18 # the higher the mag5sig of the image is the better the conditions where
    dm_cut = 0.1 #magnitude uncertainty
    out_mag = 1 # clipping magnitude from median observed magnitude
    ###############################
    
    clean_data = data[np.where((data["dm"]<=dm_cut) & (data["mag5sig"]>=mag5sig_cut) & (abs(data["m"]-predicted_v_mags)<out_mag))]
    
    return clean_data
    
def get_eph_JPL(ast_num, epochs):
    print('Querying JPL for {} epochs'.format(len(epochs)))

    #do the horizon query in batches of <80 (which seems to be the limit for a single query)
    num_batches = int(np.ceil(len(epochs)/80))# see how many batches of equal batches <80 in dataset
    num_in_batch = int(len(epochs)/num_batches)

    for i in range(num_batches):
        ephemeris = Ephem.from_horizons(ast_num,
                                  location='500',
                                  epochs=epochs[i*num_in_batch:(i*num_in_batch)+num_in_batch])
        ### create a QTable because vstack does not work with eph table directly apparently
        eph_table = QTable([ephemeris['epoch'], ephemeris['lighttime'], ephemeris['r'], ephemeris['delta'], ephemeris['alpha'],ephemeris['PABLon'],ephemeris['PABLat']],
                   names=('epoch', 'lighttime', 'r', 'delta', 'alpha', 'PABLon', 'PABLat'))

        if i == 0:
            eph = eph_table
        else:
            eph = vstack([eph,eph_table])
            
        

    return eph

def get_eph_oorb(ast_num, epochs):
    print('Querying Oorb for {} epochs'.format(len(epochs)))
    ast_num = ast_num.replace('_', ' ')
    obj = Horizons(id = ast_num, location='500@10', epochs = 2459800.5)
      
    el = obj.elements()

    my_asteroid = Orbit.from_dict({'targetname': ast_num,
                               'orbtype': 'KEP',
                               'a': el['a'][-1]* u.au,         #semimajor axis
                               'e': el['e'][-1],                #eccentricity
                               'i': el['incl'][-1] * u.deg,         #inclination
                               'w': el['w'][-1] * u.deg,        #argument of perihelion
                               'Omega': el['Omega'][-1]* u.deg,    #ascending node
                               'M': el['M'][-1] * u.deg,        #mean anomaly
                               'epoch': Time(2459800.5, format='jd'),
                               'H': el['H'].value * u.mag,               #absolute magnitude, 99 if unknown
                               'G': el['G'].value})                    #phase slope
    
    eph = Ephem.from_oo(my_asteroid, epochs, '500')
    c = 173.145 * (u.AU/u.d)
    eph.table.add_column(eph['Delta']/c, name = 'lighttime')
    
    return eph

def get_mean_mag(filt_data):
    num_iterations = 30
    filt_means = []
    for i in range(num_iterations):
        sample = np.random.choice(filt_data['m'], size = round(len(filt_data['m'])/2), replace=False)
        filt_means.append(np.mean(sample))
    
    filt_m_med = np.median(filt_means)
    filt_m_err = np.std(filt_means)
    
    return filt_m_med, filt_m_err
        
        
        
    
    
def phase_g_fit(pa,g,h): # from R. Dymock  2007

    tan_func = np.tan(0.5*np.deg2rad(pa))
    
    phi1 = np.exp(-3.332*np.power(tan_func,0.631))
    phi2 = np.exp(-1.862*np.power(tan_func,1.218))
    
    return h-2.5*np.log10( ((1-g)*phi1) + (g*phi2) )    
    
def main():
    
    asteroids_to_look_at = ["316"]
    phot_filters = ["o","c"]            
    plot_colours = {'o' : '#FFA500',
                    'c' : '#00a6a7'}
    
    
    light_curve_data_dir = os.getcwd()+"/forced_phot_data/TestAsteroid/"
    file_list = glob.glob(light_curve_data_dir+"/*.txt") # Include slash or it will search in the wrong directory!!
    
    for lc_file in file_list:
        try:
            file_name = lc_file.split('/')[-1]
            ast_num = file_name.split('.')[0]
            
        
            if ast_num not in asteroids_to_look_at:
                continue

            print('Looking at data for asteroid: {}'.format(ast_num))
        
        
            #### initialise plot for each object ####
            fig = plt.figure()
            fig.set_size_inches(5, 8)
            font_size = 12
            plt.clf()
            
            ax1 = plt.subplot2grid((3,1), (0,0),rowspan=1, colspan=1) # raw observing data:     o and c apaprent mag vs. mjd
            ax2 = plt.subplot2grid((3,1), (1,0),rowspan=1, colspan=1) # phase curve:            o and c reduced mag vs. phase angle with HG model fit
            ax3 = plt.subplot2grid((3,1), (2,0),rowspan=1, colspan=1) # absolute mag:           o and c absolute mag vs. mjd with determined colour using median values
            plt.suptitle('Asteroid number: {}'.format(ast_num))
            
            
            filt_m_meds=[]
            filt_m_errs =[]
        
            data = np.genfromtxt(lc_file,delimiter='',skip_header=1,dtype=None, encoding='UTF-8',
                                        names =["MJD","m","dm","uJy","duJy","F","err","chi_N","RA","Dec","x","y","maj","min","phi","apfit","mag5sig","Sky","Obs"])
            
            
            
            full_epochs = Time(data["MJD"],format = 'mjd', scale='utc')
            full_eph = get_eph_oorb(ast_num, full_epochs)
            
            data = scrubber1(data, full_eph.table['V'].value)
            
            if (len(data[np.where((data["F"]=='c'))]) <10) or(len(data[np.where((data["F"]=='o'))]) <10 ):
                continue
            
            
            for f, filt in enumerate(phot_filters):
                print('Parsing {}-filter data'.format(filt))
                colour_data = data[np.where((data["F"]==filt))]
        
                meds = get_mean_mag(colour_data)
                filt_m_meds.append(meds[0])
                filt_m_errs.append(meds[1])
                
                
                epochs = Time(colour_data["MJD"],format = 'mjd', scale='utc')
                
               
                eph = get_eph_oorb(ast_num, epochs)
                #eph = get_eph_JPL(ast_num, epochs) # Can switch to get ephemerides from JPL instead
                

                
                ax1.errorbar(epochs.value, colour_data['m'].astype(float), yerr=colour_data['dm'].astype(float),
                                         label = r'${}$-filter'.format(filt), markersize=6, alpha=0.3, marker='x',
                                         color=plot_colours[filt], markeredgecolor='none', linestyle='',
                                         capsize=3, elinewidth=0.5, capthick=0.5)
            
                
               
                
                colour_data = colour_data[0:len(eph)] # I think we loose some data points at the end so make the same size
                epochs = epochs[0:len(eph)] # I think we loose some data points at the end so make the same size
                                        
                reduced_mag = colour_data['m'].astype(float)-5*np.log10(np.multiply(eph['r'].value,eph['delta'].value))
                
                ax2.errorbar(np.array(eph['alpha'].value), np.array(reduced_mag), yerr=colour_data['dm'].astype(float),
                                         label = r'${}$-filter'.format(filt), markersize=6, alpha=0.3, marker='x',
                                         color=plot_colours[filt], markeredgecolor='none', linestyle='',
                                         capsize=3, elinewidth=0.5, capthick=0.5)
             
                
                G, G_err, H, H_err = get_phase_parameters(eph['alpha'].value,reduced_mag,colour_data['dm'].astype(float))
                            
                fit_pa = np.linspace(0, np.max(eph['alpha'].value),1000)
                fit =  phase_g_fit(fit_pa,G,H)
                ax2.plot(fit_pa,fit,c=plot_colours[filt], linestyle='solid',linewidth=2,zorder=50)  
                ax2.text(0.60, 0.90-(0.15*f), r"G$_{}={}\pm{}$".format(filt,"%.2f" % G,"%.2f" % G_err),verticalalignment='center', horizontalalignment='left',transform=ax2.transAxes,fontsize=font_size*1.0)
                ax2.text(0.05, 0.25-(0.15*f), r"H$_{}={}\pm{}$".format(filt,"%.2f" % H,"%.2f" % H_err),verticalalignment='center', horizontalalignment='left',transform=ax2.transAxes,fontsize=font_size*1.0)
                
                
                abs_mag = reduced_mag-phase_g_fit(eph['alpha'].value,G,H)+H
                light_corr_time = epochs.value-eph['lighttime'].to(u.day).value+(30./2/60/60/24) # also add half exposure time to make it mid-exposure, ATLAS exposure time is 30 sec
                #light_corr_time = epochs.value
                ax3.errorbar(np.array(light_corr_time), np.array(abs_mag), yerr=colour_data['dm'].astype(float),
                                         label = r'${}$-filter'.format(filt), markersize=6, alpha=0.3, marker='x',
                                         color=plot_colours[filt], markeredgecolor='none', linestyle='',
                                         capsize=3, elinewidth=0.5, capthick=0.5)
                                         
                ### create a QTable because vstack works nicely with QTable
                data_table = QTable([   colour_data['Obs'], # NB not light-fly-time corrected and also start of exposure
                                        colour_data['F'],
                                        np.array(epochs.value), 
                                        np.array(eph['lighttime'].to(u.day).value), 
                                        np.array(light_corr_time), # NB also added half exposure time to make it mid-exposure, ATLAS exposure time is 30 sec
                                        colour_data['m'].astype(float),
                                        colour_data['dm'].astype(float),
                                        np.array(eph['alpha'].value),
                                        np.array(eph['r'].value),
                                        np.array(eph['delta'].value),
                                        np.ones(len(colour_data['Obs']))*H,
                                        np.ones(len(colour_data['Obs']))*H_err,
                                        np.ones(len(colour_data['Obs']))*G,
                                        np.ones(len(colour_data['Obs']))*G_err,
                                        np.array(abs_mag)],
                           names=(  'obs',
                                    'filter',
                                    'obs_mjd',
                                    'light_time',
                                    'corr_mjd',
                                    'obs_mag',
                                    'obs_mag_err',
                                    'phase_angle',
                                    'helio_dist',
                                    'geo_dist',
                                    'H',
                                    'H_err',
                                    'G',
                                    'G_err',
                                    'abs_mag'))
        
                if f == 0: # first filter
                    final_data_table = data_table
                else:
                    final_data_table = vstack([final_data_table,data_table])                                
                                 
            
            colour = filt_m_meds[1] - filt_m_meds[0] # c-o 
            colour_err = np.sqrt((filt_m_errs[0])**2 + (filt_m_errs[1])**2)
            
            final_data_table.add_column(np.ones(len(final_data_table['obs']))*colour, name = 'c-o_colour')
            final_data_table.add_column(np.ones(len(final_data_table['obs']))*colour_err, name = 'c-o_colour_err')
        
            
            print('c-o colour: ', colour, ' +/-', colour_err)
               
            ax1.invert_yaxis()
            ax1.legend(loc=4,fontsize=font_size*1.0)
            ax1.set_xlabel('date (mjd)',fontsize=font_size*1.0)
            ax1.xaxis.set_tick_params(labelsize=font_size*1.0)
            ax1.set_ylabel('app. magnitude',fontsize=font_size*1.0)
            ax1.yaxis.set_tick_params(labelsize=font_size*1.0)
            
            ax2.invert_yaxis()
            ax2.set_xlabel('phase angle (deg)',fontsize=font_size*1.0)
            ax2.xaxis.set_tick_params(labelsize=font_size*1.0)
            ax2.set_ylabel('red. magnitude',fontsize=font_size*1.0)
            ax2.yaxis.set_tick_params(labelsize=font_size*1.0)
            
            ax3.invert_yaxis()
            ax3.set_xlabel('date (mjd)',fontsize=font_size*1.0)
            ax3.xaxis.set_tick_params(labelsize=font_size*1.0)
            ax3.set_ylabel('abs. magnitude',fontsize=font_size*1.0)
            ax3.yaxis.set_tick_params(labelsize=font_size*1.0)
            
            
            plt.tight_layout(pad=1.5, w_pad=1, h_pad=1)
            plt.savefig(lc_file.replace('.txt','_app_phase_abs.pdf'), format='pdf',dpi=150)
            
            ascii.write(final_data_table, lc_file.replace('.txt','_table.csv'), 
                        formats={   'obs_mjd': '%.6f',
                                    'light_time': '%.6f',
                                    'corr_mjd': '%.6f',
                                    'H': '%.3f',
                                    'H_err': '%.3f',
                                    'G': '%.3f',
                                    'G_err': '%.3f',
                                    'abs_mag': '%.3f',
                                    'c-o_colour': '%.4f',
                                    'c-o_colour_err': '%.4f'
                                }, overwrite=True)  
        except:
            continue

if __name__ == "__main__":
    main()
