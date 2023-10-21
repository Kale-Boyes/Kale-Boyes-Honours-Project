# Author: Kale Boyes.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from collections import OrderedDict

 
asteroids_to_look_at = ["316"]

light_curve_data_dir = os.getcwd()+"/forced_phot_data/"
file_list = glob.glob(light_curve_data_dir+"/*.csv") # Include slash or it will search in the wrong directory!!
fig, ax = plt.subplots()


with open('colour_G_data.txt', 'w') as f:
    f.write('ast_num colour colour_err G G_err tax_type \n')
    
    
training_names = pd.read_csv('list_for_tranining_set2.csv', ) #File with asteroid names and taxonomic types

for lc_file in file_list:
    file_name = lc_file.split('/')[-1]
    ast_num = file_name.split('_')[0]
    
# =============================================================================
#     if ast_num not in asteroids_to_look_at:71
#         continue
# =============================================================================
    
    ast_num = ast_num.replace('_', ' ')


    data = np.genfromtxt(lc_file,delimiter=' ',skip_header=1,dtype=None, encoding='UTF-8',
                                names =['obs',
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
                                         'abs_mag', 
                                         'c-o_colour',
                                         'c-o_colour_err'])
    
    
    
    # Taxonomic groups
    group1 = ['A','S','Q','V']
    group2 = ['C','B']
    group3 = ['X']
    group4 = ['D']
    
    
    with open('colour_G_data.txt', 'a') as f:
        
        ast_types = training_names.query('MainDesig =="' + str(ast_num) + '"')['Training_set'].values # Get taxonmic type for that asteroid number
        
        if len(ast_types) > 0:
            # Find which taxonmic group the asteroid is in, plot the point, and write the point to a text file
            if np.isin(ast_types, group1):
                ax.errorbar(data['co_colour'][0], data['G'][0], marker ='o', yerr= data['G_err'][0], xerr=data['co_colour_err'][0] , capsize=1 , ecolor = 'orange', markerfacecolor='orange', markeredgecolor='orange', label = 'S')
                f.write(str(ast_num) + ' ' + str(data['co_colour'][0]) + ' ' + str(data['co_colour_err'][0]) + ' ' + str(data['G'][0]) + ' ' + str(data['G_err'][0]) + ' S\n')
                
            elif np.isin(ast_types, group2):
                ax.errorbar(data['co_colour'][0], data['G'][0], marker ='o', yerr= data['G_err'][0], xerr=data['co_colour_err'][0] , capsize=1 , ecolor = 'lime',markerfacecolor='lime', markeredgecolor='lime', label = 'C')
                f.write(str(ast_num) + ' ' + str(data['co_colour'][0]) + ' ' + str(data['co_colour_err'][0]) + ' ' + str(data['G'][0]) + ' ' + str(data['G_err'][0]) + ' C\n')
                
            elif np.isin(ast_types, group3):
                ax.errorbar(data['co_colour'][0], data['G'][0], marker ='o', yerr= data['G_err'][0], xerr=data['co_colour_err'][0] , capsize=1 , ecolor = 'b',markerfacecolor='b', markeredgecolor='b',label = 'X')
                f.write(str(ast_num) + ' ' + str(data['co_colour'][0]) + ' ' + str(data['co_colour_err'][0]) + ' ' + str(data['G'][0]) + ' ' + str(data['G_err'][0]) + ' X\n')
                
            elif np.isin(ast_types, group4):
                ax.errorbar(data['co_colour'][0], data['G'][0], marker ='o', yerr= data['G_err'][0], xerr=data['co_colour_err'][0] , capsize=1 , ecolor = 'darkorchid',markerfacecolor='darkorchid', markeredgecolor='darkorchid',label = 'D')
                f.write(str(ast_num) + ' ' + str(data['co_colour'][0]) + ' ' + str(data['co_colour_err'][0]) + ' ' + str(data['G'][0]) + ' ' + str(data['G_err'][0]) + ' D\n')
 

# The expected c-o colour a slope parameters for S and C types
ax.axvline(x=0.388, ls = '--', color = 'orange', lw=2 ,label = 'Expected S')
ax.axvline(x=0.249, ls = '-.', color = 'lime' ,lw=2 , label = 'Expected C')

ax.axhline(y=0.23, ls = '--', color = 'orange', lw=2 )
ax.axhline(y=0.13, ls = '-.', color = 'lime' ,lw=2)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='upper right')

ax.set_ylabel('G')
ax.set_xlabel('c-o colour')