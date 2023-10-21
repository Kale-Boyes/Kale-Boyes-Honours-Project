# Author: Kale Boyes.
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn import neighbors
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import cross_val_score
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from numpy.random import uniform

n_neighbors = 1

# import data 
data_full = np.genfromtxt('colour_G_data.txt',delimiter=' ',skip_header=1,dtype=None, encoding='UTF-8',
                            names =['ast_num', 'colour', 'colour-err', 'G', 'G_err', 'tax_type'])

data = []
for c,g in zip(data_full['colour'], data_full['G']):
    data.append(np.array([c,g]))
data = np.array(data)
n_types = 3
n_samples = len(data_full['ast_num'])
n_features = 2
labels = data_full['tax_type']



X = data
y = labels

# Create color maps
cmap_light = ListedColormap(["yellow", "orange", "cornflowerblue"])
cmap_bold = ["royalblue","#db810b", "#e3e014"]



def gen_points(x, y, xerr,yerr):
    new_points = []
    for i in range(100):
        new_points.append([float(uniform(x-xerr, x+xerr,1)), float(uniform(y-yerr,y+yerr,1))])
    return new_points


for weights in ["uniform"]:
    # create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)
    
    
    
    print('CROSS VAL SCORE:')
    print(cross_val_score(clf, X, y, cv=5, scoring='accuracy'))
    
    
    # Plot decision surface
    _, ax = plt.subplots()
    DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        cmap=cmap_light,
        ax=ax,
        response_method="predict",
        plot_method="pcolormesh",
        xlabel='c-o colour',
        ylabel='G',
        shading="auto",
    )

    
    data_for_sns=[]
    for i in X:
        data_for_sns.append([float(i[0]),float(i[1])])
    
    data_for_sns = np.array(data_for_sns)
   
   
    # Plot training points
    sns.scatterplot(x=data_for_sns[:, 0], y=data_for_sns[:, 1], hue=y, palette= cmap_bold, alpha=1.0, edgecolor="black", legend=False)
    
    # test points
    #colour, G
    #colour_err, G_err 
    p1=[0.5361, 0.255] # S-type test point asteroid 9922 Catcheller
    p2=[0.1527, 0.029] # C-type test point asteroid 316 Goberta
    p1err= [0.0605, 0.024]
    p2err = [0.035, 0.009]

    # Plot test points
    plt.errorbar(x=p1[0], y=p1[1], yerr=p1err[1], xerr = p1err[0], fmt='o', ms= 5, color='#524f4d',markeredgecolor = 'white', markeredgewidth=1.5, markerfacecolor='darkorange', capsize=3,label='S-type test point')
    plt.errorbar(x=p2[0], y=p2[1], yerr=p2err[1], xerr = p2err[0], fmt='o',ms=5, color='#524f4d', markeredgecolor = 'white',markeredgewidth=1.5, markerfacecolor='#e3e014', capsize=3, label = 'C-type test point')
    
    test_x = [p1,p2]
    print('TEST POINTS:')
    print(clf.predict(test_x)) # Predicted type of each test point
    
    print('TEST ACCURACY SCORE:')
    print(clf.score(test_x, ['S','C'])) # The accuracy of the two predictions
    
    # Generate points around test points, within their uncertainty bounds
    generated_points1 = gen_points(p1[0], p1[1], p1err[0], p1err[1])
    generated_points2 = gen_points(p2[0], p2[1], p2err[0], p2err[1])
    
    # Predict the types of each generated point
    generated_predicts1 = clf.predict(generated_points1)
    generated_predicts2 = clf.predict(generated_points2)
    
    Scount1=0
    Ccount1=0
    Xcount1=0
    
    Scount2=0
    Ccount2=0
    Xcount2=0
    for p in generated_predicts1:
        if p == 'S':
            Scount1+=1
        elif p == 'C':
            Ccount1+=1
        elif p=='X':
            Xcount1+=1
    
    for p in generated_predicts2:
        if p == 'S':
            Scount2+=1
        elif p == 'C':
            Ccount2+=1
        elif p=='X':
            Xcount2+=1
    
    # Print the percentage of each type of generated points
    print('Point 1 (9922), from S type family is:')
    print(Scount1, '% S type')
    print(Ccount1, '% C type')
    print(Xcount1, '% X type')
    print()
    print('Point 2 (316), from C type family is:')
    print(Scount2, '% S type')
    print(Ccount2, '% C type')
    print(Xcount2, '% X type')
    print()
    
    
    
    plt.title(
        "Asteroid Classification (k = %i, weights = '%s')" % (n_neighbors, weights)
    )
    
    
    patchS = mpatches.Patch(color='orange', label='S-type')  
    patchX = mpatches.Patch(color='cornflowerblue', label='X-type') 
    patchC = mpatches.Patch(color='yellow', label='C-type') 
    point_patch = Line2D([], [], color="darkcyan", marker='o', markerfacecolor="#f2e9dc", markeredgecolor='k', linewidth=0,markeredgewidth=0.5, label = 'Training data')
    pointS = Line2D([], [], color="darkcyan", marker='o', markerfacecolor="#db810b", markeredgecolor='k', linewidth=0,markeredgewidth=0.5, label = 'S training data')
    pointC = Line2D([], [], color="darkcyan", marker='o', markerfacecolor="#e3e014", markeredgecolor='k', linewidth=0,markeredgewidth=0.5, label = 'C training data')
    pointX = Line2D([], [], color="darkcyan", marker='o', markerfacecolor="royalblue", markeredgecolor='k', linewidth=0,markeredgewidth=0.5, label = 'X training data')
    
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.extend([pointS, pointC, pointX, patchS,patchX,patchC])

    plt.legend(handles=handles)
    
plt.xlim((-0.5,1.25))
plt.ylim((-0.25,1.25))
plt.show()
plt.savefig('Tested_Decision_Surface.png', dpi=300, format='png')