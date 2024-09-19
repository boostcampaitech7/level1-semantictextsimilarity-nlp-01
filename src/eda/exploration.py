import matplotlib as mpt
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def vizLabel(data):
    fig, ax = plt.subplots()
    
    bins = 26
    assert bins % 5 == 1
    hist = sns.histplot(data=data, x='label', ax=ax, bins=bins, color='skyblue')
    
    cmap = ['Reds', 'Purples', 'Blues', 'Greens', 'spring','Greys']
    cmap = list(map(plt.get_cmap, cmap))
    for i, patch in enumerate(hist.patches):
        k = i // (bins//5)
        patch.set_facecolor(cmap[k](1-(i % (bins//5)) / (3*(bins//5))))
    
    barW = 5 / bins
    ax.set_xticks(np.arange(barW/2, 5, ((bins-1)//5)*barW), labels=[0,1,2,3,4,5])
    
    return fig

def vizTokenLength(data):
    fig, ax = plt.subplots()
    flierprops = dict(marker='x', markersize=3, linestyle='none')
    sns.boxplot(data=data[['tokenLength_1', 'tokenLength_2']], ax=ax, orient='h', flierprops=flierprops)
    ax.set_yticklabels(['sentence_1', 'sentence_2'])
    print(len(data[data['tokenLength_1'] > 70]), len(data[data['tokenLength_2'] > 70]))
    return fig