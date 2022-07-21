
"""
Inputs: <significance_output/dataset.lang-pair/??-vs-baseline/id-sig.txt>
Plot: For each dataset.lang-pair/??-vs-baseline, give a plot
"""

from matplotlib import pyplot as plt
import os
import os.path as osp

ROOT = 'paired-bootstrap/significance_output'

def read_data (ROOT):
    x=[], y=[]
    for fn in os.listdir(ROOT):
        i, sig = fn.rsplit('.',1)[0].split('-')
        x.append(int(i))
        y.append(float(sig))
    return x,y

fig, ax = plt.subplot()

for ds in os.listdir(ROOT):
    dataset = ds.rsplit('.', 1)[0]
    lang_pair = ds.rsplit('.', 1)[-1]
    ROOT2 = osp.join(ROOT, ds)
    x,y = read_data (ROOT2)
    ax.scatter(x,y,marker='+',)
    plt.show()
    exit(0)
