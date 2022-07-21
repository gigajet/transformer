
"""
Inputs: <significance_output/dataset.lang-pair/??-vs-baseline/id-sig.txt>
Plot: For each dataset.lang-pair/??-vs-baseline, give a plot
"""

from matplotlib import pyplot as plt
import os
import os.path as osp

ROOT = 'paired-bootstrap/significance_output'
OROOT = 'paired-bootstrap/significance_plot'
IMG_FORMAT = 'png'

def read_data (ROOT):
    x=[]; y=[]
    for fn in os.listdir(ROOT):
        i, sig = fn.rsplit('.',1)[0].split('-')
        x.append(int(i))
        y.append(float(sig))
    return x,y

def print_statistic_data (ds:str, pr:str, a:str, b:str, x:list, y:list)->None:
    n_sample = len(x)
    # dset, pair, a, b, num_sample, num_larger_than_50
    n_better = len(list(filter(lambda x : x>50, y)))
    print("{0},{1},{2},{3},{4},{5}".format(ds,pr,a,b,n_sample,n_better))

print('dataset,language_pair,model_a,model_b,num_sample,num_outperform_sample')
for ds in os.listdir(ROOT):
    dataset = ds.rsplit('.', 1)[0]
    lang_pair = ds.rsplit('.', 1)[-1]
    ROOT2 = osp.join(ROOT, ds)
    OROOT2 = osp.join(OROOT, ds)
    os.makedirs(OROOT2, exist_ok=True)
    for model_pair in os.listdir(ROOT2):
        model_a, model_b = model_pair.split('-vs-')
        ROOT3 = osp.join(ROOT2, model_pair)
        x,y = read_data (ROOT3)
        print_statistic_data(dataset,lang_pair,model_a,model_b,x,y)
        fig, ax = plt.subplots()
        ax.set_xlabel('broad sample')
        ax.set_ylabel('statistical significance')
        ax.scatter(x,y,marker='+',c='0')
        ax.axhline(y=50, xmin=0, xmax=1, color='b', linewidth='0.5')
        file_name = "{0}.{1}.{2}".format(model_a, model_b, IMG_FORMAT)
        OPATH = osp.join(OROOT2, file_name)
        plt.savefig(OPATH)
        plt.close()
