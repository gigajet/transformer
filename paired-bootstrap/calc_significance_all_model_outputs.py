# PUT THIS SCRIPT same level as 'model_output', inside is the subfolders of {dset}/{ground_truth,source,...}.txt
# For each model, compare them to the baseline, output to 'significance_output'/{dset}/{model_name}-vs-baseline.txt
import subprocess
import os
ROOT = 'model_output'
OUT_ROOT = 'significance_output'
NUM_BROAD_SAMPLES = 100
NUM_BOOTSTRAP_EACH_BROAD_SAMPLE = 5000
if __name__=="__main__":
    for dset in os.listdir(ROOT):
        for model in os.listdir(os.path.join(ROOT,dset)):
            if model.startswith('baseline') or model.startswith('ground_truth'):
                continue
            src_path = os.path.join(ROOT,dset,'source.txt')
            gt_path = os.path.join(ROOT,dset,'ground_truth.txt')
            model_A_path = os.path.join(ROOT,dset,model)
            model_B_path = os.path.join(ROOT,dset,'baseline.txt')
            output_dir = os.path.join(OUT_ROOT, dset, model.split('.')[0]+"-vs-baseline")

            cmd = """\
python significance.py -n {0} -m {1} \
-o {out} {src} {gt} {a} {b}""".format(
    NUM_BROAD_SAMPLES, NUM_BOOTSTRAP_EACH_BROAD_SAMPLE,
    out=output_dir, src=src_path, gt=gt_path, a=model_A_path, b=model_B_path
)
            exitcode = subprocess.call(cmd, shell=True)
            print('['+output_dir+']','exits',exitcode)
            if exitcode != 0:
                exit(1)

