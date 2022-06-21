import argparse
from logging import error, info
import sacrebleu
import os
import random
import tqdm
import shutil

def parse_arguments ():
    parser = argparse.ArgumentParser(description='Calculate statistical significance of BLEU of the dataset divided into broad samples.')
    parser.add_argument('source_path', type=str,
        help='Path to the text file contains source sentences. In the file, one sentence per line.')
    parser.add_argument('ground_truth_path', type=str,
        help='Path to the text file contains ground truth sentences. In the file, one sentence per line. Number of lines should be the same as the source file')
    parser.add_argument('model_A_inference', type=str,
        help='Path to the text file contains corresponding inference sentences of model A. Number of lines should be the same as the ground truth file.')
    parser.add_argument('model_B_inference', type=str,
        help='Path to the text file contains corresponding inference sentences of model B. Number of lines should be the same as the ground truth file.')
    parser.add_argument('-n','--num-broad-samples' , type=int, required=True,
        help='Number of broad samples to divide the test set. Encouraged to divides number of line in the ground truth, otherwise, last lines are dropped to fit. Default: 100. \
            Let n be number of broad samples, then broad sample i consists of examples i, i+n, i+2n...')
    parser.add_argument('-m','--num-bootstrap-samples', type=int,
        default=1000,
        help='Number of bootstrap samples (i.e number of time we sample with replacement) for each broad samples. Default: 1000'
    )
    parser.add_argument('-o','--output-dir', type=str,
        default='./output',
        help="""Output directory. Default: output. All existing files are overwritten. \
            Directory structure: 1-<sig1>.txt 2-<sig2>.txt ... <n-<sign>.txt> where 1.txt 2.txt ... <n.txt> each describes \
            a broad sample. In each of broad sample contains many groups of 5 line:
            <source sentence>
            <ground truth sentence>
            <model A inference>
            <model B inference>
            a blank line.

            The "<sigi>" in "i-<sigi>.txt" is a real number in range [0,100] indicates statistical significance of that broad samples.
            """)
    try:
        args = parser.parse_args()
        return args
    except:
        parser.print_help()
        exit(1)


def check_assumptions(args):

    n_src = 0
    with open(args.ground_truth_path) as f_src:
        for _ in f_src:
            n_src += 1

    n_gt = 0
    with open(args.ground_truth_path) as f_gt:
        for _ in f_gt:
            n_gt += 1
    
    n_a = 0
    with open(args.model_A_inference) as f_a:
        for _ in f_a:
            n_a += 1

    n_b = 0
    with open(args.model_B_inference) as f_b:
        for _ in f_b:
            n_b += 1

    if n_src != n_gt:
        error('Number of sentences in {0} is not the same as in {1}'.format(args.source_path, args.ground_truth_path))
    if n_a != n_gt:
        error('Number of sentences in {0} is not the same as in {1}'.format(args.ground_truth_path, args.model_A_inference))
    if n_b != n_gt:
        error('Number of sentences in {0} is not the same as in {1}'.format(args.ground_truth_path, args.model_B_inference))

"""
Input:
    sample[0]: list of input sentences
    sample[1]: list of ground truth sentences
    sample[2]: list of model A inference sentences
    sample[3]: list of model B inference sentences
Output: A number in range [0,100]
"""
def statistical_significance (sample, num_iteration):
    num_times_a_outperform_b = 0
    for _ in tqdm.tqdm(range(num_iteration), desc='Bootstrapping iteration'):
        boot = random.choices(sample, k=len(sample))
        hypotheses_a = []
        hypotheses_b = []
        references = []
        for _,gt,a,b in boot:
            hypotheses_a.append(a)
            hypotheses_b.append(b)
            references.append([gt])
        bleu_A = sacrebleu.compat.corpus_bleu(hypotheses_a, references).score
        bleu_B = sacrebleu.compat.corpus_bleu(hypotheses_b, references).score
        if bleu_A > bleu_B:
            num_times_a_outperform_b += 1
    return 100.0 * num_times_a_outperform_b / num_iteration

def run(args):
    # args.{ground_truth_path, model_A_inference, model_B_inference, num_broad_samples, num_bootstrap_samples}
    n = args.num_broad_samples
    i = 0 # 0..n-1
    broad_sample = {}
    for j in range(n):
        broad_sample[j] = [] # each element of this list is another list [src, gt, model_A, model_B]

    with open(args.source_path) as f_src, \
        open(args.ground_truth_path) as f_gt, \
        open(args.model_A_inference) as f_a, \
        open(args.model_B_inference) as f_b:
        for line_src in f_src:
            line_src = line_src.strip()
            line_gt, line_a, line_b = f_gt.readline().strip(), f_a.readline().strip(), f_b.readline().strip()
            broad_sample[i].append([line_src, line_gt, line_a, line_b])
            i = (i+1) % n
    
    output_dir = args.output_dir
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    result = []
    for i in tqdm.tqdm(range(n), 'Broad sample'):
        bs = broad_sample[i]
        sig = statistical_significance (bs, args.num_bootstrap_samples)
        with open(os.path.join(output_dir, "{0}-{1}.txt".format(i, sig)), 'w') as f_out:
            num_sent = len(bs)
            info('Broad sample {0} has {1} sentences with statistical significance {2}.'.format(i,num_sent,sig))
            f_out.writelines(['\n'.join([*bs[j],'\n']) for j in range(num_sent)])

if __name__=="__main__":
    args = parse_arguments()
    check_assumptions(args)
    run(args)
