num_broad_sample=100
num_bootstrap_sample=1000
src_path=cmp-p9_32-p10_16/source.txt
gt_path=cmp-p9_32-p10_16/ground_truth.txt
model_A_path=cmp-p9_32-p10_16/proposal10_16.txt
model_B_path=cmp-p9_32-p10_16/baseline.txt
output_dir=output-p10_16-vs-baseline

python significance.py -n $num_broad_sample -m $num_bootstrap_sample \
 -o $output_dir $src_path $gt_path $model_A_path $model_B_path
