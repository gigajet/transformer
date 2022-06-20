import os
import subprocess

os.makedirs("eval", exist_ok=True)
for model_name in os.listdir("checkpoints"):
    language_pair = model_name.split('.')[0][-5::]
    cmd = """
fairseq-generate dataset/europarl-v7/{language_pair} \
    --path checkpoints/{model_name}/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe \
    --user-dir .""".format(
        language_pair=language_pair,
        model_name=model_name
        )
    os.makedirs(os.path.join("eval",model_name), exist_ok=True)
    with open("eval/{model_name}/eval_command.log".format(model_name=model_name), 'w') as f:
        f.write(cmd)
    exit_code = subprocess.call(cmd, shell=True)
    print("["+model_name+']','fairseq_generate returns',exit_code)
    if exit_code != 0:
        print('Unsuccessful')
        break
