import os
import subprocess
import sys 

if __name__=="__main__":
    if len(sys.argv) < 4:
        print("usage:",sys.argv[0],"path/to/europarl-v7/dataset/dir","path/to/checkpoints/dir","path/to/user/dir", "[path/to/output/dir]")
        print()
        print('THIS SCRIPT IS REQUIRED TO RUN AT ROOT OF GITHUB FOLDER')
        print("where dataset dir contains 3 subfolders 'cs-en','de-en','fr-en', and checkpoints dir contains <model_name>'s as subfolder.")
        print("inside the user dir is directory 'models', which inside is content of github.com/gigajet/transformer, include fixed")
        print("if output dir not specified, it is default as 'eval'. Each subfolder inside output folder contains respective log.")
        exit(1)
    dataset_dir = sys.argv[1]
    checkpoint_dir = sys.argv[2]
    user_dir = sys.argv[3]
    output_dir = sys.argv[4] if len(sys.argv)>4 else 'eval'
    os.makedirs(output_dir, exist_ok=True)
    for model_name in os.listdir(checkpoint_dir):
        language_pair = model_name.split('.')[0][-5::]
        cmd = """\
fairseq-generate {full_dataset_dir} \\
    --path {full_checkpoint_dir} \\
    --log-file {full_logfile_dir} \\
    --batch-size 128 --beam 5 --remove-bpe \\
    --user-dir {user_dir}""".format(
            full_dataset_dir=os.path.join(dataset_dir, language_pair),
            full_checkpoint_dir=os.path.join(checkpoint_dir,model_name,"checkpoint_best.pt"),
            full_logfile_dir=os.path.join(output_dir,model_name,model_name+".txt"),
            user_dir=user_dir,
            )
        os.makedirs(os.path.join(output_dir,model_name), exist_ok=True)
        with open(os.path.join(output_dir,model_name,"eval_command.log"), 'w') as f_cmd, \
            open(os.path.join(output_dir,model_name,"stdout-"+model_name+".log"), 'w') as f_out, \
            open(os.path.join(output_dir,model_name,"stderr-"+model_name+".log"), 'w') as f_err:
            f_cmd.write(cmd)
            exit_code = subprocess.call(cmd, shell=True, stdout=f_out, stderr=f_err)
            print("["+model_name+']','fairseq_generate returns',exit_code)
            if exit_code != 0:
                print('Unsuccessful')
                break
