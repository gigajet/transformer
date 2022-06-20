import os
import subprocess
from datetime import date

"""
Example:
@param dataset: "europarl-v7"
@param src_lang: "de"
@param tgt_lang: "en"
@param proposal: "9", "10", "baseline"
@param dim_fuzzy: 128, 256. Doesn't matter if proposal=="baseline"
@param dim_model: 128, 512
@param dim_feedforward: 2048, 512...
year, month, day, cuda_device: self-explained.
"""
def fairseq_train_with_default_setting (dataset: str, src_lang:str, tgt_lang:str, 
proposal:str, dim_fuzzy:int, dim_model:int, dim_feedforward: int,
max_epoch: int, patience: int,
year: int, month: int, day: int, cuda_device: int, homedir: str):
    os.environ['CUDA_VISIBLE_DEVICES']=str(cuda_device)
    if proposal == "baseline":
        arch = "nntransformer_default"
        model_name = "{dataset}-{src_lang}-{tgt_lang}.baseline.trainday-{year:04}{month:02}{day:02}".format(
            dataset=dataset,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            year=year,
            month=month,
            day=day
        )
        cmd = """
fairseq-train \
{homedir}/dataset/{dataset}/{src_lang}-{tgt_lang} \
--arch {arch} \\
--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \\
--lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \\
--dropout 0.1 --weight-decay 0.0001 --bpe subword_nmt \\
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \\
--max-epoch {max_epoch} \\
--max-tokens 4096 \\
--patience {patience} \\
--eval-bleu \\
--eval-bleu-args '{{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}}' \\
--eval-bleu-detok moses \\
--eval-bleu-remove-bpe \\
--eval-bleu-print-samples \\
--best-checkpoint-metric bleu --maximize-best-checkpoint-metric \\
--no-epoch-checkpoints \\
--save-dir {homedir}/checkpoints/{model_name} \\
--user-dir {homedir} \\
--log-file {homedir}/logs/{model_name}/{model_name}.log \\
--dim-feedforward {dim_feedforward} \\
--dim-model {dim_model} \\
--max-src-len 4096 \\
--max-tgt-len 4096""".format(
        dataset=dataset,
        src_lang=src_lang, tgt_lang=tgt_lang,
        arch=arch,
        model_name=model_name,
        dim_feedforward=dim_feedforward,
        dim_model = dim_model,
        homedir = homedir,
        max_epoch=max_epoch,
        patience=patience
        )
    else:
        arch = "proposal"+proposal+"_default"
        model_name = "{dataset}-{src_lang}-{tgt_lang}.proposal-{proposal}.dfuzzy-{dfuzzy}.trainday-{year:04}{month:02}{day:02}".format(
            dataset=dataset,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            proposal=proposal,
            dfuzzy=dim_fuzzy,
            year=year,
            month=month,
            day=day
        )
        cmd = """
fairseq-train \
{homedir}/dataset/{dataset}/{src_lang}-{tgt_lang} \
--arch {arch} \\
--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \\
--lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \\
--dropout 0.1 --weight-decay 0.0001 --bpe subword_nmt \\
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \\
--max-epoch {max_epoch} \\
--max-tokens 4096 \\
--patience {patience} \\
--eval-bleu \\
--eval-bleu-args '{{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}}' \\
--eval-bleu-detok moses \\
--eval-bleu-remove-bpe \\
--eval-bleu-print-samples \\
--best-checkpoint-metric bleu --maximize-best-checkpoint-metric \\
--no-epoch-checkpoints \\
--save-dir {homedir}/checkpoints/{model_name} \\
--user-dir {homedir} \\
--log-file {homedir}/logs/{model_name}/{model_name}.log \\
--dim-fuzzy {dim_fuzzy} \\
--dim-feedforward {dim_feedforward} \\
--dim-model {dim_model} \\
--max-src-len 4096 \\
--max-tgt-len 4096""".format(
        dataset=dataset,
        src_lang=src_lang, tgt_lang=tgt_lang,
        arch=arch,
        model_name=model_name,
        dim_fuzzy = dim_fuzzy,
        dim_feedforward=dim_feedforward,
        dim_model = dim_model,
        homedir=homedir,
        max_epoch=max_epoch,
        patience=patience
        )

    os.makedirs("{homedir}/logs/{model_name}".format(model_name=model_name, homedir=homedir), exist_ok=True)
    with open("{homedir}/logs/{model_name}/training_command.log".format(model_name=model_name, homedir=homedir), 'w') as f:
        f.write(cmd)
    # os.system(cmd)
    print("training", model_name, "...")
    exit_code = subprocess.call(cmd, shell=True)
    print("["+model_name+']','fairseq_train returns',exit_code)
    # print('command', cmd)

if __name__ == "__main__":
    # fairseq_train_with_default_setting("europarl_v7","cs","en","9",128,512,2048,2022,6,7,0)
    # fairseq_train_with_default_setting("europarl_v7","cs","en","baseline",128,512,2048,2022,6,7,0)

    # ASSUME FAIRSEQ IS INSTALLED, fairseq-train can be invoked from cwd.
    homedir=os.getcwd()
    dataset = "europarl-v7"
    language_pairs = ["cs-en","fr-en","de-en"]
    proposals = ["baseline","9","10"]
    dim_fuzzies = [16,32,64,96,128,160]
    dim_models = [128]
    dim_feedforwards = [512]
    cuda_device = 0
    max_epoch = 80
    patience = 7
    for language_pair in language_pairs:
        src_lang, tgt_lang = language_pair.split('-')
        for proposal in proposals:
            for dim_model in dim_models:
                for dim_feedforward in dim_feedforwards:
                    if proposal == "baseline":
                        today = date.today()
                        year, month, day = today.year, today.month, today.day
                        fairseq_train_with_default_setting(dataset=dataset,
                            src_lang=src_lang,
                            tgt_lang=tgt_lang,
                            proposal=proposal,
                            dim_fuzzy=0, # irrelevant
                            dim_model=dim_model,
                            dim_feedforward=dim_feedforward,
                            year=year,month=month,day=day,
                            cuda_device=cuda_device,
                            homedir=homedir,
                            max_epoch=max_epoch,
                            patience=patience
                        )
                    else:
                        for dim_fuzzy in dim_fuzzies:
                            today = date.today()
                            year, month, day = today.year, today.month, today.day
                            fairseq_train_with_default_setting(dataset=dataset,
                                src_lang=src_lang,
                                tgt_lang=tgt_lang,
                                proposal=proposal,
                                dim_fuzzy=dim_fuzzy,
                                dim_model=dim_model,
                                dim_feedforward=dim_feedforward,
                                year=year,month=month,day=day,
                                cuda_device=cuda_device,
                                homedir=homedir,
                                max_epoch=max_epoch,
                                patience=patience
                            )
