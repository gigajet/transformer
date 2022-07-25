from enum import unique
import os
ROOT='.'

def procdir (dir):
    for file in os.listdir(dir):
        n_sent=0; sum_token=0; unique_token = set()
        with open(os.path.join(dir, file)) as f_in:
            for line in f_in:
                tokens = line.strip().split()
                n_sent += 1
                sum_token += len(tokens)
                for token in tokens:
                    unique_token.add(token)
        print(os.path.join(dir, file),n_sent,sum_token/n_sent,sum_token,len(unique_token),sep=',')

print('name,n_sent,avg_len,num_nonunique_token,num_unique_token')
for dset in os.listdir(ROOT):
    if os.path.isdir(dset):
        procdir(os.path.join(ROOT, dset))