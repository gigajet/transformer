python sbleu-nltk.py source.txt ground_truth.txt baseline.txt proposal9_32.txt nltk-cmpbetter_proposal9-32_baseline.txt nltk-cmpworse_proposal9-32_baseline.txt
python sbleu-nltk.py source.txt ground_truth.txt baseline.txt proposal10_16.txt nltk-cmpbetter_proposal10-16_baseline.txt nltk-cmpworse_proposal10-16_baseline.txt
python sbleu-sacrebleu.py source.txt ground_truth.txt baseline.txt proposal9_32.txt sacrebleu-cmpbetter_proposal9-32_baseline.txt sacrebleu-cmpworse_proposal9-32_baseline.txt
python sbleu-sacrebleu.py source.txt ground_truth.txt baseline.txt proposal10_16.txt sacrebleu-cmpbetter_proposal10-16_baseline.txt sacrebleu-cmpworse_proposal10-16_baseline.txt
python sbleu-both.py source.txt ground_truth.txt baseline.txt proposal9_32.txt both-cmpbetter_proposal9-32_baseline.txt both-cmpworse_proposal9-32_baseline.txt
python sbleu-both.py source.txt ground_truth.txt baseline.txt proposal10_16.txt both-cmpbetter_proposal10-16_baseline.txt both-cmpworse_proposal10-16_baseline.txt
