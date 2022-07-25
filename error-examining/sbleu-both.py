import nltk
import sys
import sacrebleu

if len(sys.argv) < 6:
    print('usage:',sys.argv[0],'SOURCE REF_FILE BASELINE_FILE PROPOSAL_FILE BETTER_OUTPUT WORSE_OUTPUT')
    exit(0)

MAGIC_NUMBER_SACREBLEU = 10
MAGIC_NUMBER_NLTK_SBLEU = 0.1
SOURCE_FILE = sys.argv[1]
REF_FILE = sys.argv[2]
BASELINE_FILE = sys.argv[3]
PROPOSAL_FILE = sys.argv[4]
BETTER_OUTPUT = sys.argv[5]
WORSE_OUTPUT = sys.argv[6]

with open(SOURCE_FILE, 'r') as fsource:
    with open(REF_FILE, 'r') as fref:
        with open(BASELINE_FILE, 'r') as fbaseline:
            with open(PROPOSAL_FILE, 'r') as fproposal:
                with open(BETTER_OUTPUT, 'w') as fbetter:
                    with open(WORSE_OUTPUT, 'w') as fworse:
                        for line_source in fsource:
                            line_ref = fref.readline()
                            line_proposal = fproposal.readline()
                            line_baseline = fbaseline.readline()
                            reference_output = line_ref.rstrip().split()
                            proposal_output = line_proposal.rstrip().split()
                            baseline_output = line_baseline.rstrip().split()

                            # sbleu_prop = nltk.translate.bleu_score.sentence_bleu([reference_output], proposal_output, weights = [0.25, 0.25, 0.25, 0.25])
                            # sbleu_base = nltk.translate.bleu_score.sentence_bleu([reference_output], baseline_output, weights = [0.25, 0.25, 0.25, 0.25])
                            sbleu_prop = nltk.translate.bleu_score.sentence_bleu([reference_output], proposal_output, weights = [1])
                            sbleu_base = nltk.translate.bleu_score.sentence_bleu([reference_output], baseline_output, weights = [1])
                            sacre_sbleu_prop = sacrebleu.compat.sentence_bleu(line_proposal.rstrip(), [line_ref.strip()]).score
                            sacre_sbleu_base = sacrebleu.compat.sentence_bleu(line_baseline.rstrip(), [line_ref.strip()]).score

                            # print(sbleu_prop, sacre_sbleu_prop, sbleu_base, sacre_sbleu_base)

                            # if sbleu_prop > sbleu_base:
                            #     fbetter.writelines([line_source,line_ref, line_baseline, line_proposal, '\n'])
                            # if sbleu_prop + MAGIC_NUMBER_NLTK_SBLEU < sbleu_base:
                            #     fworse.writelines([line_source, line_ref, line_baseline, line_proposal, '\n'])

                            if sacre_sbleu_prop > sacre_sbleu_base and sbleu_prop > sbleu_base:
                                fbetter.writelines([line_source,line_ref, line_baseline, line_proposal, '\n'])
                            if sacre_sbleu_prop + MAGIC_NUMBER_SACREBLEU < sacre_sbleu_base and sbleu_prop + MAGIC_NUMBER_NLTK_SBLEU < sbleu_base:
                                fworse.writelines([line_source, line_ref, line_baseline, line_proposal, '\n'])
