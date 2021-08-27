import sys
import random

outdir = sys.argv[3]
src = sys.argv[4]
tgt = sys.argv[5]

corpus = []

with open(sys.argv[1]) as file1, open(sys.argv[2]) as file2:
    for sample in zip(file1, file2):
        corpus.append(sample)

random.shuffle(corpus)

n_valid = 0.08 * len(corpus)
n_test = 0.001 * len(corpus)

src_test = open(f'{outdir}/test.{src}', 'w')
tgt_test = open(f'{outdir}/test.{tgt}', 'w')
src_valid = open(f'{outdir}/valid.{src}', 'w')
tgt_valid = open(f'{outdir}/valid.{tgt}', 'w')
src_train = open(f'{outdir}/train.{src}', 'w')
tgt_train = open(f'{outdir}/train.{tgt}', 'w')

for index, (line1, line2) in enumerate(corpus):
    if index < n_test:
        src_test.write(line1)
        tgt_test.write(line2)
    elif index < n_test + n_valid:
        src_valid.write(line1)
        tgt_valid.write(line2)
    else:
        src_train.write(line1)
        tgt_train.write(line2)

src_test.close()
tgt_test.close()
src_valid.close()
tgt_valid.close()
src_train.close()
tgt_train.close()
