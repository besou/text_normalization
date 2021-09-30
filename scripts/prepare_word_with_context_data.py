"""
Prepare the training corpus for a word-with-context model.
Create a sample for each token, with a 10-token context window
on the source side.
"""

import sys

with open(sys.argv[1]) as srcfile, open(sys.argv[2]) as tgtfile,\
     open(sys.argv[3], 'w') as src_out, open(sys.argv[4], 'w') as tgt_out:
    for line1, line2 in zip(srcfile, tgtfile):
        line1 = ['<pad>']*5 + line1.split() + ['<pad>']*5
        line2 = ['<pad>']*5 + line2.split() + ['<pad>']*5
        for i in range(len(line1)-10):
            sample = line1.copy()
            sample_target = line2[i+5]
            sample[i+5] = f'<token> {sample[i+5]} </token>'
            sample = sample[i:i+11]
            src_out.write(' '.join(sample)+'\n')
            tgt_out.write(sample_target+'\n')
