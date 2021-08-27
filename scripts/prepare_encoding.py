import sys

import re
import sys

print('Prepare encoding ...')

num_samples = 0
invalid = 0

with open(sys.argv[1]) as file1, open(sys.argv[2]) as file2, open(sys.argv[3], 'w') as outfile1, open(sys.argv[4], 'w') as outfile2:
    for line1, line2 in zip(file1, file2):
        num_samples += 1
        line1 = "<pad> <pad> <pad> " + line1 + " <pad> <pad> <pad>"
        line1 = line1.split()
        line2 = "<pad> <pad> <pad> " + line2 + " <pad> <pad> <pad>"
        line2 = line2.split()
        if len(line1) == len(line2):
            for i in range(3, len(line1)-3):
                sample = line2[i-3:i] + ['<token>'] + [line1[i]] + ['</token>'] + line1[i+1:i+4]
                x = re.sub(r'', ' ', '▁'.join(sample)).replace('< p a d >', '<pad>').replace('< t o k e n > ▁', '<token>').replace('▁ < / t o k e n >', '</token>').strip()
                y = re.sub(r'', ' ', line2[i]).strip()
                outfile1.write(x + '\n')
                outfile2.write(y + '\n')
        else:
            invalid += 1
        if num_samples % 1000 == 0:
            print(f'Processed {num_samples} sentences.\r', end='')
    print(f'Processed {num_samples} sentences.')
    print(f'Done. Skipped {invalid} invalid samples.')
