import sys

import re
import sys

print('Prepare encoding ...')

num_samples = 0
invalid = 0

with open(sys.argv[1]) as file1, open(sys.argv[2]) as file2, open(sys.argv[3], 'w') as outfile1, open(sys.argv[4], 'w') as outfile2:
    for line1, line2 in zip(file1, file2):
        num_samples += 1
        if line1.count(' ') == line2.count(' '):
            sample1 = re.sub('(.)', r'\1 ', line1.strip())
            sample2 = re.sub('(.)', r'\1 ', line2.strip())
            outfile1.write(sample1+'\n')
            outfile2.write(sample2+'\n')
        else:
            invalid += 1
            print(line1.strip())
            print(line2)
        if num_samples % 1000 == 0:
            print(f'Processed {num_samples} sentences.\r', end='')
    print(f'Processed {num_samples} sentences.')
    print(f'Done. Skipped {invalid} invalid samples.')
