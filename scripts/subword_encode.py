import sys
import sentencepiece as spm

source = sys.argv[1]
target = sys.argv[2]
outdir = sys.argv[3]
training_file = sys.argv[4]

spm.SentencePieceTrainer.train(input=f'{outdir}/{training_file}', model_prefix='subword', vocab_size=1000)

sp = spm.SentencePieceProcessor(model_file='subword.model')

partitions = ['train', 'valid', 'test']
langs = [source, target]

for partition in partitions:
	for lang in langs:
		with open(f'{outdir}/{partition}.{lang}') as infile, open(f'{outdir}/{partition}.sub.{lang}', 'w') as outfile:
			for line in infile:
				outfile.write(' '.join(sp.encode(line, out_type=str))+'\n')
