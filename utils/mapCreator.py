import pickle
import csv
import torch
from pytorch_transformers import *
from bz2 import BZ2File

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')

map = pickle.load(open('output.obj', 'rb'))
# w = csv.writer(open("output.csv", "wt"))
# max = csv.writer(open('output_condensed.csv', 'wt'))

condensed_map = {}
for key, val in map.items():
    if key != 'count':
        if len(val.keys()) > 1:
            # w.writerow([key, val])
            maxkey = ''
            maxval = 0
            for littlekey, littleval in val.items():
                if littleval > maxval:
                    maxkey = littlekey
                    maxval = littleval
            if maxval > 1:
                # max.writerow([key, maxkey])
                token_max = torch.tensor([tokenizer.encode(maxkey, add_special_tokens=True)])
                condensed_map.update({key: token_max})

max_obj = open('output_condensed.obj', 'wb')
pickle.dump(condensed_map, max_obj)

