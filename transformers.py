import torch
from pytorch_transformers import *

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')
test_output = torch.tensor([tokenizer.encode('test', add_special_tokens=True)])
print(test_output)
with torch.no_grad():
    last_hidden_states = model(test_output)[0][0][0]
    print(last_hidden_states)
