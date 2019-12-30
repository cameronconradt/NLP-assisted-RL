import pickle
from pytorch_transformers import RobertaModel


roberta_model = RobertaModel.from_pretrained('roberta-base').cuda()
classes = [line.strip() for line in open('data/objects.names', 'rt')]
nlp_dict = pickle.load(open('data/output_condensed.obj', 'rb'))
output = {}

combos = []
for i in classes:
    for j in classes:
        if j is not i:
            combos.append((i, j))

for combo in combos:
    if combo in nlp_dict.keys():
        output.update({combo: roberta_model(nlp_dict[combo].cuda())[0].float()})
pickle.dump(output, open('data/nlp_map_roberta_output.obj', 'wb'))