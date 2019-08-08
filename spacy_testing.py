import spacy
import csv
import os


def most_similar(word):
    queries = [w for w in word.vocab if w.is_lower == word.is_lower and w.prob >= -15]
    by_similarity = sorted(queries, key=lambda w: word.similarity(w), reverse=True)
    return by_similarity[:10]


spacy.require_gpu()
nlp = spacy.load("en_core_web_lg")
map = {}

for file in os.listdir('parsed'):
    doc = nlp(open('parsed/' + file, 'r').read())
    noun = ''
    for chunk in doc.noun_chunks:
        if chunk.root.dep_ == 'nsubj':
            noun = chunk.root.text.lower()
        elif chunk.root.dep_ == 'dobj':
            key = noun
            # key = (noun, chunk.root.text.lower())
            verb = chunk.root.head.text.lower()
            if key not in map:
                map.update({key: {verb: 1}})
            elif verb not in map[key]:
                map[key].update({verb: 1})
            else:
                count = map[key][verb]
                map[key].update({verb: count+1})
w = csv.writer(open("output_noun.csv", "w"))
for key, val in map.items():
    if len(val.keys()) > 1:
        w.writerow([key, val])