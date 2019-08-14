import xml.etree.ElementTree as ElementTree
import spacy
import csv
import string
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
import pickle
import os


def getelements(filename_or_file, tag, offset):
    """Yield *tag* elements from *filename_or_file* xml incrementaly."""
    context = iter(ElementTree.iterparse(filename_or_file, events=('start', 'end')))
    _, root = next(context) # get root element
    i = 0
    for event, elem in context:
        if event == 'end' and elem.tag == tag:
            if i >= offset:
                yield elem
                root.clear() # free memory
            i += 1


def processType(filename, tag, offset, dict):
    nlp = spacy.load("en_core_web_lg")
    # w = csv.writer(open('output' + str(offset) + '.csv', "w"))
    lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)
    count = offset
    for elem in getelements(filename, tag, offset):
        if count % 100 == 0:
            pickle.dump(dict, open('checkpoint.obj', 'wb'))
        if elem.text is not None:
            if len(elem.text) < nlp.max_length and elem.text != '':
                text = elem.text
                text = text.translate(text.maketrans("", "", string.digits))
                text = text.translate(text.maketrans("", "", string.punctuation))
                doc = nlp(text)
                noun = ''
                for chunk in doc.noun_chunks:
                    if chunk.root.dep_ == 'nsubj':
                        noun = lemmatizer.noun(chunk.root.text.lower())[0]
                    elif chunk.root.dep_ == 'dobj' and noun != '':
                        key = (noun, lemmatizer.noun(chunk.root.text.lower())[0])
                        verb = lemmatizer.verb(chunk.root.head.text.lower())[0]
                        # w.writerow([key, verb])
                        if key not in dict:
                            dict.update({key: {verb: 1}})
                        elif verb not in dict[key]:
                            dict[key].update({verb: 1})
                        else:
                            dict[key].update({verb: dict[key][verb]+1})
        count += 1
        dict.update({'count': count})


basetype = "{http://www.mediawiki.org/xml/export-0.10/}"
if os.path.exists('checkpoint.obj'):
    dict = pickle.load(open('checkpoint.obj', 'rb'))
    if 'count' in dict:
        processType('enwiki.xml', basetype + 'text', dict['count'], dict)
    else:
        processType('enwiki.xml', basetype + 'text', 0, dict)
else:
    dict = {}
    processType('enwiki.xml', basetype + 'text', 0, dict)

dict.pop('count')
file = open("output.obj", "w")
pickle.dump(dict, file)

w = csv.writer(open("output.csv", "w"))
for key, val in dict.items():
    w.writerow([key, val])
