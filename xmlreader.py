import xml.etree.ElementTree as ElementTree
import bz2
import spacy
import csv
import multiprocessing


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


def processType(filename, tag, offset):
    nlp = spacy.load("en_core_web_lg")
    w = csv.writer(open('temp/out_' + str(offset) + '.csv', "w"))
    for elem in getelements(filename, tag, offset):
        if len(elem.text) < nlp.max_length:
            doc = nlp(elem.text)
            noun = ''
            for chunk in doc.noun_chunks:
                if chunk.root.dep_ == 'nsubj':
                    noun = chunk.root.text.lower()
                elif chunk.root.dep_ == 'dobj':
                    key = (noun, chunk.root.text.lower())
                    verb = chunk.root.head.text.lower()
                    w.writerow([key, verb])
                    # if key not in dict:
                    #     dict.update({key: {verb: 1}})
                    # elif verb not in dict[key]:
                    #     dict[key].update({verb: 1})
                    # else:
                    #     dict[key].update({verb: dict[key][verb]+1})


basetype = "{http://www.mediawiki.org/xml/export-0.10/}"
processType('enwiki.xml', basetype + 'text', 0)
