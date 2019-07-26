import xml.etree.ElementTree as ElementTree
import bz2


def getelements(filename_or_file, tag):
    """Yield *tag* elements from *filename_or_file* xml incrementaly."""
    context = iter(ElementTree.iterparse(filename_or_file, events=('start', 'end')))
    _, root = next(context) # get root element
    for event, elem in context:
        if event == 'end' and elem.tag == tag:
            yield elem
            root.clear() # free memory


basetype = "{http://www.mediawiki.org/xml/export-0.10/}"
infile = bz2.BZ2File('enwiki.xml.bz2', "r")
outfile = bz2.open('enwiki-parsed.xml.bz2', "wt")

for elem in getelements(infile, basetype + 'text'):
    if type(elem.text) == str:
        if 'redirect' not in elem.text.lower():
            outfile.write(elem.text)

infile.close()
outfile.close()
