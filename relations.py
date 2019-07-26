import os
from snorkel.parser import TextDocPreprocessor
from snorkel import SnorkelSession
from snorkel.parser.spacy_parser import Spacy
from snorkel.parser import CorpusParser
from snorkel.models import Document, Sentence
from snorkel.models import candidate_subclass
from snorkel.candidates import Ngrams, CandidateExtractor
from snorkel.matchers import MiscMatcher


# TO USE A DATABASE OTHER THAN SQLITE, USE THIS LINE
# Note that this is necessary for parallel execution amongst other things...
# os.environ['SNORKELDB'] = 'postgres:///snorkel-intro'

session = SnorkelSession()

doc_preprocessor = TextDocPreprocessor('parsed/')

corpus_parser = CorpusParser(parser=Spacy())
corpus_parser.apply(doc_preprocessor)

print("Documents:", session.query(Document).count())
print("Sentences:", session.query(Sentence).count())


Spouse = candidate_subclass('Action', ['subject', 'verb'])

ngrams = Ngrams(n_max=7)
misc_matcher = MiscMatcher()
cand_extractor = CandidateExtractor(Spouse, [ngrams, ngrams], [misc_matcher, misc_matcher])

docs = session.query(Document).order_by(Document.name).all()

train_sents = set()
dev_sents   = set()
test_sents  = set()

for i, doc in enumerate(docs):
    for s in doc.sentences:
        if i % 10 == 8:
            dev_sents.add(s)
        elif i % 10 == 9:
            test_sents.add(s)
        else:
            train_sents.add(s)

for i, sents in enumerate([train_sents, dev_sents, test_sents]):
    cand_extractor.apply(sents, split=i)
    print("Number of candidates:", session.query(Spouse).filter(Spouse.split == i).count())