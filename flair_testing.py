from flair.data import Sentence
from flair.nn import Classifier

sentence = Sentence('YOUR PROMPT')

# load the NER tagger
tagger = Classifier.load('resources/taggers/example-universal-ner/best-model.pt')

# run NER over sentence
tagger.predict(sentence)

# print the sentence with all annotations
print(sentence)