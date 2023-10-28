from flair.data import MultiCorpus
from flair.datasets import NER_MULTI_WIKINER, NER_MULTI_XTREME
from flair.embeddings import FlairEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer


corpus = MultiCorpus([NER_MULTI_WIKINER()]).downsample(0.1)

# 2. what label do we want to predict?
label_type = 'ner'

# 3. make the label dictionary from the corpus
label_dict = corpus.make_label_dictionary(label_type=label_type)

# 4. initialize embeddings
embedding_types = [
    # we use multilingual Flair embeddings in this task
    FlairEmbeddings('multi-forward'),
    FlairEmbeddings('multi-backward'),
]

embeddings = StackedEmbeddings(embeddings=embedding_types)

# 5. initialize sequence tagger
tagger = SequenceTagger(hidden_size=256,
                        embeddings=embeddings,
                        tag_dictionary=label_dict,
                        tag_type=label_type,
                        use_crf=True)

# 6. initialize trainer
trainer = ModelTrainer(tagger, corpus)

# 7. start training
trainer.train('resources/taggers/example-universal-ner',
              learning_rate=0.2,
              mini_batch_size=32,
              max_epochs=10,
              )

