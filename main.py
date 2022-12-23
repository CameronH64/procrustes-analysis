# Author: Cameron Holbrook
# Purpose: Perform Procrustes analysis on two document-feature matrices
# from multiple document-feature matrices

# ========================= IMPORTS =========================
import numpy as np
import nltk
import gensim.models
from pprint import pprint               # For neater printing of information
from nltk.corpus import reuters, stopwords         # Import the reuters dataset (from the download function), also stopwords.
from scipy.spatial import procrustes
from gensim.models import LsiModel      # LSA/LSI model
from gensim.test.utils import common_dictionary, common_corpus, get_tmpfile
from gensim.utils import simple_preprocess
from gensim import corpora
from sklearn.feature_extraction.text import TfidfVectorizer
# ========================= / IMPORTS =========================

################ CREATING CORPUS ################

my_docs = reuters.words('test/21486')

tokenized_list = [simple_preprocess(doc) for doc in my_docs]      # 4. Create the corpus for the LSI model. First, simple_preprocess for each word in document.
mydict = corpora.Dictionary()           # 5. Create an empty dictionary.
mycorpus = [mydict.doc2bow(doc, allow_update=True) for doc in tokenized_list]       # 6. Generate the corpus using that dictionary for each word in the document.

print('mycorpus:')
# pprint(mycorpus)
print()

# Printing the corpus, but with words instead.
word_counts = [[(mydict[identifier], count) for identifier, count in line] for line in mycorpus]
print('Word counts:')
# pprint(word_counts)


################ / CREATING CORPUS ################



################ LSI MODELING ################

model = LsiModel(corpus=mycorpus, id2word=mydict, num_topics=100)

for value in model.print_topics():
    print(value)

################ / LSI MODELING ################

# Output document-feature matrices



# document-feature matrices into Procrustes analysis.
# one, two = procrustes(model.print_topics(), model2.print_topics())        # Work in progress.


# Display results of Procrustes analysis.
















# ===================== CODE SNIPPETS =====================

################ CREATING DICTIONARY AND CORPUS ################
# (Steps 3 and 5)

# How to create a dictionary from a list of sentences?
# documents = ["The Saudis are preparing a report that will acknowledge that",
#              "Saudi journalist Jamal Khashoggi's death was the result of an",
#              "interrogation that went wrong, one that was intended to lead",
#              "to his abduction from Turkey, according to two sources."]
#
# documents_2 = ["One source says the report will likely conclude that",
#                 "the operation was carried out without clearance and",
#                 "transparency and that those involved will be held",
#                 "responsible. One of the sources acknowledged that the",
#                 "report is still being prepared and cautioned that",
#                 "things could change."]
#
# # Tokenize(split) the sentences into words
# texts = [[text for text in doc.split()] for doc in documents]
#
# # Create dictionary
# dictionary = corpora.Dictionary(texts)



# How to create corpus
# # List with 2 sentences
# my_docs = ["Who let the dogs out?",
#            "Who? Who? Who? Who?"]
#
# # Tokenize the docs
# tokenized_list = [simple_preprocess(doc) for doc in my_docs]
#
# # Create the Corpus
# mydict = corpora.Dictionary()
# mycorpus = [mydict.doc2bow(doc, allow_update=True) for doc in tokenized_list]
# pprint(mycorpus)
# #> [[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)], [(4, 4)]]








# Vertically print topics.
# for value in model.print_topics(-1):
#     print(value)

# A test dictionary for id2word= in LsiModel
# test_dictionary = {
#
#     0: 'sarge',
#     1: 'simmons',
#     2: 'grif',
#     3: 'church',
#     4: 'caboose',
#     5: 'tucker',
#     6: 'tex',
#     7: 'washington',
#     8: 'carolina',
#     9: 'sister',
#     10: 'andy',
#     11: 'doc'
#
# }

# Example of scipy procrustes analysis.
# a = np.array([[1, 3], [1, 2], [1, 1], [2, 1]], 'd')
# b = np.array([[4, -2], [4, -4], [4, -6], [2, -6]], 'd')
# mtx1, mtx2, disparity = procrustes(a, b)
# print(round(disparity))

# First example of LSI Model
# from gensim.test.utils import common_corpus, common_dictionary
# from gensim.models import LsiModel
#
# model = LsiModel(common_corpus, id2word=common_dictionary)                      # Return type: gensim.models.lsimodel.LsiModel
# vectorized_corpus = model[common_corpus]  # vectorize input corpus in BoW format    # Return type: 'gensim.interface.TransformedCorpus'


# Second example of LSI Model
# from gensim.test.utils import common_corpus, common_dictionary, get_tmpfile
# from gensim.models import LsiModel
#
# model = LsiModel(common_corpus[:3], id2word=common_dictionary)  # train model
# vector = model[common_corpus[4]]  # apply model to BoW document
# model.add_documents(common_corpus[4:])  # update model with new documents
# tmp_fname = get_tmpfile("lsi.model")
# model.save(tmp_fname)  # save model
# loaded_model = LsiModel.load(tmp_fname)  # load model

# print("Model:", type(model))
# print("Vector:", type(vector))
# print("tmp_fname", type(tmp_fname))
# print("loaded_model:", type(loaded_model))

