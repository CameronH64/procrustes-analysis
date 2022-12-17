# Author: Cameron Holbrook
# Purpose: Perform Procrustes analysis on two document-feature matrices
# from multiple document-feature matrices

# ===================== DOWNLOAD REUTERS CORPORA =====================
# Use this code (uncomment it) to download corpora.
# nltk.download()

# ===================== BRIEF OVERVIEW OF REUTERS CORPORA =====================
# 90 categories
# 10788 new documents, 1.3 million words.

# ===================== IMPORTANT NOTE ON PROCRUSTES FUNCTION =====================
# This function was not designed to handle datasets with different numbers of datapoints (rows).
# If two data sets have different dimensionality (different number of columns), simply add columns of zeros to the smaller of the two.

# Imports
import numpy as np
import nltk
import pprint                           # For neater printing of information
from nltk.corpus import reuters         # Import the reuters dataset (from the download function)
from scipy.spatial import procrustes
from gensim.models import LsiModel      # LSA/LSI model
from gensim.test.utils import common_dictionary, common_corpus
from gensim.models import LsiModel

# Model Training
model = LsiModel(common_corpus, id2word=common_dictionary)
vectorized_corpus = model[common_corpus]  # vectorize input corpus in BoW format

pprint.pprint(common_corpus)


# Output document-feature matrices



# document-feature matrices into Procrustes analysis.



# Display results of Procrustes analysis.
















# ===================== CODE SNIPPETS =====================

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


