# Author: Cameron Holbrook
# Purpose: Perform Procrustes analysis on two document-feature matrices
# from multiple document-feature matrices

# ========================= IMPORTS =========================
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
import numpy as np
# ========================= / IMPORTS =========================

def print_dictionary(dictionary):
    for value, value2 in dictionary.items():
        print(value2)

def print_vectorized_corpus(vectorized_corpus):

    print('Vectorized Corpus: ')

    for count, value in enumerate(vectorized_corpus):
        print(f'Document {count}: ', end='')
        print(value)


def main():

    # ========================= SETTINGS =========================

    number_of_documents = 10
    number_of_topics = 10

    show_corpus_words = False
    output_to_text_file = True
    print_topics = True
    print_vectorization = False

    print('============ SETTINGS ============')
    print(f'{"Number of Documents:":>24} {number_of_documents:>6}')
    print(f'{"Number of Topics:":>24} {number_of_topics:>6}')
    print(f'{"Show Corpus Words:":>24} {show_corpus_words!r:>6}')
    print(f'{"Output Text to File:":>24} {output_to_text_file!r:>6}')
    print(f'{"Print Topics:":>24} {print_topics!r:>6}')
    print(f'{"Print Vectorization:":>24} {print_vectorization!r:>6}')
    print('==================================')
    print()

    # ========================= / SETTINGS =========================



    # ========================= SELECTING DOCUMENTS =========================

    reuters_corpus = reuters.fileids()       # Retrieve all file id strings.
    document_collection = []

    # Aggregate words into documents for proper LSI modeling.
    for file_id in range(0, number_of_documents):
        document_collection.append(reuters.words(reuters_corpus[file_id]))

    # pprint(document_collection)

    # ========================= / SELECTING DOCUMENTS =========================



    # ========================= TRAINING LSI MODEL =========================

    lsi_dictionary = corpora.Dictionary(document_collection)
    lsi_corpus = [lsi_dictionary.doc2bow(text) for text in document_collection]
    lsi_model = LsiModel(lsi_corpus, id2word=lsi_dictionary, num_topics=number_of_topics)

    lsi_vectorization = lsi_model[lsi_corpus]
    print_vectorized_corpus(lsi_vectorization)
    print()

    # ========================= / TRAINING LSI MODEL =========================



    # ========================= EXTRACT ONLY FEATURES FOR DOCUMENT-FEATURE MATRIX =========================

    # [documents][topics][index or topic]
    # index or topic needs to be always 1 to get the topic. Ignore the index.

    print('Extracting Features: ')

    document_feature_list = []

    for row in lsi_vectorization:
        new_row = []

        for feature in row:
            new_row.append(feature[1])

        document_feature_list.append(new_row)

    # ========== PRINT ==========
    for value in document_feature_list:
        pprint(value)
    # ========== PRINT ==========

    document_feature_matrix = np.array(document_feature_list)

    # ========================= / EXTRACT ONLY FEATURES FOR DOCUMENT-FEATURE MATRIX =========================



    # ========================= OUTPUT DOCUMENT-FEATURE MATRICES TO TEXT FILE =========================
    # Output document-feature matrices to text file.

    # np.savetxt(fname='lsi_document_feature_matrix.txt', X=final_document_feature_matrix_list)

    # ========================= / OUTPUT DOCUMENT-FEATURE MATRICES TO TEXT FILE =========================

    # final_document_feature_matrix_array = np.array(final_document_feature_matrix_list, dtype=object)
    # print(final_document_feature_matrix_array.shape)

    # document-feature matrices into Procrustes analysis.
    # one, two, disparity = procrustes(final_document_feature_matrix_array, final_document_feature_matrix_array)        # Work in progress.


    # ========================= LSI PROCRUSTES ANALYSIS =========================

    matrix1, matrix2, disparity = procrustes(document_feature_matrix, document_feature_matrix)

    print('Matrix 1:', matrix1)
    print('Matrix 2:', matrix2)
    print('Disparity:', disparity)

    # ========================= / LSI PROCRUSTES ANALYSIS =========================


if __name__ == '__main__':
    main()




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

