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
import numpy as np
# ========================= / IMPORTS =========================

########## SETTINGS ##########

number_of_file_ids = 5
number_of_topics = 100
show_corpus_words = False
output_to_text_file = True
print_topics = True

########## / SETTINGS ##########

running_document_feature_matrix_list = []
final_document_feature_matrix_list = []
all_files = reuters.fileids()       # Retrieve all file id strings.

for count, file_id in enumerate(range(0, number_of_file_ids)):          # Make an LSI model for specified number of documents.

    ########## CREATING CORPUS ##########

    my_docs = reuters.words(all_files[file_id])           # Retrieve document from corpus.

    tokenized_document = [simple_preprocess(word) for word in my_docs]              # Create a list of all the words. Simple_preprocess for each word in document.
    corpus_dictionary = corpora.Dictionary()                                      # Create an empty dictionary.
    corpus_document = [corpus_dictionary.doc2bow(word, allow_update=True) for word in tokenized_document]       # 6. Generate the corpus using that dictionary for each word in the document.

    ########## / CREATING CORPUS ##########

    ########## PRINTING CORPUS ##########
    # print('mycorpus:')                # Raw number print
    # pprint(mycorpus)
    # print()

    word_counts = [[(corpus_dictionary[identifier], count) for identifier, count in line] for line in corpus_document]
    # print('Word counts:')             # Dictionary print
    # pprint(word_counts)

    ########## / PRINTING CORPUS ##########

    ########## LSI MODEL TRAINING ##########

    if show_corpus_words is True:
        model = LsiModel(corpus=corpus_document, id2word=corpus_dictionary, num_topics=number_of_topics)            # Train the LSI model on the corpus.

    else:
        model = LsiModel(corpus=corpus_document, num_topics=number_of_topics)       # Train the LSI model on the corpus.

    ########## / LSI MODEL TRAINING ##########

        ########## ACCUMULATING DATA FROM LSI MODEL ##########

        # Can now use this model
        all_topics = model.print_topics()                       # Store all topics found into a list.
        running_document_feature_matrix_list.append(all_topics)         # Add topics for this document to running document-feature matrix.

        ########## / ACCUMULATING DATA FROM LSI MODEL ##########

        ########## PRINT TOPICS FOUND IN DOCUMENT ##########

        if print_topics is True:

            print(f"======================= File ID: {all_files[file_id]} - {count + 1} =======================")

            for value in running_document_feature_matrix_list[count]:          # Print LSI model
                print(value)

            # pprint(model.print_topics())          # Print LSI model with pprint for "paragraph view"

            print()
            print()

        ########## / PRINT TOPICS FOUND IN DOCUMENT ##########


########## EXTRACT ONLY FEATURES FOR DOCUMENT-FEATURE MATRIX ##########

# [documents][topics][index or topic]
# index or topic needs to be always 1 to get the topic. Ignore the index.

# Remove the index numbers from the document-feature matrix to have only features.
# for x in range(0, number_of_file_ids):
#       for y in range(0, len(running_document_feature_matrix_list)):
#           final_document_feature_matrix_list.append(running_document_feature_matrix_list[x][y][1])

for document in range(0, len(running_document_feature_matrix_list)):
    final_document_feature_matrix_list.append([topic[1] for topic in running_document_feature_matrix_list[document]])

##### PRINT #####

# for value in final_document_feature_matrix_list:
#     pprint(value)

##### / PRINT #####

########## / EXTRACT ONLY FEATURES FOR DOCUMENT-FEATURE MATRIX ##########

########## OUTPUT DOCUMENT-FEATURE MATRICES TO TEXT FILE ##########
# Output document-feature matrices to text file.

# np.savetxt(fname='lsi_document_feature_matrix.txt', X=final_document_feature_matrix_list)

########## / OUTPUT DOCUMENT-FEATURE MATRICES TO TEXT FILE ##########

final_document_feature_matrix_array = np.array(final_document_feature_matrix_list, dtype=object)
print(final_document_feature_matrix_array.shape)

# document-feature matrices into Procrustes analysis.
# one, two, disparity = procrustes(final_document_feature_matrix_array, final_document_feature_matrix_array)        # Work in progress.


# Display and output to file results of Procrustes analysis.








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

