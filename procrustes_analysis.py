# Author: Cameron Holbrook

# =============== DESCRIPTION ===============
# Purpose: Perform Procrustes analysis on document-feature matrices from two text analysis models.
# General User Process:
# 1. Define number of documents to analyze (this is the rows, N)
# 2. Do text analysis on those documents, and get its respective matrix.
# 3. Do procrustes analysis.
# ===========================================



# ========================= IMPORTS =========================
import nltk
from pprint import pprint               # For neater printing of information
from nltk.corpus import reuters, stopwords         # Import the reuters dataset (from the download function), also stopwords.
from scipy.spatial import procrustes
from gensim.models import LsiModel, LdaModel
from gensim import corpora
import numpy as np
# ========================= / IMPORTS =========================

def print_vectorized_corpus(vectorized_corpus):
    r"""Simplified Vectorized Corpus Printing

    Parameters
    ----------
    vectorized_corpus : Gensim object
        The Gensim object in which each row represents a document, and each
        column represents a latent feature and the document's rating.

    Returns
    -------
    None : N/A
    """

    print('================== Vectorized Corpus ==================')
    for count, value in enumerate(vectorized_corpus):
        print(f'Document {count}: ', end='')
        print(value)
    print('=======================================================')


def latent_semantic_indexing(document_collection):
    r"""Modified and Condensed Latent Semantic Indexing

    Parameters
    ----------
    document_collection : 2D list
        A 2D list in which each row contains a complete Reuters document, and each entry contains one word from it.

    Returns
    -------
    lsi_document_feature_matrix : numpy array
        The "array-like" object needed for Procrustes analysis.
    """

    # ========================= LSI PARAMETERS =========================

    number_of_topics        = 10
    print_vectorization     = True

    print('============== LSI PARAMETERS ==============')
    print(f'{"Number of Topics:":>24} {number_of_topics:>6}')
    print(f'{"Print Vectorization:":>24} {print_vectorization!r:>6}')
    print('============================================')
    print()

    # ========================= / LSI PARAMETERS =========================



    # ========================= TRAIN LSI MODEL =========================

    print('================== Latent Semantic Indexing ==================')

    lsi_dictionary = corpora.Dictionary(document_collection)
    lsi_corpus = [lsi_dictionary.doc2bow(text) for text in document_collection]
    lsi_model = LsiModel(lsi_corpus, id2word=lsi_dictionary, num_topics=number_of_topics)

    lsi_vectorization = lsi_model[lsi_corpus]

    if print_vectorization:
        print_vectorized_corpus(lsi_vectorization)

    # ========================= / TRAIN LSI MODEL =========================



    # ========================= EXTRACT ONLY FEATURES FOR DOCUMENT-FEATURE MATRIX =========================

    # [documents][topics][index or topic]
    # index or topic needs to be always 1 to get the topic. Ignore the index.

    lsi_document_feature_list = []

    for row in lsi_vectorization:
        new_row = []

        for feature in row:
            new_row.append(feature[1])

        lsi_document_feature_list.append(new_row)

    lsi_document_feature_matrix = np.array(lsi_document_feature_list)           # The final matrix conversion for procrustes analysis.

    print('==============================================================')
    print()

    return lsi_document_feature_matrix

    # ========================= / EXTRACT ONLY FEATURES FOR DOCUMENT-FEATURE MATRIX =========================


def select_reuters_documents(number_of_documents):
    r"""Select Reuters Documents

    Parameters
    ----------
    number_of_documents : integer
        An integer defining the number of Reuters documents to analyze (starting from the first document).

    Returns
    -------
    reuters_documents : 2D list
        A 2D list in which each row contains a complete Reuters document, and each entry contains one word from it.
    """

    reuters_corpus = reuters.fileids()  # Retrieve all file id strings.
    reuters_documents = []

    # Aggregate words into documents for proper LSI modeling.
    for file_id in range(0, number_of_documents):
        reuters_documents.append(reuters.words(reuters_corpus[file_id]))

    return reuters_documents


def modified_procrustes(document_feature_matrix_1, document_feature_matrix_2):
    r"""A Modified Procrustes Analysis

    Parameters
    ----------
    document_feature_matrix_1 : numpy array
        The first array-like object to be fed into the Procrustes Analysis function.
    document_feature_matrix_2 : numpy array
        The second array-like object to be fed into the Procrustes Analysis function.

    Returns
    -------
    matrix1 : array_like
        A standardized version of document_feature_matrix_1.
    matrix2 : array_like
        The orientation of document_feature_matrix_2 that best fits document_feature_matrix_1.
        Centered, but not necessarily tr(AAT) = 1.
    disparity : float
        M^2 value that
    """

    # A note from the documentation: the disparity value does not depend on
    # order of input matrices, but the output matrices do. Only the first output matrix
    # is guaranteed to be scaled such that tr(AAT) = 1. (Trace of matrix A times A transposed equals 1).

    # Matrix zero appending pending. (Again, from documentation).

    matrix1, matrix2, disparity = procrustes(document_feature_matrix_1, document_feature_matrix_2)

    return matrix1, matrix2, disparity


if __name__ == '__main__':

    # General User Process:
    # 1. Define number of documents to analyze (this is the rows, N)
    # 2. Do text analysis on those documents, and get its respective matrix.
    # 3. Do procrustes analysis.

    # ================ SETUP ================
    number_of_documents = 10
    document_collection = select_reuters_documents(number_of_documents)
    # ================ SETUP ================

    lsi_document_feature_matrix = latent_semantic_indexing(document_collection)

    matrix1, matrix2, disparity = modified_procrustes(lsi_document_feature_matrix, lsi_document_feature_matrix)

    print()
    print('==================== Matrix 1 ====================')
    print(matrix1)
    print()
    print('==================== Matrix 2 ====================')
    print(matrix2)
    print()
    print('==================== Disparity (Rounded) ====================')
    print(round(disparity, 2))





    # ========================= OUTPUT DOCUMENT-FEATURE MATRICES TO TEXT FILE =========================
    # Output document-feature matrices to text file.

    # np.savetxt(fname='lsi_document_feature_matrix.txt', X=final_document_feature_matrix_list)

    # ========================= / OUTPUT DOCUMENT-FEATURE MATRICES TO TEXT FILE =========================
