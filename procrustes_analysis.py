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
from datetime import datetime
# ========================= / IMPORTS =========================

def print_vectorized_corpus(vectorized_corpus, model):
    r"""Simplified Vectorized Corpus Printing

    Parameters
    ----------
    vectorized_corpus : Gensim object
        The Gensim object in which each row represents a document, and each
        column represents a latent feature and the document's rating.
    model : string
        Shows which model's vectorized corpus is being used.

    Returns
    -------
    None : N/A
    """

    print(f'================== {model} Vectorized Corpus ==================')
    for count, value in enumerate(vectorized_corpus):
        print(f'Document {count}: ', end='')
        print(value)
    print('===========================================================')


def print_modified_procrustes(matrix1, matrix2, disparity, print_matrix_1_toggle=False, print_matrix_2_toggle=False, print_disparity_toggle=False):
    r"""Simplified Vectorized Corpus Printing

    Parameters
    ----------
    matrix1 : numpy array
        The first document-feature matrix.
    matrix2 : numpy array
        The second document-feature matrix.
    disparity : float
        The M^2 value that denotes disparity.
    print_matrix_1_toggle : boolean
        If true, print the first matrix.
    print_matrix_2_toggle : boolean
        If true, print the second matrix.
    print_disparity_toggle : boolean
        If true, print the disparity value.

    Returns
    -------
    None : N/A
    """

    print()

    if print_matrix_1_toggle:
        print('==================== Matrix 1 ====================')
        print(matrix1)
        print()
    if print_matrix_2_toggle:
        print('==================== Matrix 2 ====================')
        print(matrix2)
        print()
    if print_disparity_toggle:
        print('==================== Disparity (Rounded) ====================')
        print(round(disparity, 2))
        print()


def print_settings(number_of_documents, number_of_topics, print_settings_toggle=False):
    r"""Print Settings of Text Analysis

    Parameters
    ----------
    number_of_documents : integer
        The number of documents analyzed.
    number_of_topics : integer
        The number of topics found.
    print_settings_toggle : boolean
        If true, print these settings.

    Returns
    -------
    None : N/A
    """
    if print_settings_toggle:
        print('================= SETTINGS =================')
        print(f'{"Number of Documents:":>24} {number_of_documents:>6}')
        print(f'{"Number of Topics:":>24} {number_of_topics:>6}')
        print('============================================')
        print()


def latent_semantic_indexing(document_collection, number_of_topics, print_vectorization_toggle=False):
    r"""Modified and Condensed Latent Semantic Indexing

    Parameters
    ----------
    document_collection : 2D list
        A 2D list in which each row is a complete Reuters document, and each entry contains one word from it.
    number_of_topics : integer
        The number of topics to do LSI on.
    print_vectorization_toggle : boolean
        Default: False. If true, print the LSI vectorization.

    Returns
    -------
    lsi_document_feature_matrix : numpy array
        The "array-like" object needed for Procrustes analysis.
    """

    # ========================= TRAIN LSI MODEL =========================

    lsi_dictionary = corpora.Dictionary(document_collection)
    lsi_corpus = [lsi_dictionary.doc2bow(text) for text in document_collection]
    lsi_model = LsiModel(lsi_corpus, id2word=lsi_dictionary, num_topics=number_of_topics)

    lsi_vectorization = lsi_model[lsi_corpus]

    if print_vectorization_toggle:
        print_vectorized_corpus(lsi_vectorization, 'LSI')

    # ========================= / TRAIN LSI MODEL =========================



    # ========================= EXTRACT ONLY FEATURES FOR DOCUMENT-FEATURE MATRIX =========================

    # [documents][topics][index or topic]
    # index or topic needs to be always 1 to get the topic. Ignore the index.

    lsi_document_feature_matrix = np.zeros((number_of_documents, number_of_topics))

    for document in range(len(lsi_vectorization)):

        for topic_entry in lsi_vectorization[document]:      # Do something with the column value.

            topic_placement = topic_entry[0]
            lsi_document_feature_matrix[document][topic_placement] = topic_entry[1]

    # print('LSI document-feature matrix:', lsi_document_feature_matrix)

    # ========================= / EXTRACT ONLY FEATURES FOR DOCUMENT-FEATURE MATRIX =========================



    # ========================= SAVE LSI DOCUMENT-FEATURE MATRIX TO A TEXT FILE =========================

    date_now = datetime.today()
    current_date = date_now.strftime("%Y.%m.%d")

    time_now = datetime.now()
    current_time = time_now.strftime("%H.%M.%S")

    # Save the document-feature matrix (which is a numpy array) to a text file.
    np.savetxt(f'distributional_semantic_model_outputs/lsi_document_feature_matrix_{current_date}T{current_time}Z.txt', X=lsi_document_feature_matrix)
    # fmt='%.2f' can format the output per entry.
    # May add a dynamic time appending feature the name above.

    # ========================= / SAVE LSI DOCUMENT-FEATURE MATRIX TO A TEXT FILE =========================

    return lsi_document_feature_matrix


def latent_dirichlet_indexing(document_collection, number_of_topics, print_vectorization_toggle=False):
    r"""Modified and Condensed Latent Dirichlet Allocation

    Parameters
    ----------
    document_collection : 2D list
        A 2D list in which each row is a complete Reuters document, and each entry contains one word from it.
    number_of_topics : integer
        The number of topics to do LDA on.
    print_vectorization_toggle : boolean
        Default: False. If true, print the LDA vectorization.

    Returns
    -------
    lda_document_feature_matrix : numpy array
        The "array-like" object needed for Procrustes analysis.
    """

    # ========================= LSI PARAMETERS =========================

    # ========================= / LSI PARAMETERS =========================



    # ========================= TRAIN LSI MODEL =========================

    lda_dictionary = corpora.Dictionary(document_collection)
    lda_corpus = [lda_dictionary.doc2bow(text) for text in document_collection]
    lda_model = LdaModel(lda_corpus, id2word=lda_dictionary, num_topics=number_of_topics)

    lda_vectorization = lda_model[lda_corpus]

    if print_vectorization_toggle:
        print_vectorized_corpus(lda_vectorization, 'LDA')

    # ========================= / TRAIN LSI MODEL =========================



    # ========================= EXTRACT ONLY FEATURES FOR DOCUMENT-FEATURE MATRIX =========================

    # [documents][topics][index or topic]
    # index or topic needs to be always 1 to get the topic. Ignore the index.

    lda_document_feature_matrix = np.zeros((number_of_documents, number_of_topics))

    # Every single entry in the vectorized corpus has the index value need to place into the document-feature matrix (a numpy array).

    for document in range(len(lda_vectorization)):

        for topic_entry in lda_vectorization[document]:      # Do something with the column value.

            topic_placement = topic_entry[0]
            lda_document_feature_matrix[document][topic_placement] = topic_entry[1]

    # print('LDA document-feature matrix:', lda_document_feature_matrix)

    # ========================= / EXTRACT ONLY FEATURES FOR DOCUMENT-FEATURE MATRIX =========================



    # ========================= SAVE LDA DOCUMENT-FEATURE MATRIX TO A TEXT FILE =========================

    date_now = datetime.today()
    current_date = date_now.strftime("%Y.%m.%d")

    time_now = datetime.now()
    current_time = time_now.strftime("%H.%M.%S")

    # Save the document-feature matrix (which is a numpy array) to a text file.
    np.savetxt(f'distributional_semantic_model_outputs/lda_document_feature_matrix_{current_date}T{current_time}Z.txt', X=lda_document_feature_matrix)
    # fmt='%.2f' can format the output per entry.
    # May add a dynamic time appending feature the name above.

    # ========================= / SAVE LDA DOCUMENT-FEATURE MATRIX TO A TEXT FILE =========================

    return lda_document_feature_matrix


def select_reuters_documents(number_of_documents):
    r"""Select Reuters Documents

    Parameters
    ----------
    number_of_documents : integer
        Defines the number of Reuters documents to analyze (starting from the first document).

    Returns
    -------
    reuters_documents : 2D list
        Contains a complete Reuters document, where each row is a document, and each entry in each row is a word.
    """

    reuters_corpus = reuters.fileids()  # Retrieve all file id strings.
    reuters_documents = []

    # Aggregate words into documents for proper LSI modeling.
    for file_id in range(0, number_of_documents):
        reuters_documents.append(reuters.words(reuters_corpus[file_id]))

    return reuters_documents


def modified_procrustes(document_feature_matrix_1, document_feature_matrix_2, number_of_documents, number_of_topics):
    r"""A Modified Procrustes Analysis

    Parameters
    ----------
    document_feature_matrix_1 : numpy array
        The first array-like object to be fed into the Procrustes Analysis function.
    document_feature_matrix_2 : numpy array
        The second array-like object to be fed into the Procrustes Analysis function.
    number_of_documents : integer
        Integer denoting number of documents, used for the appending zeros function.
    number_of_topics : integer
        Integer denoting number of topics, used for the appending zeros function.

    Returns
    -------
    matrix1 : array_like
        A standardized version of document_feature_matrix_1.
    matrix2 : array_like
        The orientation of document_feature_matrix_2 that best fits document_feature_matrix_1.
        Centered, but not necessarily tr(AAT) = 1.
    disparity : float
        M^2 value that denotes disparity between input matrices.
    """

    # A note from the documentation: the disparity value does not depend on
    # order of input matrices, but the output matrices do. Only the first output matrix
    # is guaranteed to be scaled such that tr(AAT) = 1. (Trace of matrix A times A transposed equals 1).

    # Matrix zero appending function pending. (Again, from documentation).
    # This function will work on any document-feature matrix inputted.

    matrix1, matrix2, disparity = procrustes(document_feature_matrix_1, document_feature_matrix_2)

    date_now = datetime.today()
    current_date = date_now.strftime("%Y.%m.%d")

    time_now = datetime.now()
    current_time = time_now.strftime("%H.%M.%S")

    with open(f'procrustes_analysis_outputs/procrustes_analysis_{current_date}T{current_time}Z.txt', 'w') as f:
        f.write('===================== Matrix 1 =====================\n')
        f.write(str(matrix1))
        f.write('\n\n')

        f.write('===================== Matrix 2 =====================\n')
        f.write(str(matrix2))
        f.write('\n\n')

        f.write('===================== Disparity =====================\n')
        f.write(str(disparity))

    return matrix1, matrix2, disparity


if __name__ == '__main__':

    # General User Process:
    # 1. Define number of documents to analyze (this is the rows, N)
    # 2. Select a distributional semantic model to use on these documents, returning a document-feature matrix.
    # 3. Use modified Procrustes Analysis function on two document-feature matrices.

    # ================ SETUP ================
    # Dimensions of proper document-feature matrix is number_of_documents x number_of_topics.
    number_of_documents = 10
    number_of_topics = 15
    document_collection = select_reuters_documents(number_of_documents)

    print_settings(number_of_documents, number_of_topics, print_settings_toggle=True)
    # ================ SETUP ================

    lsi_document_feature_matrix = latent_semantic_indexing(document_collection, number_of_topics, print_vectorization_toggle=True)
    lda_document_feature_matrix = latent_dirichlet_indexing(document_collection, number_of_topics, print_vectorization_toggle=True)

    matrix1, matrix2, disparity = modified_procrustes(lsi_document_feature_matrix, lda_document_feature_matrix, number_of_documents, number_of_topics)

    print_modified_procrustes(matrix1, matrix2, disparity, print_disparity_toggle=True)

