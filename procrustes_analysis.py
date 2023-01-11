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


def print_modified_procrustes(matrix1, matrix2, disparity):
    r"""Simplified Vectorized Corpus Printing

    Parameters
    ----------
    matrix1 : numpy array
        The first document-feature matrix.
    matrix2 : numpy array
        The second document-feature matrix.
    disparity : float
        The M^2 value that denotes disparity.

    Returns
    -------
    None : N/A
    """

    print()

    print('==================== Matrix 1 ====================')
    print(matrix1)
    print()
    print('==================== Matrix 2 ====================')
    print(matrix2)
    print()
    print('==================== Disparity (Rounded) ====================')
    print(round(disparity, 2))
    print()


def print_corpus_selection_settings(number_of_documents, number_of_topics):
    r"""Print Settings of Text Analysis

    Parameters
    ----------
    number_of_documents : integer
        The number of documents analyzed.
    number_of_topics : integer
        The number of topics found.

    Returns
    -------
    None : N/A
    """

    print('================= SETTINGS =================')
    print(f'{"Number of Documents:":>24} {number_of_documents:>6}')
    print(f'{"Number of Topics:":>24} {number_of_topics:>6}')
    print('============================================')
    print()


def create_document_feature_matrix(vectorization, number_of_documents, number_of_topics):
    r"""Convert a Vectorized Corpus to a Standard-sized Document-Feature Matrix

    Parameters
    ----------
    vectorization : gensim object
        The distributional semantic model's vectorized corpus.
    number_of_documents : integer
        The number of documents that determines the rows to append and fill.
    number_of_topics : integer
        The number of documents that determines the rows to append and fill.

    Returns
    -------
    document_feature_matrix : numpy array
        The numpy array that is the document-feature array.
    """

    document_feature_matrix = np.zeros((number_of_documents, number_of_topics))

    for document in range(len(vectorization)):

        for topic_entry in vectorization[document]:      # Do something with the column value.

            topic_placement = topic_entry[0]
            document_feature_matrix[document][topic_placement] = topic_entry[1]

    # print('Document-Feature Matrix:', lsi_document_feature_matrix)

    return document_feature_matrix


def save_document_feature_matrix_to_file(document_feature_matrix, model_type):
    r"""Convert a Vectorized Corpus to a Standard-sized Document-Feature Matrix

    Parameters
    ----------
    document_feature_matrix : numpy array
        The document-feature matrix to save to a file.
    model_type : string
        An abbreviation of the model used to make the document-feature matrix.

    Returns
    -------
    None : N/A
    """

    # Generate ISO 8601 datetime for unique file names.
    date_now = datetime.today()
    current_date = date_now.strftime("%Y.%m.%d")

    time_now = datetime.now()
    current_time = time_now.strftime("%H.%M.%S")

    # Save the document-feature matrix (which is a numpy array) to a text file.
    np.savetxt(f'distributional_semantic_model_outputs/{model_type}_document_feature_matrix_{current_date}T{current_time}Z.txt', X=document_feature_matrix)

    # fmt='%.2f' can format the output per entry.
    # May add a dynamic time appending feature the name above.


def vectorize_model(model, corpus):
    r"""Vectorize a Distributional Semantic Model Using the Model and a Corpus

    Parameters
    ----------
    model : gensim model
        The gensim model to be vectorized.
    corpus : list
        List of words in corpus.

    Returns
    -------
    model[corpus] : gensim.interfaces.TransformedCorpus
    """

    return model[corpus]


def train_latent_semantic_indexing(document_collection, number_of_topics):
    r"""Modified and Condensed Latent Semantic Indexing

    Parameters
    ----------
    document_collection : 2D list
        A 2D list in which each row is a complete Reuters document, and each entry contains one word from it.
    number_of_topics : integer
        The number of topics to do LSI on.

    Returns
    -------
    lsi_document_feature_matrix : numpy array
        The "array-like" object needed for Procrustes analysis.
    """

    lsi_dictionary = corpora.Dictionary(document_collection)
    lsi_corpus = [lsi_dictionary.doc2bow(text) for text in document_collection]
    lsi_model = LsiModel(lsi_corpus, id2word=lsi_dictionary, num_topics=number_of_topics)

    return lsi_model, lsi_corpus


def train_latent_dirichlet_allocation(document_collection, number_of_topics):
    r"""Modified and Condensed Latent Dirichlet Allocation

    Parameters
    ----------
    document_collection : 2D list
        A 2D list in which each row is a complete Reuters document, and each entry contains one word from it.
    number_of_topics : integer
        The number of topics to do LDA on.

    Returns
    -------
    lda_document_feature_matrix : numpy array
        The "array-like" object needed for Procrustes analysis.
    """

    lda_dictionary = corpora.Dictionary(document_collection)
    lda_corpus = [lda_dictionary.doc2bow(text) for text in document_collection]
    lda_model = LdaModel(lda_corpus, id2word=lda_dictionary, num_topics=number_of_topics)

    return lda_model, lda_corpus


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
    number_of_topics = 20
    document_collection = select_reuters_documents(number_of_documents)

    # print_corpus_selection_settings(number_of_documents, number_of_topics)
    # ================ SETUP ================

    # Create LSI document-feature matrices.
    lsi_model, lsi_corpus = train_latent_semantic_indexing(document_collection, number_of_topics)
    lsi_vectorized = vectorize_model(lsi_model, lsi_corpus)
    lsi_document_feature_matrix = create_document_feature_matrix(lsi_vectorized, number_of_documents, number_of_topics)

    # Create LDA document-feature matrices.
    lda_model, lda_corpus = train_latent_dirichlet_allocation(document_collection, number_of_topics)
    lda_vectorized = vectorize_model(lda_model, lda_corpus)
    lda_document_feature_matrix = create_document_feature_matrix(lda_vectorized, number_of_documents, number_of_topics)

    # Print vectorized corpora.
    # print_vectorized_corpus(lsi_vectorized, 'LSI')
    # print_vectorized_corpus(lda_vectorized, 'LDA')

    # Save document-feature matrices to a file.
    save_document_feature_matrix_to_file(lsi_document_feature_matrix, 'lsi')
    save_document_feature_matrix_to_file(lda_document_feature_matrix, 'lda')

    matrix1, matrix2, disparity = modified_procrustes(lsi_document_feature_matrix, lda_document_feature_matrix, number_of_documents, number_of_topics)

    print_modified_procrustes(matrix1, matrix2, disparity,)

