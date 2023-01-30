# Author: Cameron Holbrook

# =============== DESCRIPTION ===============
# Purpose: Perform Procrustes analysis on document-feature matrices from two text analysis models.
# General User Process:
# 1. Define number of documents to analyze (this is the rows, N)
# 2. Do text analysis on those documents, and get its respective matrix.
# 3. Do procrustes analysis.
# ===========================================

# =============== EXAMPLE (USING LSI) ===============
#     # Setup for LSI
#     number_of_topics = 50
#     print_corpus_selection_settings(number_of_documents, number_of_topics)
#
#     # Create LSI document-feature matrices.
#     lsi_model = train_latent_semantic_indexing(generic_dictionary, generic_corpus, number_of_topics)
#     lsi_vectorized = vectorize_model(lsi_model, generic_corpus)
#     lsi_document_feature_matrix = create_document_feature_matrix(lsi_vectorized, number_of_documents, number_of_topics)
#
#     # Save document-feature matrices to a file.
#     save_document_feature_matrix_to_file(lsi_document_feature_matrix, 'lsi')
#
#     matrix1, matrix2, disparity = modified_procrustes(lsi_document_feature_matrix, lsi_document_feature_matrix)
#     save_procrustes_analysis_to_file(matrix1, matrix2, disparity)
#
#     modified_procrustes takes two document-feature matrices.
# ===================================================


# ========================= IMPORTS =========================
import nltk
from pprint import pprint               # For neater printing of information
from gensim.corpora import Dictionary
from nltk.corpus import reuters, stopwords         # Import the reuters dataset (from the download function), also stopwords.
from scipy.spatial import procrustes
from gensim.models import LsiModel, LdaModel
from gensim import corpora
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Phrases
import numpy as np
from datetime import datetime
import os
# ========================= / IMPORTS =========================

def print_vectorized_corpus(vectorized_corpus, model_name):
    r"""Simplified Vectorized Corpus Printing

    Parameters
    ----------
    vectorized_corpus : Gensim object
        The Gensim object in which each row represents a document, and each
        column represents a latent feature and the document's rating.
    model_name : string
        Shows which model's vectorized corpus is being used.

    Returns
    -------
    None : N/A

    Notes
    -----
    This is only a debugging-type function for printing to the screen.
    """

    print(f'================== {model_name} Vectorized Corpus ==================')
    for count, value in enumerate(vectorized_corpus):
        print(f'Document {count + 1}: ', end='')
        print(value)
    print('===========================================================')


def print_modified_procrustes(matrix1, matrix2, disparity):
    r"""Simplified Vectorized Corpus Printing

    Parameters
    ----------
    matrix1 : numpy array
        The first document-feature matrix to be printed.
    matrix2 : numpy array
        The second document-feature matrix to be printed.
    disparity : float
        The M^2 value that denotes disparity to be printed.

    Returns
    -------
    None : N/A
    """

    print()

    print('===================== Matrix 1 =====================')

    for count, document in enumerate(matrix1):
        print(f'Document: {count + 1}', document)
        print()

    print()

    print('===================== Matrix 2 =====================')
    for count, document in enumerate(matrix2):
        print(f'Document {count + 1}:', document)
        print()

    print()

    print('===================== Disparity =====================')
    print(str(disparity))


def print_corpus_selection_settings(number_of_documents, number_of_topics):
    r"""Print Settings of Text Analysis

    Parameters
    ----------
    number_of_documents : integer
        The number of documents analyzed to be printed.
    number_of_topics : integer
        The number of topics analyzed to be printed.

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

    path = './document_feature_matrix_outputs'
    directory_exists = os.path.exists(path)
    if not directory_exists:
        os.mkdir(path)

    # Generate ISO 8601 datetime for unique file names.
    date_now = datetime.today()
    current_date = date_now.strftime("%Y.%m.%d")

    time_now = datetime.now()
    current_time = time_now.strftime("%H.%M.%S")

    # Save the document-feature matrix (which is a numpy array) to a text file.
    np.savetxt(f'document_feature_matrix_outputs/{current_date}T{current_time}Z_{model_type}.txt', X=document_feature_matrix)

    # fmt='%.2f' can format the output per entry.
    # May add a dynamic time appending feature the name above.


def save_procrustes_analysis_to_file(matrix1, matrix2, disparity):
    r"""Convert a Vectorized Corpus to a Standard-sized Document-Feature Matrix

    Parameters
    ----------
    matrix1 : numpy array
        The first matrix to be saved to a file.
    matrix2 : numpy array
        The second matrix to be saved to a file.
    disparity : float
        The disparity value to be saved to a file.

    Returns
    -------
    None : N/A
    """

    path = './procrustes_analysis_outputs'
    directory_exists = os.path.exists(path)
    if not directory_exists:
        os.mkdir(path)

    date_now = datetime.today()
    current_date = date_now.strftime("%Y.%m.%d")

    time_now = datetime.now()
    current_time = time_now.strftime("%H.%M.%S")

    with open(f'procrustes_analysis_outputs/{current_date}T{current_time}Z.txt', 'w') as procrustes_file:
        procrustes_file.write('===================== Matrix 1 =====================\n')

        for count, document in enumerate(matrix1):
            procrustes_file.write(f'Document {count+1}: {str(document)}')
            procrustes_file.write('\n\n')

        procrustes_file.write('\n\n')

        procrustes_file.write('===================== Matrix 2 =====================\n')
        for document in matrix2:
            procrustes_file.write(f'Document {count+1}: {str(document)}')
            procrustes_file.write('\n\n')

        procrustes_file.write('\n\n')

        procrustes_file.write('===================== Disparity =====================\n')
        procrustes_file.write(str(disparity))


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


def train_latent_semantic_indexing(dictionary, corpus, number_of_topics):
    r"""Modified and Condensed Latent Semantic Indexing

    Parameters
    ----------
    dictionary : 2D list
        A 2D list in which each row is a complete Reuters document, and each entry contains one word from it.
    corpus : integer
        The number of topics to do LSI on.
    number_of_topics : integer
        Number of topics to train the LSI model on.

    Returns
    -------
    lsi_document_feature_matrix : numpy array
        The "array-like" object needed for Procrustes analysis.
    """

    # lsi_dictionary = corpora.Dictionary(document_collection)
    # lsi_corpus = [lsi_dictionary.doc2bow(text) for text in document_collection]
    lsi_model = LsiModel(corpus, id2word=dictionary, num_topics=number_of_topics)

    return lsi_model


def train_latent_dirichlet_allocation(dictionary, corpus, number_of_topics):
    r"""Modified and Condensed Latent Dirichlet Allocation

    Parameters
    ----------
    dictionary : 2D list
        A 2D list in which each row is a complete Reuters document, and each entry contains one word from it.
    corpus : 2D list
        List of (x,y) points that denote the count of each word.
    number_of_topics : integer
        The number of topics to do LDA on.

    Returns
    -------
    lda_document_feature_matrix : numpy array
        The "array-like" object needed for Procrustes analysis.
    """

    # lda_dictionary = corpora.Dictionary(document_collection)
    # lda_corpus = [lda_dictionary.doc2bow(text) for text in document_collection]
    lda_model = LdaModel(corpus, id2word=dictionary, num_topics=number_of_topics)

    return lda_model


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

    # There are 3,020 test documents, and 7,768 training documents.

    training_documents = reuters.fileids()[3019:]       # Retrieve only Reuters training documents.
    selected_reuters_documents = []

    for file_id in range(0, number_of_documents):
        selected_reuters_documents.append(reuters.words(training_documents[file_id]))

    return selected_reuters_documents


def preprocess_documents(document_collection):
    r"""Preprocess Document Collection

    Parameters
    ----------
    document_collection : 2D list
        A 2D list where each row is a document, and each element in each row is a word in said document.

    Returns
    -------
    dictionary, corpus : tuple
        The generated dictionary and corpus to be trained with.
    """

    # This function does these preprocessing steps (directly from Gensim LDA Model documentation):

    # Converts all words in all documents to lowercase.
    # Remove numbers, but not words that contain numbers.
    # Remove words that are only one character.
    # Lemmatize the documents.
    # Add bigrams and trigrams to docs.
    # Remove rare words and common words based on their *document frequency*.

    # https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html

    # Debugging
    # print('Before preprocessing:')
    # for value in document_collection[0]:
    #     print(value, end='')
    #     print(' ', end='')
    # print()

    # Convert all documents to lowercase.
    document_collection = [[x.lower() for x in sublist] for sublist in document_collection]

    # Remove numbers, but not words that contain numbers.
    document_collection = [[token for token in document if not token.isnumeric()] for document in document_collection]

    # Remove words that are only one character.
    document_collection = [[token for token in document if len(token) > 1] for document in document_collection]

    # Lemmatize the documents.
    lemmatizer = WordNetLemmatizer()
    document_collection = [[lemmatizer.lemmatize(token) for token in document] for document in document_collection]

    # Add bigrams and trigrams to docs (only ones that appear 20 times or more).
    bigram = Phrases(document_collection, min_count=20)

    for idx in range(len(document_collection)):
        for token in bigram[document_collection[idx]]:
            if '_' in token:

                # Token is a bigram, add to document.
                document_collection[idx].append(token)
                # print(token)

    ###############################################################################
    # We remove rare words and common words based on their *document frequency*.
    # Below we remove words that appear in less than 20 documents or in more than
    # 50% of the documents. Consider trying to remove words only based on their
    # frequency, or maybe combining that with this approach.
    #

    # Create a dictionary representation of the documents.
    dictionary = Dictionary(document_collection)

    # Filter out words that occur less than 20 documents, or more than 50% of the documents.
    # This line of code serves the purpose of removing unnecessary words, and does this in two ways:
    # 1. Removing very common words, such as "the", "and", etc. that are very common.
    # 2. Removing very rare words, such as rare acronyms or specific names that don't carry much meaning.
    document_number = len(document_collection)
    percentage = .20
    dictionary.filter_extremes(no_below=document_number * percentage, no_above=0.50)

    # Practically speaking, having this line of code results in a lower (likely more accurate) disparity value.

    ###############################################################################
    # Finally, we transform the documents to a vectorized form. We simply compute
    # the frequency of each word, including the bigrams.
    #

    # Bag-of-words representation of the documents.
    corpus = [dictionary.doc2bow(document) for document in document_collection]

    # Debugging
    # print('After preprocessing:')
    # for value in document_collection[0]:
    #     print(value, end='')
    #     print(' ', end='')
    # print()

    # print(corpus)

    return dictionary, corpus


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
        M^2 value that denotes disparity between input matrices.
    """

    # A note from the documentation: the disparity value does not depend on
    # order of input matrices, but the output matrices do. Only the first output matrix
    # is guaranteed to be scaled such that tr(AAT) = 1. (Trace of matrix A times A transposed equals 1).

    matrix1_topic = document_feature_matrix_1.shape[1]
    matrix2_topic = document_feature_matrix_2.shape[1]

    if matrix1_topic > matrix2_topic:
        document_feature_matrix_2 = np.pad(document_feature_matrix_2, [(0, 0), (0, matrix1_topic-matrix2_topic)], mode='constant')
    else:
        document_feature_matrix_1 = np.pad(document_feature_matrix_1, [(0, 0), (0, matrix2_topic-matrix1_topic)], mode='constant')

    matrix1, matrix2, disparity = procrustes(document_feature_matrix_1, document_feature_matrix_2)

    return matrix1, matrix2, disparity


if __name__ == '__main__':

    # General User Process:
    # 1. Define number of documents to analyze (this is the rows, N), and number of topics.
    # 2. Preprocess documents; clean the text data. This returns a dictionary and a corpus.
    # 3. Train a semantic model with this dictionary and corpus. This returns said semantic model.
    # 4. Get a vectorization using this model and corpus. Returns a vectorized corpus.
    # 5. Create a document feature matrix from the vectorized corpus and number of documents and topics. Returns a document feature matrix.
    # 6. Use modified Procrustes Analysis function on two document-feature matrices made with steps 1-5.

    # ================ SETUP ================
    # Dimensions of proper document-feature matrix is number_of_documents x number_of_topics.
    number_of_documents = 51
    document_collection = select_reuters_documents(number_of_documents)

    # generic_dictionary: a dictionary with identifier numbers and words to match.
    # generic_corpus: the corpus represented with a list of tuples, x being a word identifier and y being the count.
    generic_dictionary, generic_corpus = preprocess_documents(document_collection)

    # ================ SETUP ================

    # Setup for LSI
    number_of_topics = 20
    print_corpus_selection_settings(number_of_documents, number_of_topics)

    # Create LSI document-feature matrices.
    lsi_model = train_latent_semantic_indexing(generic_dictionary, generic_corpus, number_of_topics)
    lsi_vectorized = vectorize_model(lsi_model, generic_corpus)
    lsi_document_feature_matrix = create_document_feature_matrix(lsi_vectorized, number_of_documents, number_of_topics)

    # Setup for LDA
    number_of_topics = 20
    print_corpus_selection_settings(number_of_documents, number_of_topics)

    # Create LDA document-feature matrices.
    lda_model = train_latent_dirichlet_allocation(generic_dictionary, generic_corpus, number_of_topics)
    lda_vectorized = vectorize_model(lda_model, generic_corpus)
    lda_document_feature_matrix = create_document_feature_matrix(lda_vectorized, number_of_documents, number_of_topics)

    # Print vectorized corpora.
    # print_vectorized_corpus(lsi_vectorized, 'LSI')
    # print_vectorized_corpus(lda_vectorized, 'LDA')

    # Save document-feature matrices to a file.
    save_document_feature_matrix_to_file(lsi_document_feature_matrix, 'lsi')
    save_document_feature_matrix_to_file(lda_document_feature_matrix, 'lda')

    matrix1, matrix2, disparity = modified_procrustes(lsi_document_feature_matrix, lda_document_feature_matrix)
    save_procrustes_analysis_to_file(matrix1, matrix2, disparity)

    print_modified_procrustes(matrix1, matrix2, disparity)

