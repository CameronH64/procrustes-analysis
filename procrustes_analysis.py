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
from gensim.corpora import Dictionary
from nltk.corpus import reuters, stopwords         # Import the reuters dataset (from the download function), also stopwords.
from scipy.spatial import procrustes
from gensim.models import LsiModel, LdaModel
from gensim import corpora
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Phrases
from gensim.test.utils import get_tmpfile
import numpy as np
from datetime import datetime
import os
# ========================= / IMPORTS =========================

def print_vectorized_corpus(vectorized_corpus, model_name):
    r"""Simplified Vectorized Corpus Print

    Parameters
    ----------
    vectorized_corpus : gensim object
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
    r"""Simplified Vectorized Corpus Print

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
    r"""Text Analysis Settings Print

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


def save_document_feature_matrix_to_file(document_feature_matrix, model_type, k):
    r"""Save Document-feature Matrix to File

    Parameters
    ----------
    document_feature_matrix : numpy array
        The document-feature matrix to save to a file.
    model_type : string
        An abbreviation of the model used to name the newly generated file.

    Returns
    -------
    None : N/A
    """

    # Ensure that this output folder exists so that the individual models will be created as needed.
    path = f'./document_feature_matrix_outputs'
    if not os.path.exists(path):
        os.mkdir(path)

    model_folders = ['lsi', 'lda', 'doc2vec', 'bert', 'gpt2']

    # Ensure that all the necessary folders for each model are created.
    # To add more folders for each model, simply add more entries to the models list.
    for entry in model_folders:
        path = os.path.join(path, entry)
        if not os.path.exists(path):
            os.mkdir(path)
        path = f'./document_feature_matrix_outputs'


    # Generate ISO 8601 datetime for unique file names.
    date_now = datetime.today()
    current_date = date_now.strftime("%Y.%m.%d")

    time_now = datetime.now()
    current_time = time_now.strftime("%H.%M.%S")

    file_name = f"{current_date}T{current_time}Z_{model_type}_k{k}.txt"

    # Save the document-feature matrix (which is a numpy array) to a text file.
    np.savetxt(os.path.join(path, model_type, file_name), X=document_feature_matrix)

    # fmt='%.2f' can format the output per entry.
    # May add a dynamic time appending feature the name above.


def save_procrustes_analyses_to_folder(matrix1, matrix2, disparity):
    r"""Save Procrustes Analyses Results to a Folder

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

    # 1. Make the procrustes folder.
    procrustes_folder = f"{current_date}T{current_time}Z"
    os.mkdir(os.path.join(path, procrustes_folder))


    # 2. Write standardized_matrix_1 to file.
    procrustes_matrix1_file = f"{current_date}T{current_time}Z_standardized_matrix_1.txt"

    with open(os.path.join(path, procrustes_folder, procrustes_matrix1_file), 'w') as matrix1_standardization:
        for document in matrix1:
            for feature in document:
                matrix1_standardization.write(str(feature) + " ")
            matrix1_standardization.write("\n")


    # 3. Write standardized_matrix_2 to file.
    procrustes_matrix2_file = f"{current_date}T{current_time}Z_standardized_matrix_2.txt"

    with open(os.path.join(path, procrustes_folder, procrustes_matrix2_file), 'w') as matrix2_standardization:
        for document in matrix2:
            for feature in document:
                matrix2_standardization.write(str(feature) + " ")
            matrix2_standardization.write("\n")


    # 4. Write disparity to file.
    disparity_filename = f"{current_date}T{current_time}Z_disparity.txt"

    with open(os.path.join(path, procrustes_folder, disparity_filename), 'w') as disparity_value:
        disparity_value.write(str(disparity))


def save_model(model, model_name):
    r"""Save a Model to Files

    Parameters
    ----------
    model : gensim.models.lsimodel.LsiModel
        The string that determines what model is being saved.
    model_name : string
        The string that determines both where the model is saved and how the files are named.

    Returns
    -------
    None : N/A
    """

    # Ensure that this output folder exists so that the individual models will be created as needed.
    path = f'./saved_models'
    if not os.path.exists(path):
        os.mkdir(path)

    model_folders = ['lsi', 'lda', 'doc2vec', 'bert', 'gpt2']

    # Ensure that all the necessary folders for each model are created.
    # To add more folders for each model, simply add more entries to the models list.
    for entry in model_folders:
        path = os.path.join(path, entry)
        if not os.path.exists(path):
            os.mkdir(path)
        path = f'./saved_models'

    # Generate information needed for the folder that holds the saved model.
    current_date = datetime.today().strftime("%Y.%m.%d")
    current_time = datetime.now().strftime("%H.%M.%S")

    # Generate the name of the model being saved.
    model_folder = f"{current_date}T{current_time}Z"
    os.mkdir(os.path.join(path, model_name, model_folder))

    model.save(os.path.join(path, model_name, model_folder, model_folder+"."+model_name))


def load_model(model_type, model_index=0):
    r"""Load a Model from Files

    Parameters
    ----------
    model_type : string
        The string that sets the model path and type of model to be loaded.
    model_index : integer
        The integer that specifies which model to load, after the type is selected.

    Returns
    -------
    model : gensim model
    """

    # Set the folder path for the models to be loaded.
    model_path = fr'.\saved_models\{model_type}'

    # Generate a list of models to be loaded. Ensure only folders are added to said list.
    model_folders = [f for f in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, f))]

    if model_index < 0:
        print("Negative model_index is not valid. model_index is reset to 0.")
        model_index = 0
    elif model_index > len(model_folders):
        print("model_index value exceeds the possible models to load. model_index is reset to a value to load the most recent model.")
        model_index = len(model_folders) - 1

    model_timestamp = model_folders[model_index]
    model_to_load = os.path.join(model_path, model_timestamp, model_timestamp+'.'+model_type)

    if model_type == 'lsi':
        return LsiModel.load(model_to_load)
    elif model_type == 'lda':
        return LdaModel.load(model_to_load)


def vectorize_model(model, corpus):
    r"""Vectorize a Model Using the Model and a Corpus

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

    lda_model = LdaModel(corpus, id2word=dictionary, num_topics=number_of_topics)

    return lda_model


def select_reuters_documents(number_of_documents):
    r"""Select Reuters Documents to Analyze

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
    r"""Preprocess the Document Collection

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
    r"""Modified Procrustes Analysis

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
    number_of_documents = 50
    document_collection = select_reuters_documents(number_of_documents)

    # generic_dictionary: a dictionary with identifier numbers and words to match.
    # generic_corpus: the corpus represented with a list of tuples, x being a word identifier and y being the count.
    generic_dictionary, generic_corpus = preprocess_documents(document_collection)

    # ================ SETUP ================

    # ================ LSI ==================

    # Setup for LSI
    lsi_k = 20
    print_corpus_selection_settings(number_of_documents, lsi_k)

    # Create LSI document-feature matrices.
    lsi_model = train_latent_semantic_indexing(generic_dictionary, generic_corpus, lsi_k)

    save_model(lsi_model, "lsi")
    # lsi_model = load_model('lsi', model_index=0)
    lsi_vectorized = vectorize_model(lsi_model, generic_corpus)
    lsi_document_feature_matrix = create_document_feature_matrix(lsi_vectorized, number_of_documents, lsi_k)

    # ================ / LSI ==================



    # ================ LDA ==================

    # Setup for LDA
    lda_k = 10
    print_corpus_selection_settings(number_of_documents, lda_k)

    # Create LDA document-feature matrices.
    lda_model = train_latent_dirichlet_allocation(generic_dictionary, generic_corpus, lda_k)

    save_model(lda_model, "lda")
    # lda_model = load_model('lda', model_index=0)
    lda_vectorized = vectorize_model(lda_model, generic_corpus)
    lda_document_feature_matrix = create_document_feature_matrix(lda_vectorized, number_of_documents, lda_k)

    # ================ / LDA ==================



    # Print vectorized corpora.
    # print_vectorized_corpus(lsi_vectorized, 'LSI')
    # print_vectorized_corpus(lda_vectorized, 'LDA')

    # Save document-feature matrices to a file.
    save_document_feature_matrix_to_file(lsi_document_feature_matrix, 'lsi', lsi_k)
    save_document_feature_matrix_to_file(lda_document_feature_matrix, 'lda', lda_k)

    matrix1, matrix2, disparity = modified_procrustes(lsi_document_feature_matrix, lda_document_feature_matrix)
    save_procrustes_analyses_to_folder(matrix1, matrix2, disparity)

    # print_modified_procrustes(matrix1, matrix2, disparity)

