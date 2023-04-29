# --------------------- DESCRIPTION ---------------------
# Author: Cameron Holbrook
# Purpose: Perform Procrustes analysis on document-feature matrices from two text analysis models.



# ----------------------- IMPORTS -----------------------

# Basic imports
import os
import nltk
import numpy as np
from pprint import pprint
from datetime import datetime

# Semantic model imports
from nltk.corpus import reuters         # For importing the Reuters dataset (nltk.download('reuters'))
from bertopic import BERTopic
from scipy.spatial import procrustes
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.corpora import Dictionary
from gensim.models import LsiModel, LdaModel
from gensim.models.doc2vec import TaggedDocument, Doc2Vec

# ----------------------- / IMPORTS -----------------------



# ------------------------ TRAINING -----------------------
def train_latent_model(dictionary, corpus, number_of_topics, latent_model_string=''):
    r"""Latent Model Training

    Description
    -----------
    Having these two models in their separate functions is silly when they're each so simple
    and take the same parameters. So, I combined them into one function. To train one model
    or the other, input the correct model_type string.

    Parameters
    ----------
    dictionary : 2D list
        A 2D list in which each row is a complete Reuters document, and each entry contains one word from it.
    corpus : list
        The number of topics to do LSI on.
    number_of_topics : integer
        Number of topics to train the LSI model on.
    latent_model_string : str
        The string for the type of model. Can be either 'lsi' or 'lda', respectively.

    Returns
    -------
    lsi_model : LsiModel
        The LSI model that will be used to create a document-feature matrix from.
    OR
    lda_model : LdaModel
        The LDA model that will be used to create a document-feature matrix from.
    """

    if latent_model_string == 'lsi':
        lsi_model = LsiModel(corpus, id2word=dictionary, num_topics=number_of_topics)
        return lsi_model

    elif latent_model_string == 'lda':
        lda_model = LdaModel(corpus, id2word=dictionary, num_topics=number_of_topics)
        return lda_model


def train_doc2vec(model_tokens, vector_size=10, alpha=0.1, epochs=100):
    r"""Doc2Vec Training

    Parameters
    ----------
    model_tokens : list
        The tokens of the document_collection.
    vector_size : int
        The size of vector to train doc2vec with. This will also be the output of each document vector.
    alpha : float
        Controls the learning rate during model training.
    epochs : int
        The number of passes to analyze the corpus.
    Returns
    -------
    doc2vec_model : doc2vec model
        The doc2vec model that will be used to create a document-feature matrix from.
    """

    doc2vec_model = Doc2Vec(vector_size=vector_size, window=2, min_count=2, epochs=epochs, alpha=alpha)
    doc2vec_model.build_vocab(model_tokens)

    # print('DOC2VEC Model Training: START')
    doc2vec_model.train(model_tokens, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)
    # print('Epochs: {} \tTraining loss: {}'.format(mdl.epochs, mdl.get_latest_training_loss()))
    # print('DOC2VEC Model Training: END')

    return doc2vec_model


def train_bert(document_collection, number_of_topics, min_topic_size=5, verbose=False):
    r"""BERT Training using BERTopic

    Parameters
    ----------
    document_collection : 2D list
        The selected Reuters documents.
    number_of_topics : int
        The number of topics to find.
    min_topic_size : int
        The minimum size of the topic. Increasing this value will lead to a lower number of clusters/topics.
    verbose : bool
        A boolean that determines if BERTopic training debugging will show.

    Returns
    -------
    bert_model : bert model
        The bert model that will be used to create a document-feature matrix from.
    """

    # Turns the 2D list of words into a 1D list of strings, constructed from all those words and separated with spaces.
    document_collection = consolidate(document_collection)

    # Build the BERT model.
    bert_model = BERTopic(nr_topics=number_of_topics, min_topic_size=min_topic_size,
                          calculate_probabilities=True, verbose=verbose)

    # Fine-tune the BERT model with the Reuters corpus.
    bert_model.fit(document_collection)

    return bert_model

# ----------------------- CREATING DOCUMENT FEATURE MATRICES -----------------------
def create_latent_document_feature_matrix(vectorization, number_of_documents, number_of_topics):
    r"""Create a Document-Feature Matrix from Latent Vectorization

    Parameters
    ----------
    vectorization : gensim object
        The latent model's vectorized corpus.
    number_of_documents : integer
        The number of documents that determines the rows to append and fill.
    number_of_topics : integer
        The number of documents that determines the rows to append and fill.

    Returns
    -------
    document_feature_matrix : numpy array
        The numpy array that is the document-feature array.

    Description
    -----------
    This function must be used for both of the "latent" models, latent semantic indexing and
    latent dirichlet allocation. This function works for both, since they are so similar.
    """

    document_feature_matrix = np.zeros((number_of_documents, number_of_topics))         # Initialize an empty document-feature matrix to be populated.

    for document in range(len(vectorization)):              # For the length of the number of documents,

        for topic_entry in vectorization[document]:         # For each topic found, place it in the document-feature matrix.

            topic_placement = topic_entry[0]
            document_feature_matrix[document][topic_placement] = topic_entry[1]

    # print('Document-Feature Matrix:')
    # print('Length:', len(document_feature_matrix))
    # pprint(document_feature_matrix)

    return document_feature_matrix


def create_doc2vec_document_feature_matrix(doc2vec_model, doc2vec_k, document_collection):
    r"""Create a Document-Feature Matrix from Doc2Vec Model

    Parameters
    ----------
    doc2vec_model : doc2vec model
        The trained Doc2Vec model that will be used to make the document-feature matrix.
    doc2vec_k : integer
        The number of features for Doc2Vec to analyze.
    document_collection : 2D list
        The list of selected Reuters documents to use.

    Returns
    -------
    document_feature_matrix : numpy array
        The numpy array that is the document-feature array.
    """

    document_feature_matrix = np.zeros((len(document_collection), doc2vec_k))

    for i, document in enumerate(document_collection):

        temp_vector = doc2vec_model.infer_vector(document)
        for j, entry in enumerate(temp_vector):      # Do something with the column value.
            document_feature_matrix[i][j] = entry

    return document_feature_matrix


def create_bert_document_feature_matrix(bert_model, document_collection):
    r"""Create a Document-Feature Matrix from BERT Model

    Parameters
    ----------
    bert_model : bert_model
        The BERT model in which the document-feature matrix will be extracted from.
    document_collection : 2D list
        The selected Reuters documents that will be trained from.

    Returns
    -------
    document_feature_matrix : numpy array
        The numpy array that is the document-feature array.
    """

    document_collection = consolidate(document_collection)

    topics, probs = bert_model.transform(document_collection)

    return np.array(probs)

# ----------------------- SAVING -----------------------
def save_document_feature_matrix_to_file(document_feature_matrix, model_type, k):
    r"""Save Document-feature Matrix to File

    Parameters
    ----------
    document_feature_matrix : numpy array
        The document-feature matrix to save to a file.
    model_type : string
        An abbreviation of the model used to name the newly generated file.
    k : integer
        The value for number of topics.

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
    current_time = time_now.strftime("%H.%M.%S.%f")

    file_name = f"{current_date}T{current_time}Z_{model_type}_rows_{len(document_feature_matrix)}_k_{k}.txt"

    # Save the document-feature matrix (which is a numpy array) to a text file.
    np.savetxt(os.path.join(path, model_type, file_name), X=document_feature_matrix)

    # fmt='%.2f' can format the output per entry.
    # May add a dynamic time appending feature the name above.


def save_procrustes_analyses_to_folder(matrix1, matrix2, disparity, k1, k2, model1_name, model2_name):
    r"""Save Procrustes Analyses Results to a Folder

    Parameters
    ----------
    matrix1 : numpy array
        The first matrix to be saved to a file.
    matrix2 : numpy array
        The second matrix to be saved to a file.
    disparity : float
        The disparity value to be saved to a file.
    k1 : integer
        The value for matrix 1's number of topics.
    k2 : integer
        The value for matrix 2's number of topics.
    model1_name : str
        The name of the first model to be used in the file name.
    model2_name : str
        The name of the second model to be used in the file name.


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
    current_time = time_now.strftime("%H.%M.%S.%f")

    # 1. Make the procrustes folder.
    # Note: k1 and k2 correspond to matrix 1 and matrix 2, respectively.
    # "rows" is the number of documents. Also, for simplicity, this is derived from the length of one matrix, not as a function argument.
    procrustes_folder = f"{current_date}T{current_time}Z_{model1_name}_{model2_name}_rows_{len(matrix1)}_k1_{k1}_k2_{k2}"
    os.mkdir(os.path.join(path, procrustes_folder))


    # 2. Write standardized_matrix_1 to file.
    procrustes_matrix1_file = f"{current_date}T{current_time}Z_standardized_matrix_1_k_{k1}.txt"

    with open(os.path.join(path, procrustes_folder, procrustes_matrix1_file), 'w') as matrix1_standardization:
        for document in matrix1:
            for feature in document:
                matrix1_standardization.write(str(feature) + " ")
            matrix1_standardization.write("\n")


    # 3. Write standardized_matrix_2 to file.
    procrustes_matrix2_file = f"{current_date}T{current_time}Z_standardized_matrix_2_k_{k2}.txt"

    with open(os.path.join(path, procrustes_folder, procrustes_matrix2_file), 'w') as matrix2_standardization:
        for document in matrix2:
            for feature in document:
                matrix2_standardization.write(str(feature) + " ")
            matrix2_standardization.write("\n")


    # 4. Write disparity to file.
    disparity_filename = f"{current_date}T{current_time}Z_disparity.txt"

    with open(os.path.join(path, procrustes_folder, disparity_filename), 'w') as disparity_value:
        disparity_value.write(str(disparity))


def save_model(model, model_name, k, rows):
    r"""Save a Model to Files

    Parameters
    ----------
    model : gensim.model
        The string that determines what model is being saved.
    model_name : string
        The string that determines both where the model is saved and how the files are named.
    k : integer
        The value for the model's number of topics.
    rows : integer
        The value for how many documents the model was trained on.

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
    current_time = datetime.now().strftime("%H.%M.%S.%f")

    # Generate the name of the model being saved.
    model_folder = f"{current_date}T{current_time}Z_rows_{rows}_k_{k}"
    os.mkdir(os.path.join(path, model_name, model_folder))

    model.save(os.path.join(path, model_name, model_folder, model_folder+"."+model_name))

# ----------------------- PRINTING -----------------------
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

    print(f'------------------- {model_name} Vectorized Corpus -------------------')
    for count, value in enumerate(vectorized_corpus):
        print(f'Document {count + 1}: ', end='')
        print(value)
    print('-----------------------------------------------------------')


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

    print('------------------- Matrix 1 -------------------')

    for count, document in enumerate(matrix1):
        print(f'Document: {count + 1}', document)
        print()

    print()

    print('------------------- Matrix 2 -------------------')
    for count, document in enumerate(matrix2):
        print(f'Document {count + 1}:', document)
        print()

    print()

    print('--------------------- Disparity ---------------------')
    print(str(disparity))


def print_corpus_selection_settings(model_string, number_of_documents, number_of_topics):
    r"""Text Analysis Settings Print

    Parameters
    ----------
    model_string : str
        The string for the name of the model settings being printed.
    number_of_documents : integer
        The number of documents analyzed to be printed.
    number_of_topics : integer
        The number of topics analyzed to be printed.

    Returns
    -------
    None : N/A
    """

    print(f'--------------- {model_string.upper()} SETTINGS ---------------')
    print(f'{"Number of Documents:":>24} {number_of_documents:>6}')
    print(f'{"Number of Topics:":>24} {number_of_topics:>6}')
    for value in range(len(f'--------------- {model_string.upper()} SETTINGS ---------------')):
        print('-', end='')
    print()

# ----------------------- MISCELLANEOUS -----------------------
def select_reuters_training_documents(number_of_documents):
    r"""Select Reuters Training Documents to Analyze

    Parameters
    ----------
    number_of_documents : integer
        Defines the number of Reuters training documents to return.

    Returns
    -------
    reuters_documents : 2D list
        Contains all the selected documents, where each row is a document, and each entry in each row is a word.
    """

    # There are 3,019 test documents, and 7,769 training documents.

    if number_of_documents > 7769: number_of_documents = 7769       # If the user enters a number out of range, the
    elif number_of_documents < 0: number_of_documents = 0

    training_documents = reuters.fileids()[3019:]       # Retrieve only Reuters training documents. (0 to 3018 equals 3019 total) So, retrieve everything after that.
    selected_reuters_documents = []

    # Cycle through the training documents, and append as many as specified.
    for file_id in range(0, number_of_documents):     # +1 to include the last document.
        selected_reuters_documents.append(reuters.words(training_documents[file_id]))

    return selected_reuters_documents


def select_reuters_testing_documents(number_of_documents):
    r"""Select Reuters Testing Documents to Analyze

    Parameters
    ----------
    number_of_documents : integer
        Defines the number of Reuters testing documents to return.

    Returns
    -------
    reuters_documents : 2D list
        Contains all the selected documents, where each row is a document, and each entry in each row is a word.
    """

    # There are 3,019 test documents, and 7,769 training documents.

    if number_of_documents > 3019: number_of_documents = 3019
    elif number_of_documents < 0: number_of_documents = 0

    testing_documents = reuters.fileids()[:3019]       # Retrieve only Reuters training documents (0 to 3019, non-inclusive equals 3018 total documents).
    selected_reuters_documents = []

    # Cycle through the training documents, and append as many as specified.
    for file_id in range(0, number_of_documents):
        selected_reuters_documents.append(reuters.words(testing_documents[file_id]))

    return selected_reuters_documents


def preprocess_documents(document_collection):
    r"""Preprocess the Document Collection

    Parameters
    ----------
    document_collection : 2D list
        A 2D list where each row is a document, and each element in each row is a word in said document.

    Returns
    -------
    document_collection : 2D list
        The same document_collection, but cleaned for the latent semantic models.
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

    # Remove words that are only three characters.
    document_collection = [[token for token in document if len(token) > 3] for document in document_collection]

    # Lemmatize the documents (turn words like "banks" to "bank")
    lemmatizer = WordNetLemmatizer()
    document_collection = [[lemmatizer.lemmatize(token) for token in document] for document in document_collection]

    ###############################################################################
    # We remove rare words and common words based on their *document frequency*.
    # Below we remove words that appear in less than 20 documents or in more than
    # 50% of the documents. Consider trying to remove words only based on their
    # frequency, or maybe combining that with this approach.
    #

    # Debugging
    # for value in document_collection:
    #     print(value)

    return document_collection


def get_latent_dictionary_and_corpus(document_collection):
    r"""Get the latent dictionary and corpus for LSI and LDA

    Parameters
    ----------
    document_collection : 2D list
        A 2D list where each row is a document, and each element in each row is a word in said document.

    Returns
    -------
    dictionary, corpus : tuple
        The generated dictionary and corpus to be trained with.
    """

    # Create a dictionary representation of the documents.
    dictionary = Dictionary(document_collection)

    # Filter out words that occur less than 20 documents, or more than 50% of the documents.
    # This line of code serves the purpose of removing unnecessary words, and does this in two ways:
    # 1. Removing very common words, such as "the", "and", etc. that are very common.
    # 2. Removing very rare words, such as rare acronyms or specific names that don't carry much meaning.
    document_number = len(document_collection)
    percentage = .20
    dictionary.filter_extremes(no_above=0.50)

    # Practically speaking, having this line of code results in a lower (likely more accurate) disparity value.
    # From the remaining words, LSI will choose the "semantically significant words."

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


def vectorize_latent_model(model, corpus):
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


def get_tagged_document(document_collection):
    r"""Get the Tagged Documents for Doc2Vec

    Parameters
    ----------
    document_collection : 2D list
        A 2D list of the selected Reuters documents.

    Returns
    -------
    tagged_data : list
        A list of tagged data, the input for Doc2Vec.
    """

    corpus_file_ids = reuters.fileids()
    tagged_data = [TaggedDocument(d, [corpus_file_ids[i]]) for i, d in enumerate(document_collection)]

    return tagged_data


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
        The distributional semantic model to be loaded.
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
    elif model_type == 'doc2vec':
        return Doc2Vec.load(model_to_load)
    elif model_type == 'bert':
        return BERTopic.load(model_to_load)


def consolidate(document_collection):
    r"""Consolidate the 2D list of words into a 1D list of strings, each string being a document.

    Parameters
    ----------
    document_collection : 2D list
        The selected Reuters documents to be consolidated.

    Returns
    -------
    document_collection : 1D list
    """

    return [' '.join(sublist) for sublist in document_collection]


if __name__ == '__main__':

    # ---------------- TRAINING SETUP FOR ALL MODELS ----------------

    number_of_training_documents = 200
    document_collection = select_reuters_training_documents(number_of_training_documents)

    # ---------------- / TRAINING SETUP FOR ALL MODELS ----------------



    # ---------------- LATENT MODEL SETUPS ----------------

    # generic_dictionary:   A dictionary with identifier numbers and words to match.
    # generic_corpus:       The corpus represented with a list of tuples, x being a word identifier and y being the count.
    document_collection = preprocess_documents(document_collection)
    generic_dictionary, generic_corpus = get_latent_dictionary_and_corpus(document_collection)

    # ---------------- / LATENT MODEL SETUPS ----------------



    # ---------------- LSI ----------------

    # Setup for LSI
    lsi_k = 20
    print_corpus_selection_settings('lsi', number_of_training_documents, lsi_k)

    # Generate an LSI model. (either load existing model or train one from data)
    # lsi_model = load_model('lsi', model_index=0)
    lsi_model = train_latent_model(generic_dictionary, generic_corpus, lsi_k, latent_model_string='lsi')

    # Optional: Save the LSI model.
    save_model(lsi_model, 'lsi', lsi_k, number_of_training_documents)

    # Create the document-feature matrix.
    lsi_vectorized = vectorize_latent_model(lsi_model, generic_corpus)
    lsi_document_feature_matrix = create_latent_document_feature_matrix(lsi_vectorized, number_of_training_documents, lsi_k)

    # ---------------- / LSI ----------------



    # ---------------- LDA ----------------

    # Setup for LDA
    lda_k = 20
    print_corpus_selection_settings('lda', number_of_training_documents, lda_k)

    # Generate an LDA model. (either load existing model or train one from data)
    # lda_model = load_model('lda', model_index=0)
    lda_model = train_latent_model(generic_dictionary, generic_corpus, lda_k, latent_model_string='lda')

    # Optional: Save the LDA model.
    save_model(lda_model, 'lda', lda_k, number_of_training_documents)

    # Create the document-feature matrix.
    lda_vectorized = vectorize_latent_model(lda_model, generic_corpus)
    lda_document_feature_matrix = create_latent_document_feature_matrix(lda_vectorized, number_of_training_documents, lda_k)

    # ---------------- / LDA ----------------



    # ---------------- DOC2VEC ----------------

    # Setup for Doc2Vec
    doc2vec_k = 20
    print_corpus_selection_settings('doc2vec', number_of_training_documents, doc2vec_k)

    # Generate a Doc2Vec model. (either load existing model or train one from data)
    # doc2vec_model = load_model('doc2vec', model_index=0)
    doc2vec_tagged_tokens = get_tagged_document(document_collection)
    doc2vec_model = train_doc2vec(doc2vec_tagged_tokens, vector_size=doc2vec_k, epochs=50)

    # Optional: Save the Doc2Vec model
    save_model(doc2vec_model, 'doc2vec', doc2vec_k, number_of_training_documents)

    # Create Doc2Vec document-feature matrix.
    doc2vec_document_feature_matrix = create_doc2vec_document_feature_matrix(doc2vec_model, doc2vec_k, document_collection)

    # ---------------- / DOC2VEC ----------------



    # ---------------- BERT ----------------

    # Setup for BERT
    bert_k = 20
    print_corpus_selection_settings('bert', number_of_training_documents, bert_k)

    # Generate an LSI model. (either load existing model or train one from data)
    # bert_model = load_model('bert', model_index=0)
    bert_model = train_bert(document_collection, bert_k, verbose=True)

    # Optional: Save the BERT model.
    save_model(bert_model, 'bert', bert_k, number_of_training_documents)

    # Create BERT document-feature matrices.
    bert_document_feature_matrix = create_bert_document_feature_matrix(bert_model, document_collection)

    # ---------------- / BERT ----------------



    # ---------------- SAVE DOCUMENT-FEATURE MATRICES TO FILE ----------------

    save_document_feature_matrix_to_file(lsi_document_feature_matrix, 'lsi', lsi_k)
    save_document_feature_matrix_to_file(lda_document_feature_matrix, 'lda', lda_k)
    save_document_feature_matrix_to_file(doc2vec_document_feature_matrix, 'doc2vec', doc2vec_k)
    save_document_feature_matrix_to_file(bert_document_feature_matrix, 'bert', bert_k)

    # ---------------- / SAVE DOCUMENT-FEATURE MATRICES TO FILES ----------------



    # ------------------- PROCRUSTES ANALYSIS -------------------

    # Takes in two document-feature matrices. Will output standardized matrices and disparity value.
    matrix1, matrix2, disparity = modified_procrustes(lsi_document_feature_matrix, lda_document_feature_matrix)
    save_procrustes_analyses_to_folder(matrix1, matrix2, disparity, lsi_k, lda_k, 'lsi', 'lda')

    # Print analysis results to screen.
    print_modified_procrustes(matrix1, matrix2, disparity)

    # ------------------- / PROCRUSTES ANALYSIS -------------------
