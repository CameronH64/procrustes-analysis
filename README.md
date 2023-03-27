# Procrustes Analysis

This python module (which is simply a .py file) is meant to be imported into a working project and its functions used as needed. The code under dunder name == dunder main is not meant to be used. Rather, it is simply testing code for developing the module. Therefore, it is advised to form your own experiments, and simply use these functions to accomplish that, instead of using this code as the be-all and end-all for these function's functionality.

See, when the module is imported, the code within the module itself will run. This is not always ideal. Therefore, dunder name == dunder main removes this "auto-run" feature of sorts, while still allowing efficient development and testing. 

This python file has a variety of functions for these purposes:
- Preparation of a Reuters corpus.
- Model training on a corpus.
- Outputting a document-feature matrix from a trained model.
- Procrustes analysis between document-feature matrices.
- Saving model to file(s) for retrieval and analysis results replication.
- Both document-feature matrices and Procrustes analysis results are outputted to an organized file structure (more on this hierarchy later).
- Miscellaneous functions

# Setting Up (Adding this Code into an Already Existing Project)
Although this is common Python knowledge, I'm including this section for the sake of clarity and completeness.
After creating a project or having a project ready:
#### Import the Procrustes Analysis File:
1. Place the procrustes_analysis.py file into your project directory, alongside the .py file that is the actual one being executed.
2. In the execution .py file (or notebook, if you're using that), import procrustes_analysis.py.

        import procrustes_analysis

    Note: To write more efficient code, you can use `import procrustes_analysis as pa` so you don't have to use the full filename to use the functions.

3. You can now use any function from this module within your Python code.

#### Install Necessary Packages for this Module:
Install the necessary packages using the requirements.txt file. When you place the requirements.txt file in the project folder, PyCharm will detect it and give the option to install its packages. Wait a few minutes for all the packages to download and PyCharm to index them.

However, in the low chance that doesn't work or you accidentally dismiss the notification, you can do this manually by using Git Bash.

1. First, ensure that the requirements.txt file is in the project folder (it could be anywhere, but it's simpler to move it in this directory).
2. Then, activate your virtual environment (although you can make these manually, PyCharm can and should automatically create and configure these for you to use.)

        source venv/Scripts/activate

3. With your virtual environment activated, install the necessary packages from the requirements file.

        pip install -r requirements.txt

Again, wait a few minutes for all the packages to download and PyCharm to index them.

#### A Quick Note on the Reuters Corpus
You may need to install the Reuters corpus. You can do that using this line of code:

    nltk.download('reuters')

This corpus will be downloaded somewhere else your computer. You can remove this line of code without needing to use it again because it'll be downloaded for use from now on.

Just for reference, the Reuters corpus is downloaded to this location:

    C:\Users\Username\AppData\Roaming\nltk_data\corpora

# Using the Functions
Even though you *can* use these functions in any order after importing, these fundamental steps should be kept in mind to use them properly and effectively. For step 2, they are grouped into letters because each model works a bit differently. But the input and output is the same.

- Setup
- Training a model
- Generating a document-feature matrix from that model
- Doing Procrustes analysis between two document-feature matrices.

### 1. Setup (this is consistent for all models):
Determine the number of documents for each model to analyze (these are the rows in the document-feature matrices)

    number_of_documents = 50
    document_collection = select_reuters_documents(number_of_documents)

### 2.a. LSI/LDA
First, set a k value for the model training (number of features).

    lsi_k = 20
    print_corpus_selection_settings(number_of_documents, lsi_k)

Second, generate a generic dictionary and generic corpus for LSI/LDA (you only need to do this once).

    generic_dictionary, generic_corpus = get_latent_dictionary_and_corpus(document_collection)

Lastly, generate the document-feature matrices:

    lsi_model = train_latent_model(generic_dictionary, generic_corpus, lsi_k, model_type='lsi')
    lsi_vectorized = vectorize_latent_model(lsi_model, generic_corpus)
    lsi_document_feature_matrix = create_latent_document_feature_matrix(lsi_vectorized, number_of_documents, lsi_k)

You now have the document-feature matrix that can be used for Procrustes Analysis.

### 2.b. Doc2Vec
First, set up Doc2Vec.

    doc2vec_k = 10
    doc2vec_tagged_tokens = get_tagged_document(document_collection)
    doc2vec_model = train_doc2vec(doc2vec_tagged_tokens, vector_size=doc2vec_k, epochs=50)
    
Then, train the model.

    doc2vec_model = load_model('doc2vec', model_index=0)

Create Doc2Vec document-feature matrices.

    doc2vec_document_feature_matrix = create_doc2vec_document_feature_matrix(doc2vec_model, doc2vec_k, document_collection)

### 2.c. BERT

First, set up BERT.

    bert_k = 10
    bert_model = train_bert(document_collection, bert_k, verbose=True)
    
Then, create its document feature matrix.

    bert_document_feature_matrix = create_bert_document_feature_matrix(bert_model, document_collection)


### 3. Procrustes Analyasis: 

    matrix1, matrix2, disparity = modified_procrustes(lsi_document_feature_matrix, lsi_document_feature_matrix)

## Optional Steps:
These aren't optional in the sense that they're not important. Although each has its own preconditions, they don't need to be done in any particular order.

### Save Procrustes Analysis to File (Optional):
Precondition:
-Must have the matrix1, matrix2, and disparity values returned from the modified_procrustes() function.

    save_procrustes_analysis_to_file(matrix1, matrix2, disparity)

### Save Document-Feature Matrix to File (Optional):
Preconditions:
-Must have a document-feature matrix generated from the model.
-The model name must match exactly what is in the source code. This is to save the document_feature matrix to the correct folder.

    save_document_feature_matrix_to_file(document_feature_matrix, 'lowercase_model_name')

### Save Model to File:
Preconditions:
-Must have a trained model in order to save it to a file.
-When a model is created, this is how it can be saved to a file:

    save_model(model_object, 'lowercase_model_name')

### Load Model from File:
Preconditions:
-The model\_index parameter is the index position in the model directory, represented by an integer. It must be zero or above. If a given index value is too high or low, the upper and lower bounds will be used.
-(You must make sure there are saved models that you can load *before* you load them):

    loaded_model = load_model('lowercase_model_name', model_index=integer_value)

### Print Corpus Selection Settings:
This is just for printing to the screen some Settings.
Preconditions:
An integer variable for the number of documents.
An integer variable for the k value of the model you're printing.

    print_corpus_selection_settings(number_of_documents, model_name_k)
    

# Output Folder Hierarchy
These three folders contain all of the output from this module. The ellipses denote directories that will have more files generated when their respective functions are executed.

- document_feature_matrix_outputs
    - bert
    - doc2vec
    - gpt2
    - lda
    - lsi
        - timestamp_here.txt
        - ...
- procrustes_analysis_outputs
    - timestamp folder
        - timestamp_disparity.txt
        - timestamp_standardized_matrix_1.txt
        - timestamp_standardized_matrix_2.txt
    - ...
- saved_models
    - timestamp folder
        - timestamp.modelname
        - Whatever other files the load method needs for the specific text analysis models. The output and input for models varies, depending on 
    - ...

# Miscellaneous

- Python has a help() function that you can use. You can put in any function name from procrustes_analysis, and it'll print out the docstring. This is more useful in something like JupyterLab, but it's a feature, nonetheless.