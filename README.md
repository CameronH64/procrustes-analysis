# Procrustes Analysis

#### Introduction
The objective of this Python module is to generate document-feature matrices from LSI, LDA, Doc2Vec, and BERT. Then, with those document-feature matrices, compare them using Procrustes analysis. The outputted value, the disparity is a measure of how similar (or not) the two are. You can specify how many documents (training or testing) to analyze, as well as a k value for any of these models. This information will be displayed in the generated names for these files.

This Python file has a variety of functions for these purposes:
- Preparation of a Reuters corpus.
- Model training on a corpus.
- Outputting a document-feature matrix from a trained model.
- Procrustes analysis between document-feature matrices.
- Saving model to file(s) for retrieval and analysis results replication.
- Both document-feature matrices and Procrustes analysis results are outputted to an organized file structure.
- Miscellaneous functions

# Setting Up (Adding this Code into an Already Existing Project)
Although this is common Python knowledge, I'm including this section for the sake of clarity and completeness.
After creating a project or having a project ready:
#### Import the Procrustes Analysis File:
1. Place the procrustes_analysis.py file into your project directory, alongside the .py file that is the actual one being executed.
2. In the execution .py file, import procrustes_analysis.py.

        import procrustes_analysis

    Note: To write more efficient code, you can use `import procrustes_analysis as pa` so you don't have to use the full filename to use the functions.

3. Be sure to use "pa" before any function that you want to use. You can now use any function from this module within your Python code.

#### Install Necessary Packages for this Module:
Install the necessary packages using the requirements.txt file. When you place the requirements.txt file in the project folder, PyCharm will detect it and give the option to install its packages. Wait a few minutes for all the packages to download and PyCharm to index them.

However, in the low chance that doesn't work or you accidentally dismiss the notification, you can do this manually by using Git Bash.

1. First, ensure that the requirements.txt file is in the project folder (it could be anywhere, but it's simpler to move it in this directory).
2. Then, activate your virtual environment (although you can make these manually, PyCharm can and should automatically create and configure one of these for you to use for your project). "source" is the command, and it takes an argument, the path to the activate file.

        source venv/Scripts/activate

3. With your virtual environment activated, install the necessary packages from the requirements file. 

        pip install -r requirements.txt

Again, wait a few minutes for all the packages to download and PyCharm to index them.

#### A Quick Note on the Reuters Corpus
If you haven't used the Reuters corpus, you'll need to install it. You can do that using this line of code:

    nltk.download('reuters')

This corpus will be downloaded somewhere else your computer. You can remove this line of code without needing to use it again because it'll be downloaded for use from now on.

Just for reference, the Reuters corpus is downloaded to this location:

    C:\Users\Username\AppData\Roaming\nltk_data\corpora

# Using the Functions
Even though you *can* use these functions in any order after importing, these fundamental steps should be kept in mind to use them properly and effectively. For step 2, they are grouped into letters because each model works a bit differently. But the input and output is the same.

- Setup
- Training a model
- Generating a document-feature matrix from that model
- Procrustes analysis between two document-feature matrices.

### 1. Setup (this is consistent for all models):
Determine the number of documents for each model to analyze (these are the rows in the document-feature matrices)

    number_of_documents = 50
    document_collection = pa.select_reuters_training_documents(number_of_documents)

### 2.a. LSI/LDA
First, set a k value for the model training (this value is the number features).

    lsi_k = 20
    pa.print_corpus_selection_settings(number_of_documents, lsi_k)

Second, generate a generic dictionary and generic corpus for LSI/LDA (you only need to do this once. Then, you can simply use these variables for LSI/LDA later on).

    generic_dictionary, generic_corpus = pa.get_latent_dictionary_and_corpus(document_collection)

Lastly, generate the document-feature matrices:

    lsi_model = pa.train_latent_model(generic_dictionary, generic_corpus, lsi_k, model_type='lsi')
    lsi_vectorized = pa.vectorize_latent_model(lsi_model, generic_corpus)
    lsi_document_feature_matrix = pa.create_latent_document_feature_matrix(lsi_vectorized, number_of_documents, lsi_k)

You now have the document-feature matrix that can be used for Procrustes Analysis.

### 2.b. Doc2Vec
First, set up Doc2Vec.

    doc2vec_k = 10
    doc2vec_tagged_tokens = pa.get_tagged_document(document_collection)
    doc2vec_model = pa.train_doc2vec(doc2vec_tagged_tokens, vector_size=doc2vec_k, epochs=50)
    
Then, train the model.

    doc2vec_model = pa.load_model('doc2vec', model_index=0)

Create Doc2Vec document-feature matrices.

    doc2vec_document_feature_matrix = pa.create_doc2vec_document_feature_matrix(doc2vec_model, doc2vec_k, document_collection)

### 2.c. BERT

First, set up BERT.

    bert_k = 10
    bert_model = pa.train_bert(document_collection, bert_k, verbose=True)
    
Then, create its document-feature matrix.

    bert_document_feature_matrix = pa.create_bert_document_feature_matrix(bert_model, document_collection)


### 3. Procrustes Analyasis: 

    matrix1, matrix2, disparity = pa.modified_procrustes(lsi_document_feature_matrix, lsi_document_feature_matrix)

## Optional Steps:
These aren't optional in the sense that they're not important. They're a very important features of the module. They just don't have to be done in any particular order. Although, they do each have their own preconditions.

### Save Procrustes Analysis to File (Optional):
Preconditions:  
-Must have the matrix1, matrix2, and disparity values returned from the modified_procrustes() function.

    pa.save_procrustes_analysis_to_file(matrix1, matrix2, disparity)

### Save Document-Feature Matrix to File (Optional):
Preconditions:  
-Must have a document-feature matrix generated from a model.  
-The model name must match exactly what is in the source code. This is to save the document_feature matrix to the correct folder.

    pa.save_document_feature_matrix_to_file(document_feature_matrix, 'lowercase_model_name')

### Save Model to File:
Preconditions:  
-Must have a trained model in order to save it to a file.  
-When a model is created, this is how it can be saved to a file:

    pa.save_model(model_object, 'lowercase_model_name')

### Load Model from File:
Preconditions:  
-The model\_index parameter is the index position in the model directory, represented by an integer. It must be zero or above. If a given index value is too high or low, the upper and lower bounds will be used.  
-(You must make sure there are saved models that you can load *before* you load them):

    loaded_model = pa.load_model('lowercase_model_name', model_index=integer_value)

### Print Corpus Selection Settings:
This is just some organizational code that will print a header containing the number of documents and model_name before document-feature matrix is printed.
Preconditions:  
-An integer variable for the number of documents.  
-An integer variable for the k value of the model you're printing.

    pa.print_corpus_selection_settings(number_of_documents, model_name_k)
    

# Output Folder Hierarchy
These three folders contain all of the output from this module. The ellipses denote directories that will have more files generated when their respective functions are executed.

- document_feature_matrix_outputs
    - bert
    - doc2vec
    - gpt2
    - lda
    - lsi
        - document_feature_matrix.txt
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
        - Whatever other files the load method needs for the specific text analysis models. The input/output of the models depends on the model being saved/loaded.
    - ...

# Miscellaneous

- Python has a help() function that you can use. You can put in any function name from procrustes_analysis, and it'll print out the docstring. This is more useful in something like JupyterLab because you can directly read the docstrings in the code, but it's a feature nonetheless.

- The code under if\_\_name\_\_ == '\_\_main\_\_' is simply my testing code for developing the module; it's not meant to be the only way to use this module. It is advised to form your own experiments, and simply use these functions to accomplish that.
See, when the module is imported, the entire code within the module itself will run. This is not always desirable. Therefore, if \_\_name\_\_ == '\_\_main\_\_' removes this "auto-run" feature of sorts, which allows for using the functions directly. That's why this code is great for testing. Although it's there, it won't run when importing the module, making the code essentially invisible. This is just one of Python's interesting features.

- LSI and LDA, Doc2Vec, and BERT have to have separate functions for their categories because they take in different arguments. That's why I couldn't make a "one size fits all" function to "train" a model. The model would've been very convoluted and confusing. However, the functions are organized by category, shown by comment headers.

- A note about the document_collection variable:

document_collection MUST be a list such that:  
-Each entry (row) in the list represents a document.  
-Each entry in each row is a string for each word.  
-Some examples on the internet show each corpus with many long strings representing a document, but the reuters corpus skips this step and separates them into words automaticaly. That's why there's a consolidate function. The format to use for input depends on the model being used.

Example:
[['BAHIA', 'COCOA', 'REVIEW', 'Showers', 'continued', ...],  
['COMPUTER', 'TERMINAL', 'SYSTEMS', '&', 'lt', ';', ...],  
['N', '.', 'Z', '.', 'TRADING', 'BANK', 'DEPOSIT', ...],  
['NATIONAL', 'AMUSEMENTS', 'AGAIN', 'UPS', 'VIACOM', ...],  

