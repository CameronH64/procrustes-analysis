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
Install the necessary packages through the requirements.txt file. When you place the requirements.txt file in the project folder, PyCharm *should* detect it and give the option to install its packages.

However, in the case that doesn't work, I'll cover the pip method (I use Git Bash, although this should work in most terminals).
1. First, ensure that the requirements.txt file is in the project folder (it could be anywhere, but it's simpler to move it in this directory).
2. Then, activate your virtual environment (although you can make these manually, PyCharm can and should automatically create and configure these for you to use.)

        source venv/Scripts/activate

3. With your virtual environment activated, install the necessary packages from the requirements file.
        
        pip install -r requirements.txt

When you run the code, the module should work normally.

#### A Quick Note on the Reuters Corpus
You may need to install the Reuters corpus. You can do that using this line of code:

        nltk.download('reuters')

You can remove this line of code without needing to use it again because it'll be downloaded for use from now on.

# Using the Functions
Even though you *can* use these functions in any order after importing, these fundamental steps should be kept in mind to use them properly and effectively. For step 2, they are grouped into letters because each model works a bit differently. But the input and output is the same.

- Setup
- Training a model
- Generating a document-feature matrix from that model
- Doing Procrustes analysis between two document-feature matrices.

### 1. Setup (this is consistent for all models):

    number_of_documents = 50
    document_collection = select_reuters_documents(number_of_documents)
    document_collection = preprocess_documents(document_collection)

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

### 4. Procrustes Analyasis: 

    matrix1, matrix2, disparity = modified_procrustes(lsi_document_feature_matrix, lsi_document_feature_matrix)

## Optional Steps:
These aren't optional in the sense that they're not important. They're just not necessarily required to do in any particular order, although they do require certain preconditions.

### Save Procrustes Analysis to File (Optional):
Precondition: Must have the values returned from the modified_procrustes() function.

    save_procrustes_analysis_to_file(matrix1, matrix2, disparity)

### Save Document-Feature Matrix to File (Optional):
Precondition: Must have a document-feature matrix generated from the model.

    save_document_feature_matrix_to_file(lsi_document_feature_matrix, 'lsi')

### Save Model to File:
Precondition: Must have a trained model in order to save it to a file.
(You must make sure there are saved models that you can load *before* you load them):
When a model is created, this is how it can be saved to a file:

    save_model(lsi_model, 'lsi')

### Load Model from File:
The model_index parameter is the position in the model directory, as an index. It must be zero or above. If below or above a valid model_index, the lower or upper bound value will be used.

    lsi_model = load_model('lsi', model_index=0)

### Print Corpus Selection Settings:
Precondition: None
This is just for printing to the screen some Settings.

    print_corpus_selection_settings(number_of_documents, doc2vec_k)


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