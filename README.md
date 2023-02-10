# Procrustes Analysis
This python file has a variety of functions for these purposes:
- Preparation of a Reuters corpus.
- Model training on a corpus.
- Outputting a document-feature matrix from a trained model.
- Procrustes analysis between document-feature matrices.
- Saving model to file for retrieval and analysis results replication.
- Both document-feature matrices and Procrustes analysis results are outputted to a hierarchy (more on hierarchy later).

# Setting Up (For Adding this Code into an Already Established Project)
Although this is common Python knowledge, I'm including this section for the sake of clarity and completeness.
After creating a project or having a project ready to use this module:
#### Import the Procrustes Analysis File:
1. Place the procrustes_analysis.py file into the same directory as the .py file that is the actual one being executed.
2. In the execution .py file (or notebook, if you're using that), import procrustes_analysis.py.

        import procrustes_analysis

    Note: To write more efficient code, you can use `import procrustes_analysis as pa` so you don't have to use the full filename to use the functions.
3. You can now use any function from this module within your Python code by using `pa`.

#### Install Necessary Packages for this Module:
Install the necessary packages through the requirements.txt file. When you place the requirements.txt file in the project folder, PyCharm *should* detect it, and give the option to install its packages.

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

# Phase Order
- Even though you *can* use these functions in any order after importing, this phases order must be kept in mind to use this module effectively:
    - Select the Reuters documents to analyze and preprocess them.
    - Setup and train a model.
    - Generate a document-feature matrix from that model.
    - Do Procrustes analysis between two document-feature matrices.

# How to Use (Example Using LSI)

#### (The three code snippets below must be done in this order):
Setup for LSI:
        
        number_of_documents = 50`
        document_collection = select_reuters_documents(number_of_documents)

Create LSI document-feature matrices:

        lsi_model = train_latent_semantic_indexing(generic_dictionary, generic_corpus, number_of_topics)
        lsi_vectorized = vectorize_model(lsi_model, generic_corpus)
        lsi_document_feature_matrix = create_document_feature_matrix(lsi_vectorized, number_of_documents, number_of_topics)

Save document-feature matrices to a file:

        save_document_feature_matrix_to_file(lsi_document_feature_matrix, 'lsi')

Save Procrustes Analysis to file:

        matrix1, matrix2, disparity = modified_procrustes(lsi_document_feature_matrix, lsi_document_feature_matrix)
        save_procrustes_analysis_to_file(matrix1, matrix2, disparity)

#### (You must make sure there are saved models that you can load *before* you load them):
When a model is created, this is how it can be saved to a file:

        save_model(lda_model, 'lda')

When a model has been outputted to a file, this is how it can be loaded into Python:
Note: Both the model string and timestamp are used to identify which model to load. The timestamp will need to be checked in the folder before you enter it as an argument.

        lda_model = load_model('lda', '2023.02.04T13.27.37Z')

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
        - Whatever other files the load method needs for the specific text analysis models.
    - ...

# Miscellaneous

- Python has a help() function that you can use. You can put in any function name from procrustes_analysis, and it'll print out the docstring. This is more useful in something like JupyterLab, but it's a feature, nonetheless.