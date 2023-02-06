# Procrustes Analysis
This python file has a variety of functions for these purposes:
- Preparation of a Reuters corpus.
- Model training on a corpus.
- Procrustes analysis between document-feature matrices.
- Saving model to file for retrieval and analysis results replication.
- Both document-feature matrices and Procrustes analysis results are outputted to a hierarchy (more on hierarchy later).

# Design Choices
- When this module is imported, you can freely use all of the functions defined within it.
- There are three phases to use this module effectively:
    - Select the Reuters documents to analyze and preprocess them.
    - Setup, train a model, and create a document-feature matrix from that model.
    - Do Procrustes analysis between two document-feature matrices.

# Setting Up
1. Place the .py file into the same directory as another .py file that is executed.
2. In the execution .py file, import procrustes_analysis.py.
    `import procrustes_analysis`
    Note: To write more efficient code, you can use `import procrustes_analysis as pa` so you don't have to use the full filename to use the functions.
3. You can now use any function from this module within your Python code by using `pa`.

# How to Use (Example Using LSI)

#### (The three code snippets below must be done in this order):
Setup for LSI:\
`number_of_documents = 50`\
`document_collection = select_reuters_documents(number_of_documents)`\

Create LSI document-feature matrices:\
`lsi_model = train_latent_semantic_indexing(generic_dictionary, generic_corpus, number_of_topics)`\
`lsi_vectorized = vectorize_model(lsi_model, generic_corpus)`\
`lsi_document_feature_matrix = create_document_feature_matrix(lsi_vectorized, number_of_documents, number_of_topics)`\

Save document-feature matrices to a file:\
`save_document_feature_matrix_to_file(lsi_document_feature_matrix, 'lsi')`\

Save Procrustes Analysis to file:\
`matrix1, matrix2, disparity = modified_procrustes(lsi_document_feature_matrix, lsi_document_feature_matrix)`\
`save_procrustes_analysis_to_file(matrix1, matrix2, disparity)`\

#### (The two code snippets below must be done in this order):
When a model is created, this is how it can be saved to a file:\
`save_model(lda_model, 'lda')`\

When a model has been outputted to a file, this is how it can be loaded into Python:\
Note: Both the model string and timestamp are used to identify which model to load. The timestamp will need to be checked in the folder before you enter it as an argument.\
`lda_model = load_model('lda', '2023.02.04T13.27.37Z')`\

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
