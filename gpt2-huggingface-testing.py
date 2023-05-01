# ----------------------- DESCRIPTION -----------------------

# I wasn't able to properly implement GPT in my Procrustes analysis module.
# I have some code here that is close, but I did not want to include it in the final module because "close enough" is not good enough.
# However, to show that I did make some leeway on GPT, and maybe help the next programmers, I have included this .py file in the Git repository.
# Also, If this HuggingFace GPT model can work, it may be preferable to use the HuggingFact BERT model as well, for consistency sake.

# The original goal was this:
# 1. Make a GPT model.
# 2. Fine-tune that GPT model on the Reuters corpus.
# 3. Extract document embeddings from that GPT model.

# However, I was only able to do the code within this file:
# 1. Make a GPT model.
# 2. Extract document embedding from that GPT model.

# The problem is that this GPT model isn't being fine-tuned. Therefore, the embeddings, as far as I know, are
# not proper embeddings that would want to be used for Procrustes analysis.

# Also, I'm not sure if BERTopic was the preferred BERT model to be used in this project. However, I went with it because
# I was able to fine-tune it (using the .fit method).
# With that said, because these GPT and BERT models are both from HuggingFace, they are quite similar. So, it wasn't
# too much trouble to find an alternative BERT method. Again, as stated above, maybe BERT could be considered to be plugged into the Procrustes Analysis module.

# Final note: These GPT and BERT models themselves are quite big; they're either hundreds of megabytes or even a couple gigabytes.

# - Cameron Holbrook




# ----------------------- GPT EMBEDDINGS -----------------------

# 1. Make the GPT model.
# from transformers import GPT2Tokenizer, GPT2Model           # Import the GPT stuff from transformers
#
# models = ['gpt2', 'gpt2-medium', 'gpt2-large']              # For testing, there are multiple GPT2 models you can use.
#
# model_name = models[0]                                      # Easy model selection for testing.
#
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)       # Make a GPT2Tokenizer object.
#
# model = GPT2Model.from_pretrained(model_name)               # Make a GPT2Model object.
#
#
# document = "This is a sample document that we want to encode using a GPT model."        # This is testing document. The embedding code will want to be cycled through for each document.
# tokenized_doc = tokenizer.encode(document, add_special_tokens=False)                    # Encode the document so that it can be used for the GPT2 model.
#
#
# # 2. Extract the embeddings.
# import torch
#
# input_ids = torch.tensor([tokenized_doc])                       # Convert tokenized_doc to a pytorch tensor object?
#
# with torch.no_grad():
#     output = model(input_ids)
#
# document_embedding = output.last_hidden_state.mean(dim=1)       # This is the magical step where you extract the GPT2 document embeddings. Bear in mind that this does not fine-tune the GPT2 model.
#
# document_embedding = document_embedding.numpy()                 # For the sake of Procrustes analysis, the document embeddings need to be a numpy array.
#
# print(f'{model_name} Embedding:')                               # For testing. Will show the document embeddings that were extracted.
# for count, value in enumerate(document_embedding[0]):
#     print(count+1, value)





# ----------------------- BERT EMBEDDINGS -----------------------

# Comments are basically the same as GPT2 above.

# from transformers import BertTokenizer, BertModel
# import torch
#
# # Load the tokenizer and model
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')
#
# # Define a text sequence
# text = "This is a sample input sentence."
#
# # Tokenize the text and convert to tensor
# inputs = tokenizer(text, return_tensors='pt')
#
# # Generate the document embedding
# outputs = model(**inputs)
# document_embedding = outputs.last_hidden_state.mean(dim=1)
#
# # Convert the document embedding to a NumPy array
# document_embedding_numpy = document_embedding.detach().numpy()
#
#
# print('BERT Embedding:')
# for count, value in enumerate(document_embedding_numpy[0]):
#     print(count+1, value)
#
#
