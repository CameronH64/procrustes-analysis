# Author: Cameron Holbrook
# Purpose: Perform Procrustes analysis on two document-feature matrices.

import numpy as np
import nltk
import pprint                           # For neater printing of information
from nltk.corpus import reuters         # Import the reuters dataset (from the download function)

# ===================== DOWNLOAD REUTERS CORPORA =====================
# Use this code to download corpora.
# nltk.download()

# ===================== BRIEF OVERVIEW =====================
# 90 categories
# 10788 new documents, 1.3 million words.
# test/14826 is drawn from test set.

print(reuters.words('training/9865'))

