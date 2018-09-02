
# coding: utf-8

# In[43]:

import sys
import re, string
import pandas as pd

from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split


get_ipython().magic('matplotlib inline')


# In[45]:

sys.version_info


# ### Funcs

# In[14]:

def iou(y_real, y_pred):
    return (y_real & y_pred).sum() / (y_real | y_pred).sum()


# In[36]:

re_tok = re.compile('[%s]' % re.escape(string.punctuation+"“”¨«»®´·º½¾¿¡§£₤‘’"))
def tokenize(s): return re_tok.sub(r' \1 ', s).split()


# ### Read and Explore Train Data

# #### 1. Read:

# In[15]:

train = pd.read_csv("data/train.csv")


# In[16]:

train.head(10)


# In[17]:

train['comment_text'][0]


# In[18]:

train['comment_text'][2]


# In[19]:

# replace Nones by "unknown"
COMMENT = 'comment_text'
train[COMMENT].fillna("unknown", inplace=True)


# #### 2. Describe labels fracs and stats:

# In[20]:

for col in train.columns[2:]:
    print(col, ":", train[col].sum()/train.shape[0])


# In[21]:

# add none label and describe labels stats:
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train['none'] = 1-train[label_cols].max(axis=1)
train.iloc[:, 2:].describe()


# #### 3. Analyze Comments lengths

# In[22]:

lens = train.comment_text.str.len() # number of chars
lens.mean(), lens.std(), lens.max()


# In[23]:

lens.hist()


# In[42]:




# ### Tokenization

# In[40]:

tokenize("I am the right guy for you!")


# In[34]:




# ### Simple Binary Classification

# In[ ]:

X = train.comment_text
y = train.toxic


# In[ ]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)


# #### 1. Simple SVM

# In[ ]:

vect = CountVectorizer(ngram_range=(1,2))
XX_train = vect.fit_transform(X_train)


# In[ ]:

clf = SVC()
clf.fit(XX_train, y_train)


# In[ ]:



