
# coding: utf-8

# In[1]:


import sys
import re, string
import pandas as pd

from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


sys.version_info


# ### Funcs

# In[3]:


def iou(y_real, y_pred):
    return (y_real & y_pred).sum() / (y_real | y_pred).sum()


# In[4]:


re_tok = re.compile('([%s])' % re.escape(string.punctuation+"“”¨«»®´·º½¾¿¡§£₤‘’"))
def tokenize(s): return re_tok.sub(r' \1 ', s).split()

tokenize("I am the right guy for you!")


# ### Read and Explore Train Data

# #### 1. Read:

# In[5]:


train = pd.read_csv("data/train.csv")


# In[6]:


train.head(10)


# In[7]:


train['comment_text'][0]


# In[8]:


train['comment_text'][2]


# In[9]:


# replace Nones by "unknown"
COMMENT = 'comment_text'
train[COMMENT].fillna("unknown", inplace=True)


# #### 2. Describe labels fracs and stats:

# In[10]:


for col in train.columns[2:]:
    print(col, ":", train[col].sum()/train.shape[0])


# In[11]:


# add none label and describe labels stats:
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train['none'] = 1-train[label_cols].max(axis=1)
train.iloc[:, 2:].describe()


# #### 3. Analyze Comments lengths

# In[12]:


lens = train.comment_text.str.len() # number of chars
lens.mean(), lens.std(), lens.max()


# In[13]:


lens.hist()


# ### Tokenization

# ### Simple Binary Classification

# In[14]:


X = train.comment_text
y = train.toxic


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)


# #### 0. Tokenization

# In[16]:


# vect = CountVectorizer(ngram_range=(1,2))


# In[17]:


n = X_train.shape[0]
vect = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize, min_df=3, max_df=0.9, strip_accents="unicode", 
                       use_idf=1, smooth_idf=1, sublinear_tf=1)
trn_term_doc = vect.fit_transform(X_train)
tst_term_doc = vect.transform(X_test)


# In[18]:


trn_term_doc, tst_term_doc


# #### 1. Simple SVM

# In[ ]:


clf = SVC()
clf.fit(trn_term_doc, y_train)


# In[ ]:


y_tst_pred = clf.predict(tst_term_doc)
y_tst_proba = clf.predict_proba(tst_term_doc)


# #### 2. NB-SVM

# In[26]:


def get_p_terms_given_label(x, labels, desired_label, soft_coef=1):
    n_occurances_for_term = x[labels==desired_label].sum(0)
    n_occurances_of_desired_label = (labels==desired_label).sum()
    return (n_occurances_for_term + soft_coef) / (n_occurances_of_desired_label + soft_coef)


# In[30]:


class NB_SVM_binary():
    def __init__(self, model=LogisticRegression(C=4., dual=True), ):
        self.model = model
    def fit(self, x_train, y_train):
        pr_terms_given_1 = get_p_terms_given_label(x_train, y_train, 1)
        pr_terms_given_0 = get_p_terms_given_label(x_train, y_train, 0)
        self.r_ = np.log(pr_terms_given_1 / pr_terms_given_0)
        x_nb = x.multiply(self.r_)
        self.model.fit(self, x_nb, y_train)
        return self
    def predict_proba(self, x):
        return self.model.predict_proba(x.multiply(self.r_))[:, 1]


# In[31]:


clf = NB_SVM_binary()


# In[32]:


clf.fit(trn_term_doc, y_train)

