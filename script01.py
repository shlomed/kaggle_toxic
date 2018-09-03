
# coding: utf-8

# In[ ]:

import sys
import numpy as np
import re, string
import pandas as pd

from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin

get_ipython().magic('matplotlib inline')


# In[ ]:

sys.version_info


# ### Funcs

# In[ ]:

def iou(y_real, y_pred):
    return (y_real & y_pred).sum() / (y_real | y_pred).sum()


# In[ ]:

def accuracy(y_real, y_pred):
    return (y_real==y_pred).sum() / y_real.shape[0]


# In[ ]:

re_tok = re.compile('([%s])' % re.escape(string.punctuation+"“”¨«»®´·º½¾¿¡§£₤‘’"))
def tokenize(s): return re_tok.sub(r' \1 ', s).split()

tokenize("I am the right guy for you!")


# ### Read and Explore Train Data

# #### 1. Read:

# In[ ]:

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")


# In[ ]:

train.head(10)


# In[ ]:

train['comment_text'][0]


# In[ ]:

train['comment_text'][2]


# In[ ]:

# replace Nones by "unknown"
COMMENT = 'comment_text'
train[COMMENT].fillna("unknown", inplace=True)


# #### 2. Describe labels fracs and stats:

# In[ ]:

for col in train.columns[2:]:
    print(col, ":", train[col].sum()/train.shape[0])


# In[ ]:

# add none label and describe labels stats:
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train['none'] = 1-train[label_cols].max(axis=1)
train.iloc[:, 2:].describe()


# #### 3. Analyze Comments lengths

# In[ ]:

lens = train.comment_text.str.len() # number of chars
lens.mean(), lens.std(), lens.max()


# In[ ]:

lens.hist()


# #### 4. train/test split

# In[ ]:

X = train.comment_text
y = train.toxic


# In[ ]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ### Simple Binary Classification

# #### 0. Tokenization

# In[ ]:

# vect = CountVectorizer(ngram_range=(1,2))


# In[ ]:

n = X_train.shape[0]
vect = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize, min_df=3, max_df=0.9, strip_accents="unicode", 
                       use_idf=1, smooth_idf=1, sublinear_tf=1)
trn_term_doc = vect.fit_transform(X_train)
tst_term_doc = vect.transform(X_test)


# In[ ]:

trn_term_doc, tst_term_doc


# #### 1. Simple SVM

# In[ ]:

# clf = SVC()
# clf.fit(trn_term_doc, y_train)


# In[ ]:

# y_tst_pred = clf.predict(tst_term_doc)
# y_tst_proba = clf.predict_proba(tst_term_doc)


# #### 2. NB-SVM

# In[ ]:

def get_p_terms_given_label(x, labels, desired_label, soft_coef=1):
    n_occurances_for_term = x[labels==desired_label].sum(0)
    n_occurances_of_desired_label = (labels==desired_label).sum()
    return (n_occurances_for_term + soft_coef) / (n_occurances_of_desired_label + soft_coef)


# In[ ]:

class NB_SVM_binary(BaseEstimator):
    def __init__(self, C=4.):
        self.C = C
    def fit(self, x_train, y_train):
        try:
            y_train = y_train.values
        except:
            pass
        self.model = LogisticRegression(C=self.C, dual=True)
        pr_terms_given_1 = get_p_terms_given_label(x_train, y_train, 1)
        pr_terms_given_0 = get_p_terms_given_label(x_train, y_train, 0)
        self.r_ = np.log(pr_terms_given_1 / pr_terms_given_0)
        x_nb = x_train.multiply(self.r_)
        self.model.fit(x_nb, y_train)
        return self
    def predict_proba(self, x):
        return self.model.predict_proba(x.multiply(self.r_))[:, 1]
    def predict(self, x, ts=0.5):
        return self.model.predict_proba(x.multiply(self.r_))[:, 1]>ts
    def accuracy(self, x, y):
        try:
            y = y.values
        except:
            pass
        y_pred = self.predict(x).astype(int)
        y = y.astype(int)
        return accuracy(y, y_pred)
    def iou(self, x, y):
        try:
            y = y.values
        except:
            pass
        y_pred = self.predict(x).astype(int)
        y = y.astype(int)
        return iou(y, y_pred)
    def score(self, x, y, method="iou"):
        if method=="iou":
            return self.iou(x, y)
        elif method=="accuracy":
            return self.accuracy(x, y)
        


# In[ ]:

powers = np.arange(-5,5)
Cs = sorted([10.**i for i in powers] + [3*10.**i for i in powers])


# In[ ]:

acc_train = []
acc_test = []
iou_train = []
iou_test = []

for C in Cs:
    print("start %.5f"%(C))
    clf = NB_SVM_binary(C=C)
    clf.fit(trn_term_doc, y_train)
    iou_train.append(clf.iou(trn_term_doc, y_train))
    iou_test.append(clf.iou(tst_term_doc, y_test))
    acc_train.append(clf.accuracy(trn_term_doc, y_train))
    acc_test.append(clf.accuracy(tst_term_doc, y_test))


# In[ ]:

df_results = pd.DataFrame({"C":Cs, "acc_train":acc_train, "acc_test":acc_test, "iou_train":iou_train, "iou_test":iou_test})
df_results["logC"] = np.log(df_results["C"])


# In[ ]:

df_results.plot(x="logC", y=["acc_train", "acc_test"], grid=1, style="-o")


# In[ ]:

df_results.plot(x="logC", y=["iou_train", "iou_test"], grid=1, style="-o")


# #### 3. GridSearchCV over NB-SVM

# In[ ]:

parameters = {'C':[10, 3e1, 1e2, 3e2, 1e3]}
base_clf = NB_SVM_binary()
clf = GridSearchCV(base_clf, parameters, verbose=1)


# In[ ]:

get_ipython().run_cell_magic('time', '', 'clf.fit(trn_term_doc, y_train)')


# In[ ]:

y_proba = clf.predict_proba(tst_term_doc)
y_pred = clf.predict(tst_term_doc)


# In[ ]:

iou(y_pred, y_test), accuracy(y_pred, y_test)


# In[ ]:

clf.best_estimator_.get_params()


# #### 4. LSTM, Glove, Dropout

# In[ ]:

import sys, os, re, csv, codecs, numpy as np, pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers


# In[ ]:

path = '../input/'
comp = 'jigsaw-toxic-comment-classification-challenge/'
EMBEDDING_FILE='word_embeddings/glove.6B.50d.txt'


# In[ ]:

embed_size = 50        # how big is each word vector
max_features = 20000   # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100           # max number of words in a comment to use


# In[ ]:

train.head()


# In[ ]:

list_sentences_train = train["comment_text"].fillna("_na_").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("_na_").values


# In[ ]:

tokenizer = Tokenizer(num_words=max_features) # keras tokenizer
tokenizer.fit_on_texts(list(list_sentences_train))


# In[ ]:

list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)


# In[ ]:

# hist of lengths of texts in train
pd.Series([len(i) for i in list_tokenized_train]).hist(alpha=0.5, bins=50)
pd.Series([len(i) for i in list_tokenized_test]).hist(alpha=0.5, bins=50)


# In[ ]:

X_train = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_test = pad_sequences(list_tokenized_test, maxlen=maxlen)
X_train.shape, X_test.shape


# In[ ]:

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE, encoding="utf8"))


# In[ ]:

len(embeddings_index)


# In[ ]:

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
emb_mean,emb_std


# In[ ]:

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


# In[ ]:

inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
x = GlobalMaxPool1D()(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(6, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:

model.fit(X_train, y, batch_size=32, epochs=2, validation_split=0.1);

