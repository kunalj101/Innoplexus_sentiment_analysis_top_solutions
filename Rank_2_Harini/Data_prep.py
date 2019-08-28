import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import re
import string
from nltk.tokenize import word_tokenize
import warnings

df_train = pd.read_csv('../input/train_F3WbcTw.csv')
df_test = pd.read_csv('../input/test_tOlRoBf.csv')

class CleanText(BaseEstimator, TransformerMixin):    
    def emoji_oneword(self, input_text):
        # By compressing the underscore, the emoji is kept as one word
        return input_text.replace('_','')
    def remove_urls(self, input_text):
        return re.sub(r'http.?://[^\s]+[\s]?', '', input_text)

    
    def remove_punctuation(self, input_text):
        seperator = ''
        # Make translation table
        input_text = re.sub('\n', '', input_text)
        input_text = re.sub('\xa0', '', input_text)
        punct = seperator.join([c for c in string.punctuation if c not in ['.']])+'“'+'”'
        trantab = str.maketrans(punct, len(punct)*' ')  # Every punctuation symbol will be replaced by a space
        return input_text.translate(trantab)
    def remove_digits(self, input_text):
        return re.sub('\d+', '', input_text)
    
    def to_lower(self, input_text):
        return input_text.lower()
    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X, **transform_params):
        clean_X = X.apply(self.remove_urls).apply(self.emoji_oneword).apply(self.remove_punctuation).apply(self.remove_digits).apply(self.to_lower)
        return clean_X
    
    
       
    
    
ct = CleanText()
df_train['text'] = ct.fit_transform(df_train.text)
df_test['text'] = ct.fit_transform(df_test.text)

df_train['drug'] = ct.fit_transform(df_train.drug)
df_test['drug'] = ct.fit_transform(df_test.drug)


df_train['text_modified'] = df_train[['text', 'drug']].apply(lambda x: re.sub(x.drug, '$T$', x.text), axis=1)

df_test['text_modified'] = df_test[['text', 'drug']].apply(lambda x: re.sub(x.drug, '$T$', x.text), axis=1)


def modify_sent(y):
    seperatior = '.'
    sent = seperatior.join([x for x in y.split('.') if '$T$' in x])
    if(sent!= ''):
        return sent
    else:
        return y
df_train['text_modified'] = df_train['text_modified'].apply(modify_sent)
df_test['text_modified'] = df_test['text_modified'].apply(modify_sent)

sentiment_dict = {2: 0,
                 1: -1,
                 0: 1}

with open('../input/train_sent.raw', 'w') as w:
    for idx, row in df_train.iterrows():
        w.write(row[4]+'\n')
        w.write(row[2]+'\n')
        w.write(str(sentiment_dict[row[3]])+'\n')
                
        
with open('../input/test_sent.raw', 'w') as w:
    for idx, row in df_test.iterrows():
        w.write(row[3]+'\n')
        w.write(row[2]+'\n')
        w.write(str(0)+'\n')
        
