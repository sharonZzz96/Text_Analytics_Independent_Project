
# coding: utf-8

import numpy as np
import pandas as pd
import ast 
import warnings
warnings.filterwarnings('ignore')
import spacy
from nltk import Tree
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords 
from nltk.tokenize import sent_tokenize, word_tokenize
en_nlp = spacy.load('en')
from nltk.stem.lancaster import LancasterStemmer
st = LancasterStemmer()
import xgboost as xgb
import pickle
from textblob import TextBlob
from scipy import spatial
import joblib
import json


def load_data(question_path, context_path):
    """Load data """
    with open(question_path, 'r') as f:
        question = f.read()
    with open (context_path, 'r') as f:
        context = f.read()
    df_list = [[context, question]]
    df = pd.DataFrame(df_list)
    df.columns = ['context', 'question']
    return df


def embedding(df, infersent_path):
    """Embed each sentence and question using Infersent """
    
    # Load our pre-trained model
    with open(infersent_path, 'rb') as f:
        infersent = pickle.load(f)

    # embedding question
    dict_embeddings = {}
    question = df["question"].tolist()[0]
    try:
        dict_embeddings[question] = infersent.encode([question], tokenize=True)
    except:
        print(question)
        print('infersent cannot embed it')

    # embedding context
    paras = df["context"].unique().tolist()
    blob = TextBlob(" ".join(paras))
    sentences = [item.raw for item in blob.sentences]
    for i in range(len(sentences)):
        try:
            dict_embeddings[sentences[i]] = infersent.encode([sentences[i]], tokenize=True)
        except:
            print(sentence)
            print('infersent cannot embed it')
    
    return dict_embeddings


def process_data(df, dict_emb):
    """Turn each sentence in the context and question embeddings as df columns"""

    df['sentences'] = df['context'].apply(lambda x: [item.raw for item in TextBlob(x).sentences])
    df['sent_emb'] = df['sentences'].apply(lambda x: [dict_emb[item][0] if item in                                                           dict_emb else np.zeros(4096) for item in x])
    df['quest_emb'] = df['question'].apply(lambda x: dict_emb[x] if x in dict_emb else np.zeros(4096) )

    return df


def common_words(df):
    """Count identical words between question and each sentence as a list """

    ps = PorterStemmer()
    stop_words = set(stopwords.words('english')) 

    question = df['question'].lower()
    sentences = en_nlp(df["context"].lower()).sents

    question_word = word_tokenize(question)
    
    # create common word list
    num_common_word = []
    for i, sent in enumerate(sentences):
        sent_word = word_tokenize(str(sent))
        
        num = len(list(set(question_word) & set(sent_word) - stop_words))
        num_common_word.append(num)
        
    diff = 10 - len(num_common_word)
    num_common_word.extend([None] * diff)
            
    return num_common_word


def get_columns_from_common_word(train):
    """Create 10 columns indicating the number of common words between question and each sentence"""
    
    columns = ['column_common_0', 'column_common_1', 'column_common_2', 'column_common_3', 'column_common_4',
              'column_common_5', 'column_common_6', 'column_common_7', 'column_common_8', 'column_common_9']
    tmp = pd.DataFrame(columns=columns)
    
    for i in range(len(train)):
        tmp.loc[i,] = train['num_common_word'][i][:10]

    return tmp


def cosine_sim(x):
    li = []
    for item in x["sent_emb"]:
        li.append(spatial.distance.cosine(item,x["quest_emb"][0]))
    return li


def calculate_distance(df):   
    """Calculate Euclidean distance and Cosine similarity of question with each sentence as a list"""

    df["cosine_sim"] = df.apply(cosine_sim, axis = 1)
    df["diff"] = (df["quest_emb"] - df["sent_emb"])**2
    df["euclidean_dis"] = df["diff"].apply(lambda x: list(np.sum(x, axis = 1)))
    
    return df


def semantic_similarity(data):
    """Create 10 columns of Euclidean distance and Cosine similarity of question with each sentence as features"""
    
    train = pd.DataFrame()
     
    for k in range(len(data["euclidean_dis"])):
        dis = data["euclidean_dis"][k]
        dis.extend([None] * (10-len(dis)))
        for i in range(10):
            train.loc[k, "column_euc_"+"%s"%i] = dis[i]
    
    for k in range(len(data["cosine_sim"])):
        dis = data["cosine_sim"][k]
        dis.extend([None] * (10-len(dis)))
        for i in range(10):
            train.loc[k, "column_cos_"+"%s"%i] = dis[i]
        
    return train


def match_roots(x):
    """Find the index of the sentence in the context with matched root with the question"""
    
    question = x["question"].lower()
    sentences = en_nlp(x["context"].lower()).sents
    
    question_root = st.stem(str([sent.root for sent in en_nlp(question).sents][0]))
    
    li = []
    for i,sent in enumerate(sentences):
        roots = [st.stem(chunk.root.head.text.lower()) for chunk in sent.noun_chunks]

        if question_root in roots: 
            li.append(i)
            
    return li


def get_columns_from_root(train):
    """Create 10 columns of root match indicating whether each sentence share root with the question"""
    
    columns = ['column_root_0', 'column_root_1', 'column_root_2', 'column_root_3', 'column_root_4',
              'column_root_5', 'column_root_6', 'column_root_7', 'column_root_8', 'column_root_9']
    tmp = pd.DataFrame(columns=columns)
    
    for i in range(10):
        tmp.loc[0, 'column_root_'+'%s'%i] = 0
    for item in train['root_match_idx'][0]:
        tmp.loc[i, "column_root_"+"%s"%item] = 1
    
    return tmp


# read and process data
df = load_data('data/question.txt', 'data/context.txt')
dict_emb = embedding(df, 'infersent.pkl')
df = process_data(df, dict_emb)

# check if context has <= 10 sentences
df_len_before = len(df)
df = df[df["sentences"].apply(lambda x: len(x))<11].reset_index(drop=True)
df_len_after = len(df)
assert df_len_before == df_len_after

# baseline feature
df['num_common_word'] = df.apply(common_words, axis = 1)
baseline_feature = get_columns_from_common_word(df)
baseline_feature = baseline_feature.fillna(0)

# semantic similarity feature
df = calculate_distance(df)
sem_similar_feature_tmp = semantic_similarity(df)
# fill NA using max
subset1 = sem_similar_feature_tmp.iloc[:,:10].fillna(41)
subset2 = sem_similar_feature_tmp.iloc[:,10:].fillna(1.5)
sem_similar_feature = pd.concat([subset1, subset2],axis=1, join_axes=[subset1.index])

# root match feature
df["root_match_idx"] = df.apply(match_roots, axis = 1)
root_feature = get_columns_from_root(df)
root_feature.fillna(0, inplace=True)

# combine feature
data = pd.concat([baseline_feature, sem_similar_feature, root_feature], axis=1, join_axes=[baseline_feature.index])

# load model and predict
xgb_model = joblib.load('model/xgb_model.sav')
y_pred_index = xgb_model.predict(data)[0]
pred_sent = df['sentences'].tolist()[0][y_pred_index]
print('the sentence with the answer is:', pred_sent)
# save prediction to json
with open('prediction_result.json', 'w') as fp:
    json.dump(pred_sent, fp)

