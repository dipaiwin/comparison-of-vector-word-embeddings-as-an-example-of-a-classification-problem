import pickle
import json
import re
import nltk
import pandas as pd
import string


def decor_tokenizer(func):
    tokenizer = nltk.data.load('nltk_data/tokenizers/punkt/russian.pickle')
    
    def preproccesing_text(text: list):
        result = []
        for item in text:
            try:
                result.append(tokenizer.tokenize(item))
            except:
                continue
        return func(result)
    
    return preproccesing_text


def create_titels_texts(fname):
    df = pd.read_csv(fname)
    titels = []
    texts = []
    for idx in range(len(df.index)):
        titels.append(df['title'].loc[idx])
        texts.append(df['text'].loc[idx])
    return titels, texts


def processing_sentences(sentences: str):
    sentences = re.sub(r'\d+', 'numb', sentences)
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    sentences = regex.sub(' punct ', sentences)
    sentences = sentences.replace(u'\xa0', u' ')
    return re.findall(r'\w+', sentences.lower())


@decor_tokenizer
def processing_text(text):
    cache = []
    for item in text:
        for sent in item:
            words = processing_sentences(sent)
            cache.append(words)
    return cache


if __name__ == '__main__':
    filename = './date/lenta-ru-news.csv'
    print('Start read from csv')
    Titels, Texts = create_titels_texts(filename)
    print('Start tokinaze texts')
    res = processing_text(Texts)
    print('Start tokinaze titels')
    res += processing_text(Titels)
    with open('./pickle_data/list_sentences_lenta.pickle', 'wb') as f:
        pickle.dump(res, f)
