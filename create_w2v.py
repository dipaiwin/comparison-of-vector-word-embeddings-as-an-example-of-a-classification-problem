import pickle
import os
import re
import pymorphy2
import gensim
from gensim.models import FastText
from gensim.models import Word2Vec
from glove import Corpus, Glove
from gensim.corpora.wikicorpus import WikiCorpus


def decor_count(func):
    my_dir = './pickle_data/test'
    max_count = len(os.listdir(my_dir))
    name = 'part'
    id_file = 0
    
    def update_count():
        nonlocal id_file
        if id_file < max_count:
            fm = '{}_{}.pickle'.format(name, id_file)
            id_file += 1
            file_name = os.path.join(my_dir, fm)
            return func(file_name)
        else:
            return func(None)
    
    return update_count


@decor_count
def get_batch(f_name=None):
    if f_name:
        with open(f_name, 'rb') as f:
            res = pickle.load(f)
        return res
    else:
        return None


def train_embed(texts, save_model, embed_cls, sg=0, size=300):
    model = embed_cls(size=size, sg=sg, workers=10)
    model.build_vocab(sentences=texts)
    model.train(texts, total_examples=model.corpus_count, epochs=model.epochs)
    model.save(save_model)


def test_model_embed(test_model):
    print(len(test_model.wv.vocab))
    while True:
        a = input('Введите слово: ')
        print(test_model.most_similar(positive=[a]))


def get_full_corpus():
    date = []
    while True:
        b = get_batch()
        if b is not None:
            date += b
        else:
            break
    return date


def train_glove(save_dir, size):
    print('START')
    f_corpus = get_full_corpus()
    corpus = Corpus()
    print('CREATE CORPUS')
    corpus.fit(f_corpus, window=10)
    word_dict = corpus.dictionary.keys()
    glove = Glove(no_components=size, learning_rate=0.05)
    print('START LEARNING')
    glove.fit(corpus.matrix, epochs=60, no_threads=8, verbose=True)
    glove.add_dictionary(corpus.dictionary)
    dict_in_bin = dict()
    print('START SAVE')
    for item in word_dict:
        word_indx = glove.dictionary[item]
        dict_in_bin[item] = glove.word_vectors[word_indx]
    with open(save_dir, "wb") as file:
        pickle.dump(dict_in_bin, file)
    print('COMMON TEST')
    while True:
        try:
            s = input("Введите строку: ")
            print(glove.most_similar(s, number=10))
            word_indx = glove.dictionary[s]
            print(glove.word_vectors[word_indx])
        except:
            continue


def get_text():
    wiki = WikiCorpus('ruwiki-20181020-pages-articles-multistream.xml.bz2 ')
    for text in wiki.get_texts():
        yield [word for word in text]


def manager_train(args):
    b = get_text()
    if args['model'] == 'w2v':
        train_embed(b, args['save_path'], Word2Vec, args['sg'], args['size'])
        model = gensim.models.Word2Vec.load(args['save_path'])
        test_model_embed(model)
    elif args['model'] == 'fasttext':
        train_embed(b, args['save_path'], FastText, args['sg'], args['size'])
        model = gensim.models.FastText.load(args['save_path'])
        test_model_embed(model)
    else:
        train_glove(args['save_path'], args['size'])


if __name__ == '__main__':
    args = {
        'model': 'fasttext',
        'sg': 'cbow',
        'size': 300
    }
    if args['model'] == 'glove':
        args['sg'] = ''
    args['save_path'] = os.path.join('./embeddings/wiki', str(args['size']), args['model'], args['sg'],
                                     '{}.model'.format(args['model']))
    print(args['save_path'])
    args['sg'] = 1 if args['sg'] == 'skip-gramm' else 0
    manager_train(args)
