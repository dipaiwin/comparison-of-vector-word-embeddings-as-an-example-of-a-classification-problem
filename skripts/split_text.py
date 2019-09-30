import pandas as pd
import itertools
import os


def read_csv(filename):
    data = pd.read_csv(filename)
    return data


def create_dct(tup_tag_key, data: pd.DataFrame):
    key_text, key_tag = tup_tag_key
    test = data.loc[:, [key_text, key_tag]]
    t = list(set(pd.Series(test[key_tag]).values))
    ids = [t.index(test[key_tag].loc[i]) for i in range(len(test.index))]
    test['id'] = pd.Series(ids)
    return t, test


def write_in_file(full_texts: pd.DataFrame, head: tuple):
    save_dir = './dataset/'
    datasets_file = os.path.join(save_dir, head[0], '{0[0]}_{0[1]}.csv'.format(head))
    full_texts.to_csv(datasets_file)


def write_tags(full_tags, head: tuple):
    save_dir = './dataset/'
    print(full_tags)
    save_tags = os.path.join(save_dir, '{}.txt'.format(head[1]))
    with open(save_tags, 'w', encoding='utf8') as f:
        for idx, tag in enumerate(full_tags):
            f.write('{}:{}\n'.format(idx, tag))


if __name__ == '__main__':
    df = read_csv('./date/lenta-ru-news.csv')
    t = []
    heads = list(itertools.product(['title', 'text'], ['tags', 'topic']))
    for item in heads:
        tags, texts = create_dct(item, df)
        write_in_file(texts, item)
        if item[1] not in t:
            write_tags(tags, item)
            t.append(item[1])
