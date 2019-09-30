import os

import pandas as pd
import random


def get_dict(save_dir, name_col_key):
    df = pd.read_csv(save_dir)
    res = { }
    for item in range(len(df.index)):
        idx = df['id'].loc[item]
        if idx not in res.keys():
            res[idx] = []
        res[idx].append(df[name_col_key].loc[item])
    return res


def write_in_file(data, name_file):
    columns = ['titel', 'id']
    save_dir = './dataset/title/titel_tag_separate'
    f_path = os.path.join(save_dir, name_file)
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(f_path)


if __name__ == '__main__':
    sd = './dataset/title/title_tags.csv'
    key = 'title'
    test = get_dict(sd, key)  # type: dict
    train_lst = []
    test_lst = []
    dev_lst = []
    name_file = ['train.csv', 'dev.csv', 'test.csv']
    for item in test.keys():
        data = [[sent, item] for sent in test[item]]
        l_data = len(data)
        if l_data > 1000:
            random.shuffle(data)
            cnt_train = int(l_data * 0.7)
            cnt_dev = int(l_data * 0.9)
            train_lst += data[:cnt_train]  # тренировка 70
            dev_lst += data[cnt_train:cnt_dev]  # валидация 20
            test_lst += data[cnt_dev:]  # тест 10
    full_data = [train_lst, dev_lst, test_lst]
    for d, f_name in zip(full_data, name_file):
        write_in_file(d, f_name)
