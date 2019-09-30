import re
import string
import random
import pandas as pd


class LoadDatasetManager:
    def __init__(self, path_to_data, return_count_class=False, dict_classes: dict = None):
        if dict_classes is not None:
            self.dict_classes = dict_classes
        else:
            self.dict_classes = dict()
        self.path_data = path_to_data
        self.__return_cnt_cls = return_count_class
    
    @staticmethod
    def tokinaze_text(text):
        text = re.sub(r'\d+', 'numb', text)
        regex = re.compile('[%s]' % re.escape(string.punctuation))
        text = regex.sub(' punct ', text)
        text = text.replace(u'\xa0', u' ')
        return re.findall(r'\w+', text.lower())
    
    def get_dataset(self):
        data = []
        bad_class = [3, 7, 9]
        # bad_class = []
        df = pd.read_csv(self.path_data)
        len_sum = 0
        cnt_class = dict()
        cnt_ex = len(df.index)
        for item in range(len(df.index)):
            id_text = df['id'].loc[item]
            if id_text not in cnt_class:
                cnt_class[id_text] = 0
            cnt_class[id_text] += 1
            texts = self.tokinaze_text(df['titel'].loc[item])
            texts = self.create_typos(texts)
            if id_text not in bad_class:
                if id_text not in self.dict_classes:
                    self.dict_classes[id_text] = len(self.dict_classes)
                len_sum += len(texts)
                data.append(
                    {
                        'id_origin': id_text,
                        'id': self.dict_classes[id_text],
                        'words': texts
                    }
                )
        s = 0
        for key in cnt_class:
            s +=cnt_class[key] / cnt_ex
            print(key, cnt_class[key] / cnt_ex)
        print(s)
        input()
        if self.__return_cnt_cls:
            return data, len(self.dict_classes), self.dict_classes
        else:
            return data
    
    def create_typos(self, tokens):
        abc = 'йцукенгшщзхъэждлорпавыфячсмитьбю'
        len_abc = len(abc) - 1
        clean_tokens = [(ind, item) for ind, item in enumerate(tokens) if item != 'numb' and item != 'punct']
        count_bad_word = int(len(clean_tokens) * 0.3)
        random.shuffle(clean_tokens)
        clean_words = clean_tokens[0:count_bad_word]
        for pair in clean_words:
            word = list(pair[1])
            index = pair[0]
            option = random.randint(1, 3)
            rnd_index = random.randint(0, len(word) - 1)
            if option == 1:
                word[rnd_index] = abc[random.randint(0, len_abc)]
            elif option == 2:
                word[rnd_index] = ''
            else:
                word.insert(rnd_index, abc[random.randint(0, len_abc)])
            tokens[index] = ''.join(word)
        return tokens
