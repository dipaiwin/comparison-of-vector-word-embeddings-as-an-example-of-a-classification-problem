import numpy as np

from config_project import ConfigProject
from status_dataset import StatusDatasets


class CreaterBatch:
    def __init__(self, config: ConfigProject):
        self.__embed_manager = config['embed']
        self.cfg = config
    
    def create_batch(self, data):
        size = len(data)
        batch = {
            'w2v': np.zeros([size, self.cfg['len(Sent)'], self.cfg['len(w2v_vector)']],
                            np.float32
                            ),
            'answer': np.zeros([size], np.int32),
            'lengths': np.zeros(
                [size]),
            'origin_ids': np.zeros([size], np.int32)
            
        }
        for i, sentences in enumerate(data):
            words = sentences['words'][:self.cfg['len(Sent)']]
            matrix_w2v = np.zeros([self.cfg['len(Sent)'], self.cfg['len(w2v_vector)']])
            for idx, word in enumerate(words):
                matrix_w2v[idx] = self.__embed_manager[word]
            batch['w2v'][i] = matrix_w2v
            batch['answer'][i] = sentences['id']
            batch['origin_ids'][i] = sentences['id_origin']
            batch['lengths'][i] = len(words)
        return batch


class BatchManager:
    def __init__(self, datasets: list, config: ConfigProject, status: StatusDatasets):
        self.__datasets = datasets
        self.cfg = config
        self.__size = self.size_determination(status)
        self.__creater_bacth = CreaterBatch(self.cfg)
        self.__probs_cls = self.culc_prob()
    
    def size_determination(self, status: StatusDatasets):
        if status == StatusDatasets.Train:
            return self.cfg['size_train_batch']
        elif status == StatusDatasets.Dev:
            return self.cfg['size_dev_batch']
        else:
            return self.cfg['size_test_batch']
    
    def culc_prob(self):
        dct_cls = dict()
        for item in self.__datasets:
            if item['id'] not in dct_cls:
                dct_cls[item['id']] = 0
            dct_cls[item['id']] += 1.0
        cnt_class = self.cfg['count_class']
        for key in dct_cls.keys():
            dct_cls[key] = 1 / (dct_cls[key] * cnt_class)
        datasets_probs = []
        for item in self.__datasets:
            datasets_probs.append(dct_cls[item['id']])
        s_dp = sum(datasets_probs)
        if s_dp != 1:
            for idx in range(len(datasets_probs)):
                datasets_probs[idx] /= s_dp
        return datasets_probs
    
    def get_batch(self):
        return self.__creater_bacth.create_batch(
            np.random.choice(
                self.__datasets,
                self.__size,
                p=self.__probs_cls
            )
        )
