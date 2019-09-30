import pickle
from ManagerEmbaddings.w2v_manager import W2VManager
from ManagerEmbaddings.ft_manager import FTManager
from ManagerEmbaddings.glove_manager import GloVeManager
from file_manager import FileManager
from config_project import ConfigProject
from dataset_loader import LoadDatasetManager
from batch_manager import BatchManager
from network.tester import Tester
from status_dataset import StatusDatasets
from network.trainer import Trainer
from log_manager import LogManager

args = {
    'status': 'train',
    'embed': '../embeddings/300/glove/glove.model',
    'dataset': './datasets/first_test',
    'save_dir': './model/300/glove',
    'embed_model': GloVeManager
}


def train(fm: FileManager):
    cfg = ConfigProject(fm, args['embed_model'])
    train_data, cfg['count_class'], cfg['classes'] = LoadDatasetManager(fm['train'], True).get_dataset()
    bm_train = BatchManager(train_data, cfg, StatusDatasets.Train)
    test_data = LoadDatasetManager(fm['dev'], dict_classes=cfg['classes']).get_dataset()
    bm_dev = BatchManager(test_data, cfg, StatusDatasets.Dev)
    with open(fm['config'], 'wb') as f:
        pickle.dump(cfg, f)
    Trainer(cfg, bm_train, bm_dev, path=fm['model']).run()


def test(fm: FileManager):
    with open(fm['config'], 'rb') as f:
        cfg = pickle.load(f)
    test_data = LoadDatasetManager(fm['test'], dict_classes=cfg['classes']).get_dataset()
    bm_test = BatchManager(test_data, cfg, StatusDatasets.Test)
    res = Tester(cfg, bm_test, fm['model']).run()
    LogManager.write_log(cfg, res, './logs/log.txt')


if __name__ == '__main__':
    print('Путь до embed:{0}\nПуть сохранения: {1}\nМодель: {2}'.format(args['embed'], args['save_dir'],
                                                                        args['embed_model']))
    fm = FileManager(args)
    if args['status'] == 'train':
        train(fm)
    else:
        test(fm)
