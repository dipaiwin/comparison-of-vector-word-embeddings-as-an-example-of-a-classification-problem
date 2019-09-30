import os


class FileManager:
    def __init__(self, args: dict):
        self.values = { }
        self['status'] = args['status']
        self['embed'] = args['embed']
        if self['status'] == 'train':
            self['train'] = os.path.join(args['dataset'], 'train.csv')
            self['dev'] = os.path.join(args['dataset'], 'dev.csv')
        else:
            self['test'] = os.path.join(args['dataset'], 'test.csv')
        self['model'] = os.path.join(args['save_dir'], 'model.ckpt')
        self['config'] = os.path.join(args['save_dir'], 'config.pickle')
    
    def __getitem__(self, item):
        return self.values[item]
    
    def __setitem__(self, key, value):
        self.values[key] = value
