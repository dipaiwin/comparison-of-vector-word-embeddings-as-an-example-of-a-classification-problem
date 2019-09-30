from ManagerEmbaddings.w2v_manager import W2VManager
from ManagerEmbaddings.ft_manager import FTManager
from ManagerEmbaddings.glove_manager import GloVeManager
from config_project import ConfigProject


class LogManager:
    @staticmethod
    def write_log(cfg: ConfigProject, result: int, log_file: str):
        model_type = type(cfg['embed'])
        if model_type == W2VManager:
            model = 'w2v'
        elif model_type == FTManager:
            model = 'fasttext'
        else:
            model = 'glove'
        sg = ''
        if model in ['w2v', 'fasttext']:
            if cfg['embed'].model.sg == 1:
                sg = "sg"
            else:
                sg = "cbow"
        size = str(cfg['len(w2v_vector)'])
        with open(log_file, 'a', encoding='utf8') as f:
            f.write("model: {}\ttype:{}\tsize: {}\tresult: {}\n".format(model, sg, size, result))
