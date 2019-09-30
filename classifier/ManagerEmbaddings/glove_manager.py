from ManagerEmbaddings.embedding_manager import EmbeddingManager
import pickle


class GloVeManager(EmbeddingManager):
    def __init__(self, path_to_model):
        self.__vector_size = None
        super().__init__(path_to_model)
    
    def load_model(self, path_to_model):
        with open(path_to_model, 'rb') as f:
            model = pickle.load(f)
        return model
    
    @property
    def size_embedding(self):
        if self.__vector_size is None:
            self.__vector_size = len(self.model[list(self.model.keys())[0]])
        return self.__vector_size
    
    def get_get_embedding(self, word):
        if word in self.model.keys():
            return self.model[word]
        else:
            return self.create_empty_vector()
