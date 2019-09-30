from gensim.models import FastText
from ManagerEmbaddings.embedding_manager import EmbeddingManager


class FTManager(EmbeddingManager):
    def load_model(self, path_to_model):
        model = FastText.load(path_to_model)
        return model
    
    @property
    def size_embedding(self):
        return self.model.vector_size
    
    def get_get_embedding(self, word):
        try:
            return self.model[word]
        except:
            return self.create_empty_vector()
