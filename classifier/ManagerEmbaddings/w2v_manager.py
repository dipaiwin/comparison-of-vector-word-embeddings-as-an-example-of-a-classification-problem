from gensim.models import Word2Vec
from ManagerEmbaddings.embedding_manager import EmbeddingManager


class W2VManager(EmbeddingManager):
    def load_model(self, path_to_model):
        model = Word2Vec.load(path_to_model)
        return model
    
    @property
    def size_embedding(self):
        return self.model.vector_size
    
    def get_get_embedding(self, word):
        if word in self.model.wv.vocab:
            return self.model[word]
        return self.create_empty_vector()