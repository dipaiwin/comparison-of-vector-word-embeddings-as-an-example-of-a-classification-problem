from abc import ABC, abstractmethod
import numpy as np


class EmbeddingManager(ABC):
    def __init__(self, path_to_model):
        self.model = self.load_model(path_to_model)
        self.empty_vector = self.create_empty_vector()
    
    @abstractmethod
    def load_model(self, path_to_model):
        pass
    
    @property
    @abstractmethod
    def size_embedding(self):
        pass
    
    def create_empty_vector(self):
        return np.zeros([self.size_embedding])
    
    def __getitem__(self, word):
        return self.get_get_embedding(word)
    
    @abstractmethod
    def get_get_embedding(self, word):
        pass
