from .baselayer import BaseLayer

class FilterInputLayer(BaseLayer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_type(self):
        return "FilterInput"

    def is_trainable(self) -> bool:
        return False

    def _internal_compile(self, prev_layer, initializer):
        from ..engine import layers

        index= self.get_hyperparameter("index",default=None)
        if index is None:
            raise Exception("Index not set")
        
        branch = layers.Lambda(lambda x:x[:,:,index:index+1])(prev_layer)

        return branch