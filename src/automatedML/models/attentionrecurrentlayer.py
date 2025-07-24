from .baselayer import HyperparametersManager, BaseLayer

class AttentionRecurrentLayer(BaseLayer):
    def get_type(self):
        return "AttentionRecurrent"
    
    def is_trainable(self) -> bool:
        return True
    
    def _register_custom_hyperparameters(self, hp_manager: HyperparametersManager):
        self._register_hyperparameter("type", 
                                      hp_manager.hp_recurrent_types)
        self._register_hyperparameter("unit1", 
                                      hp_manager.hp_recurent_units)
        self._register_hyperparameter("unit2", 
                                      hp_manager.hp_recurent_units)
        self._register_hyperparameter("enable_attention", 
                                      hp_manager.hp_active)
        
    def _internal_compile(self, prev_layer, initializer):
        from ..engine import layers

        type = self.get_hyperparameter("type")
        unit1 = self.get_hyperparameter("unit1")
        unit2 = self.get_hyperparameter("unit2")
        enable_attention = self.get_hyperparameter("enable_attention")

        if len(prev_layer.shape) == 3:
            x = prev_layer
        else:
            raise Exception("Not supported: input not compatible")

        if type == "LSTM":
            x = layers.LSTM(units=unit1,
                            kernel_initializer=initializer,
                            return_sequences=True)(x)
            if enable_attention:
                x = layers.Attention()([x, x])
            x = layers.LSTM(units=unit2,
                            kernel_initializer=initializer)(x)
        elif type == "GRU":
            x = layers.GRU(units=unit1,
                           kernel_initializer=initializer,
                           return_sequences=True)(x)
            if enable_attention:
                x = layers.Attention()([x, x])
            x = layers.GRU(units=unit2,
                            kernel_initializer=initializer)(x)
        elif type == "FC":
            x = layers.SimpleRNN(units=unit1,
                                 kernel_initializer=initializer,
                                 return_sequences=True)(x)
            if enable_attention:
                x = layers.Attention()([x, x])
            x = layers.SimpleRNN(units=unit2,
                                kernel_initializer=initializer)(x)
        elif type == "BiLSTM":
            x = layers.Bidirectional(
                layers.LSTM(units=unit1,
                            kernel_initializer=initializer,
                            return_sequences=True)
            )(x)
            if enable_attention:
                x = layers.Attention()([x, x])
            x = layers.Bidirectional(
                layers.LSTM(units=unit2,
                            kernel_initializer=initializer)
            )(x)
        else:
            raise Exception("Type is not valid")

        return x
        
    