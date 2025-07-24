from .base import ANNModel

from ..models import BaseLayer



class ParallelANNModel(ANNModel):

    def __init__(self, input_size: tuple, output_size: tuple, type_of_task: str, seed: int = None) -> None:
        super().__init__(input_size, output_size, type_of_task, seed)
    