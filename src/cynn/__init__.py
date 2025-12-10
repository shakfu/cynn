from ._core import TinnNetwork, GenannNetwork, FannNetwork, FannNetworkDouble, CNNNetwork, CNNLayer, seed

# KANN neural network library
from .kann import (
    # Constants
    KANN_FLAG_IN,
    KANN_FLAG_OUT,
    KANN_FLAG_TRUTH,
    KANN_FLAG_COST,
    COST_BINARY_CROSS_ENTROPY,
    COST_MULTI_CROSS_ENTROPY,
    COST_BINARY_CROSS_ENTROPY_NEG,
    COST_MSE,
    RNN_NORM,
    RNN_VAR_H0,
    KAD_FLAG_VAR,
    KAD_FLAG_CONST,
    # Exceptions
    KannError,
    KannModelError,
    KannTrainingError,
    # Classes
    NeuralNetwork,
    GraphBuilder,
    DataSet,
    Array2D,
    # Functions
    set_seed as kann_set_seed,
    set_verbose as kann_set_verbose,
    one_hot_encode,
    one_hot_encode_2d,
    softmax_sample,
    prepare_sequence_data,
    list_to_2d_array,
)

__all__ = [
    # Tinn
    'TinnNetwork',
    # Genann
    'GenannNetwork',
    # FANN
    'FannNetwork',
    'FannNetworkDouble',
    # CNN/nn1
    'CNNNetwork',
    'CNNLayer',
    # Seed
    'seed',
    # KANN constants
    'KANN_FLAG_IN',
    'KANN_FLAG_OUT',
    'KANN_FLAG_TRUTH',
    'KANN_FLAG_COST',
    'COST_BINARY_CROSS_ENTROPY',
    'COST_MULTI_CROSS_ENTROPY',
    'COST_BINARY_CROSS_ENTROPY_NEG',
    'COST_MSE',
    'RNN_NORM',
    'RNN_VAR_H0',
    'KAD_FLAG_VAR',
    'KAD_FLAG_CONST',
    # KANN exceptions
    'KannError',
    'KannModelError',
    'KannTrainingError',
    # KANN classes
    'NeuralNetwork',
    'GraphBuilder',
    'DataSet',
    'Array2D',
    # KANN functions
    'kann_set_seed',
    'kann_set_verbose',
    'one_hot_encode',
    'one_hot_encode_2d',
    'softmax_sample',
    'prepare_sequence_data',
    'list_to_2d_array',
]