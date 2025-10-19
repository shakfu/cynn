"""Type stubs for cynn._core module."""

from typing import ClassVar, Sequence
from os import PathLike

def square(x: float) -> float:
    """Square a float value."""
    ...

def seed(seed_value: int = 0) -> None:
    """
    Seed the C random number generator used for weight initialization.

    Args:
        seed_value: Random seed (if 0, uses current time)

    Note:
        Call this before creating networks for reproducible results.
        The Tinn library uses C's rand() for weight initialization.
    """
    ...

class TinnNetwork:
    """
    A three-layer neural network (input, hidden, output).

    This class wraps the Tinn C library, providing a Python interface
    for creating, training, and using neural networks.
    """

    def __init__(self, inputs: int = 0, hidden: int = 0, outputs: int = 0) -> None:
        """
        Create a new neural network.

        Args:
            inputs: Number of input neurons (must be positive if provided)
            hidden: Number of hidden layer neurons (must be positive if provided)
            outputs: Number of output neurons (must be positive if provided)

        Raises:
            ValueError: If any dimension is <= 0 when creating a new network
        """
        ...

    @property
    def input_size(self) -> int:
        """Number of input neurons."""
        ...

    @property
    def hidden_size(self) -> int:
        """Number of hidden layer neurons."""
        ...

    @property
    def output_size(self) -> int:
        """Number of output neurons."""
        ...

    @property
    def shape(self) -> tuple[int, int, int]:
        """Network shape as (inputs, hidden, outputs)."""
        ...

    def train(
        self,
        inputs: Sequence[float] | memoryview,
        targets: Sequence[float] | memoryview,
        rate: float
    ) -> float:
        """
        Train the network on one example using backpropagation.

        Supports any object implementing the buffer protocol (lists, tuples,
        array.array, numpy arrays, etc.) containing float32 values.

        Args:
            inputs: Input values (length must match input_size).
                    Can be any buffer-compatible object with float32 dtype.
            targets: Target output values (length must match output_size).
                     Can be any buffer-compatible object with float32 dtype.
            rate: Learning rate (typically 0.0 to 1.0)

        Returns:
            Training error for this example

        Raises:
            ValueError: If inputs or targets have wrong length
            TypeError: If buffer types are incompatible
        """
        ...

    def predict(self, inputs: Sequence[float] | memoryview) -> list[float]:
        """
        Make a prediction given input values.

        Supports any object implementing the buffer protocol (lists, tuples,
        array.array, numpy arrays, etc.) containing float32 values.

        Args:
            inputs: Input values (length must match input_size).
                    Can be any buffer-compatible object with float32 dtype.

        Returns:
            List of output values (length matches output_size)

        Raises:
            ValueError: If inputs has wrong length
            TypeError: If buffer type is incompatible
        """
        ...

    def save(self, path: str | bytes | PathLike[str] | PathLike[bytes]) -> None:
        """
        Save the network weights to a file.

        Args:
            path: File path where the network will be saved

        Raises:
            TypeError: If path is not str, bytes, or PathLike
        """
        ...

    @classmethod
    def load(cls, path: str | bytes | PathLike[str] | PathLike[bytes]) -> TinnNetwork:
        """
        Load a network from a file.

        Args:
            path: File path from which to load the network

        Returns:
            Loaded TinnNetwork instance

        Raises:
            TypeError: If path is not str, bytes, or PathLike
        """
        ...

class GenannNetwork:
    """
    A multi-layer neural network using the GENANN library.

    This class wraps the GENANN C library, providing a Python interface
    for creating, training, and using neural networks with arbitrary depth.
    """

    def __init__(self, inputs: int = 0, hidden_layers: int = 0, hidden: int = 0, outputs: int = 0) -> None:
        """
        Create a new neural network.

        Args:
            inputs: Number of input neurons (must be positive if provided)
            hidden_layers: Number of hidden layers (must be positive if provided)
            hidden: Number of neurons per hidden layer (must be positive if provided)
            outputs: Number of output neurons (must be positive if provided)

        Raises:
            ValueError: If any dimension is <= 0 when creating a new network
            MemoryError: If network allocation fails
        """
        ...

    @property
    def input_size(self) -> int:
        """Number of input neurons."""
        ...

    @property
    def hidden_layers(self) -> int:
        """Number of hidden layers."""
        ...

    @property
    def hidden_size(self) -> int:
        """Number of neurons per hidden layer."""
        ...

    @property
    def output_size(self) -> int:
        """Number of output neurons."""
        ...

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """Network shape as (inputs, hidden_layers, hidden, outputs)."""
        ...

    @property
    def total_weights(self) -> int:
        """Total number of weights in the network."""
        ...

    @property
    def total_neurons(self) -> int:
        """Total number of neurons plus inputs."""
        ...

    def train(
        self,
        inputs: Sequence[float] | memoryview,
        targets: Sequence[float] | memoryview,
        rate: float
    ) -> None:
        """
        Train the network on one example using backpropagation.

        Supports any object implementing the buffer protocol (lists, tuples,
        array.array, numpy arrays, etc.) containing float64 values.

        Args:
            inputs: Input values (length must match input_size).
                    Can be any buffer-compatible object with float64 dtype.
            targets: Target output values (length must match output_size).
                     Can be any buffer-compatible object with float64 dtype.
            rate: Learning rate (typically 0.0 to 1.0)

        Raises:
            ValueError: If inputs or targets have wrong length
            TypeError: If buffer types are incompatible
            RuntimeError: If network not initialized
        """
        ...

    def predict(self, inputs: Sequence[float] | memoryview) -> list[float]:
        """
        Make a prediction given input values.

        Supports any object implementing the buffer protocol (lists, tuples,
        array.array, numpy arrays, etc.) containing float64 values.

        Args:
            inputs: Input values (length must match input_size).
                    Can be any buffer-compatible object with float64 dtype.

        Returns:
            List of output values (length matches output_size)

        Raises:
            ValueError: If inputs has wrong length
            TypeError: If buffer type is incompatible
            RuntimeError: If network not initialized
        """
        ...

    def randomize(self) -> None:
        """
        Randomize all network weights.

        Raises:
            RuntimeError: If network not initialized
        """
        ...

    def copy(self) -> GenannNetwork:
        """
        Create a deep copy of the network.

        Returns:
            New GenannNetwork instance with copied weights

        Raises:
            MemoryError: If copy allocation fails
            RuntimeError: If network not initialized
        """
        ...

    def save(self, path: str | bytes | PathLike[str] | PathLike[bytes]) -> None:
        """
        Save the network weights to a file.

        Args:
            path: File path where the network will be saved

        Raises:
            TypeError: If path is not str, bytes, or PathLike
            IOError: If file cannot be opened for writing
            RuntimeError: If network not initialized
        """
        ...

    @classmethod
    def load(cls, path: str | bytes | PathLike[str] | PathLike[bytes]) -> GenannNetwork:
        """
        Load a network from a file.

        Args:
            path: File path from which to load the network

        Returns:
            Loaded GenannNetwork instance

        Raises:
            TypeError: If path is not str, bytes, or PathLike
            IOError: If file cannot be opened for reading
            ValueError: If network cannot be loaded from file
        """
        ...

class FannNetwork:
    """
    A multi-layer neural network using the FANN library.

    This class wraps the FANN (Fast Artificial Neural Network) C library,
    providing a Python interface for creating, training, and using neural
    networks with flexible architectures.
    """

    def __init__(self, layers: Sequence[int] | None = None, connection_rate: float = 1.0) -> None:
        """
        Create a new neural network.

        Args:
            layers: List/tuple of layer sizes [input, hidden1, ..., hiddenN, output].
                    Must have at least 2 layers. If None, creates an uninitialized
                    network (for use with load()).
            connection_rate: Connection density (0.0 to 1.0). 1.0 = fully connected,
                           < 1.0 = sparse network with random connections.

        Raises:
            TypeError: If layers is not a list or tuple
            ValueError: If layers has fewer than 2 elements or any size is <= 0
            MemoryError: If network allocation fails
        """
        ...

    @property
    def input_size(self) -> int:
        """Number of input neurons."""
        ...

    @property
    def output_size(self) -> int:
        """Number of output neurons."""
        ...

    @property
    def total_neurons(self) -> int:
        """Total number of neurons in the network."""
        ...

    @property
    def total_connections(self) -> int:
        """Total number of connections in the network."""
        ...

    @property
    def num_layers(self) -> int:
        """Number of layers in the network."""
        ...

    @property
    def layers(self) -> list[int]:
        """List of neuron counts for each layer."""
        ...

    @property
    def learning_rate(self) -> float:
        """Get or set the learning rate."""
        ...

    @learning_rate.setter
    def learning_rate(self, rate: float) -> None: ...

    @property
    def learning_momentum(self) -> float:
        """Get or set the learning momentum."""
        ...

    @learning_momentum.setter
    def learning_momentum(self, momentum: float) -> None: ...

    def predict(self, inputs: Sequence[float] | memoryview) -> list[float]:
        """
        Make a prediction given input values.

        Supports any object implementing the buffer protocol (lists, tuples,
        array.array, numpy arrays, etc.) containing float32 values.

        Args:
            inputs: Input values (length must match input_size).
                    Can be any buffer-compatible object with float32 dtype.

        Returns:
            List of output values (length matches output_size)

        Raises:
            ValueError: If inputs has wrong length
            TypeError: If buffer type is incompatible
            RuntimeError: If network not initialized
        """
        ...

    def train(
        self,
        inputs: Sequence[float] | memoryview,
        targets: Sequence[float] | memoryview
    ) -> None:
        """
        Train the network on one example using backpropagation.

        Uses the current learning_rate and learning_momentum settings.

        Supports any object implementing the buffer protocol (lists, tuples,
        array.array, numpy arrays, etc.) containing float32 values.

        Args:
            inputs: Input values (length must match input_size).
                    Can be any buffer-compatible object with float32 dtype.
            targets: Target output values (length must match output_size).
                     Can be any buffer-compatible object with float32 dtype.

        Raises:
            ValueError: If inputs or targets have wrong length
            TypeError: If buffer types are incompatible
            RuntimeError: If network not initialized
        """
        ...

    def randomize_weights(self, min_weight: float = -0.1, max_weight: float = 0.1) -> None:
        """
        Randomize all network weights to values in [min_weight, max_weight].

        Args:
            min_weight: Minimum weight value
            max_weight: Maximum weight value

        Raises:
            RuntimeError: If network not initialized
        """
        ...

    def copy(self) -> FannNetwork:
        """
        Create a deep copy of the network.

        Returns:
            New FannNetwork instance with copied weights and structure

        Raises:
            MemoryError: If copy allocation fails
            RuntimeError: If network not initialized
        """
        ...

    def save(self, path: str | bytes | PathLike[str] | PathLike[bytes]) -> None:
        """
        Save the network to a file.

        Args:
            path: File path where the network will be saved

        Raises:
            TypeError: If path is not str, bytes, or PathLike
            IOError: If file cannot be saved
            RuntimeError: If network not initialized
        """
        ...

    @classmethod
    def load(cls, path: str | bytes | PathLike[str] | PathLike[bytes]) -> FannNetwork:
        """
        Load a network from a file.

        Args:
            path: File path from which to load the network

        Returns:
            Loaded FannNetwork instance

        Raises:
            TypeError: If path is not str, bytes, or PathLike
            IOError: If file cannot be opened for reading
        """
        ...

__all__ = ['TinnNetwork', 'GenannNetwork', 'FannNetwork', 'square', 'seed']
