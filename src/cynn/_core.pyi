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

    def evaluate(
        self,
        inputs: Sequence[float] | memoryview,
        targets: Sequence[float] | memoryview
    ) -> float:
        """
        Compute loss without training.

        Args:
            inputs: Input values (length must match input_size)
            targets: Target output values (length must match output_size)

        Returns:
            Mean squared error between prediction and targets

        Raises:
            ValueError: If inputs or targets have wrong length
        """
        ...

    def train_batch(
        self,
        inputs_list: list,
        targets_list: list,
        rate: float,
        shuffle: bool = False
    ) -> dict[str, float]:
        """
        Train on multiple examples in batch.

        Args:
            inputs_list: List of input arrays
            targets_list: List of target arrays
            rate: Learning rate
            shuffle: Whether to shuffle the batch before training

        Returns:
            dict with keys: 'mean_loss', 'total_loss', 'count'

        Raises:
            ValueError: If inputs_list and targets_list have different lengths
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

    def __enter__(self) -> TinnNetwork:
        """Enter context manager, returning self."""
        ...

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit context manager. Cleanup handled by __dealloc__."""
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
    ) -> float:
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

        Returns:
            Mean squared error for this training example

        Raises:
            ValueError: If inputs or targets have wrong length
            TypeError: If buffer types are incompatible
            RuntimeError: If network not initialized
        """
        ...

    def evaluate(
        self,
        inputs: Sequence[float] | memoryview,
        targets: Sequence[float] | memoryview
    ) -> float:
        """
        Compute loss without training.

        Args:
            inputs: Input values (length must match input_size)
            targets: Target output values (length must match output_size)

        Returns:
            Mean squared error between prediction and targets

        Raises:
            ValueError: If inputs or targets have wrong length
            RuntimeError: If network not initialized
        """
        ...

    def train_batch(
        self,
        inputs_list: list,
        targets_list: list,
        rate: float,
        shuffle: bool = False
    ) -> dict[str, float]:
        """
        Train on multiple examples in batch.

        Args:
            inputs_list: List of input arrays
            targets_list: List of target arrays
            rate: Learning rate
            shuffle: Whether to shuffle the batch before training

        Returns:
            dict with keys: 'mean_loss', 'total_loss', 'count'

        Raises:
            ValueError: If inputs_list and targets_list have different lengths
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

    def __enter__(self) -> GenannNetwork:
        """Enter context manager, returning self."""
        ...

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit context manager. Cleanup handled by __dealloc__."""
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
    ) -> float:
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

        Returns:
            Mean squared error for this training example

        Raises:
            ValueError: If inputs or targets have wrong length
            TypeError: If buffer types are incompatible
            RuntimeError: If network not initialized
        """
        ...

    def evaluate(
        self,
        inputs: Sequence[float] | memoryview,
        targets: Sequence[float] | memoryview
    ) -> float:
        """
        Compute loss without training.

        Args:
            inputs: Input values (length must match input_size)
            targets: Target output values (length must match output_size)

        Returns:
            Mean squared error between prediction and targets

        Raises:
            ValueError: If inputs or targets have wrong length
            RuntimeError: If network not initialized
        """
        ...

    def train_batch(
        self,
        inputs_list: list,
        targets_list: list,
        shuffle: bool = False
    ) -> dict[str, float]:
        """
        Train on multiple examples in batch.

        Uses the current learning_rate and learning_momentum settings.

        Args:
            inputs_list: List of input arrays
            targets_list: List of target arrays
            shuffle: Whether to shuffle the batch before training

        Returns:
            dict with keys: 'mean_loss', 'total_loss', 'count'

        Raises:
            ValueError: If inputs_list and targets_list have different lengths
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

    def __enter__(self) -> FannNetwork:
        """Enter context manager, returning self."""
        ...

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit context manager. Cleanup handled by __dealloc__."""
        ...

class CNNLayer:
    """
    Represents a single layer in a convolutional neural network.

    This class wraps the C Layer structure and should not be instantiated directly.
    Use CNNNetwork methods to build layers.
    """

    @property
    def layer_id(self) -> int:
        """Layer ID in the network."""
        ...

    @property
    def shape(self) -> tuple[int, int, int]:
        """Layer shape as (depth, width, height)."""
        ...

    @property
    def depth(self) -> int:
        """Layer depth dimension."""
        ...

    @property
    def width(self) -> int:
        """Layer width dimension."""
        ...

    @property
    def height(self) -> int:
        """Layer height dimension."""
        ...

    @property
    def num_nodes(self) -> int:
        """Total number of nodes (depth * width * height)."""
        ...

    @property
    def num_weights(self) -> int:
        """Number of weights in this layer."""
        ...

    @property
    def num_biases(self) -> int:
        """Number of biases in this layer."""
        ...

    @property
    def layer_type(self) -> str:
        """Layer type as string: 'input', 'full', or 'conv'."""
        ...

    @property
    def kernel_size(self) -> int:
        """
        Kernel size for convolutional layers.

        Raises:
            ValueError: If this is not a convolutional layer
        """
        ...

    @property
    def padding(self) -> int:
        """
        Padding for convolutional layers.

        Raises:
            ValueError: If this is not a convolutional layer
        """
        ...

    @property
    def stride(self) -> int:
        """
        Stride for convolutional layers.

        Raises:
            ValueError: If this is not a convolutional layer
        """
        ...

    def get_outputs(self) -> list[float]:
        """
        Get the output values of this layer.

        Returns:
            List of output values

        Raises:
            RuntimeError: If layer not initialized
        """
        ...


class CNNNetwork:
    """
    A convolutional neural network with support for input, convolutional, and fully-connected layers.

    This class wraps the nn1 CNN C library. Networks are built by chaining layers together
    starting from an input layer.

    Example:
        net = CNNNetwork()
        net.create_input_layer(1, 28, 28)  # MNIST-like input
        net.add_conv_layer(8, 24, 24, kernel_size=5, stride=1)
        net.add_conv_layer(16, 12, 12, kernel_size=5, stride=2)
        net.add_full_layer(10)  # 10 output classes

        # Train
        error = net.train(input_data, target_data, learning_rate=0.01)

        # Predict
        outputs = net.predict(input_data)
    """

    def create_input_layer(self, depth: int, width: int, height: int) -> CNNLayer:
        """
        Create an input layer. This must be called first to start building a network.

        Args:
            depth: Input depth (number of channels, e.g., 1 for grayscale, 3 for RGB)
            width: Input width in pixels
            height: Input height in pixels

        Returns:
            CNNLayer wrapper for the created input layer

        Raises:
            ValueError: If network already has an input layer or dimensions invalid
            MemoryError: If layer creation fails
        """
        ...

    def add_conv_layer(
        self,
        depth: int,
        width: int,
        height: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
        std: float = 0.1
    ) -> CNNLayer:
        """
        Add a convolutional layer to the network.

        Args:
            depth: Number of output channels (filters/feature maps)
            width: Output width after convolution
            height: Output height after convolution
            kernel_size: Size of the convolution kernel (must be odd, e.g., 3, 5, 7)
            padding: Zero-padding added to input (default: 0)
            stride: Stride for the convolution operation (default: 1)
            std: Standard deviation for random weight initialization (default: 0.1)

        Returns:
            CNNLayer wrapper for the created convolutional layer

        Raises:
            ValueError: If network has no input layer or parameters are invalid
            MemoryError: If layer creation fails

        Note:
            The relationship between input/output dimensions must satisfy:
            (width-1) * stride + kernel_size <= prev_width + padding*2
            (height-1) * stride + kernel_size <= prev_height + padding*2
        """
        ...

    def add_full_layer(self, num_nodes: int, std: float = 0.1) -> CNNLayer:
        """
        Add a fully-connected layer to the network.

        Args:
            num_nodes: Number of nodes in this layer
            std: Standard deviation for random weight initialization (default: 0.1)

        Returns:
            CNNLayer wrapper for the created fully-connected layer

        Raises:
            ValueError: If network has no input layer or num_nodes invalid
            MemoryError: If layer creation fails
        """
        ...

    @property
    def input_shape(self) -> tuple[int, int, int]:
        """
        Shape of the input layer as (depth, width, height).

        Raises:
            RuntimeError: If network has no input layer
        """
        ...

    @property
    def output_size(self) -> int:
        """
        Number of output nodes in the final layer.

        Raises:
            RuntimeError: If network has no layers
        """
        ...

    @property
    def num_layers(self) -> int:
        """Total number of layers in the network."""
        ...

    @property
    def layers(self) -> list[CNNLayer]:
        """List of all layer wrappers in the network."""
        ...

    def predict(self, inputs: Sequence[float] | memoryview) -> list[float]:
        """
        Make a prediction given input values.

        The network performs a forward pass through all layers and returns
        the outputs from the final layer.

        Args:
            inputs: Input values as a flat array (length must match input layer size).
                   Size should be depth * width * height of the input layer.
                   Can be any buffer-compatible object with float64 dtype.

        Returns:
            List of output values from the final layer

        Raises:
            RuntimeError: If network has no layers
            ValueError: If input size doesn't match the network's input layer
        """
        ...

    def train(
        self,
        inputs: Sequence[float] | memoryview,
        targets: Sequence[float] | memoryview,
        learning_rate: float
    ) -> float:
        """
        Train the network on one example using backpropagation.

        Performs forward pass, computes error, backpropagates gradients,
        and updates all weights and biases in the network.

        Args:
            inputs: Input values as a flat array (length must match input layer size).
                   Can be any buffer-compatible object with float64 dtype.
            targets: Target output values (length must match output layer size).
                    Can be any buffer-compatible object with float64 dtype.
            learning_rate: Learning rate for gradient descent (typically 0.001 to 0.1)

        Returns:
            Mean squared error for this training example

        Raises:
            RuntimeError: If network has no layers
            ValueError: If input or target size doesn't match the network dimensions
        """
        ...

    def evaluate(
        self,
        inputs: Sequence[float] | memoryview,
        targets: Sequence[float] | memoryview
    ) -> float:
        """
        Compute loss without training.

        Args:
            inputs: Input values (length must match input layer size)
            targets: Target output values (length must match output layer size)

        Returns:
            Mean squared error between prediction and targets

        Raises:
            ValueError: If inputs or targets have wrong length
            RuntimeError: If network has no layers
        """
        ...

    def train_batch(
        self,
        inputs_list: list,
        targets_list: list,
        learning_rate: float,
        shuffle: bool = False
    ) -> dict[str, float]:
        """
        Train on multiple examples in batch.

        Args:
            inputs_list: List of input arrays
            targets_list: List of target arrays
            learning_rate: Learning rate for weight updates
            shuffle: Whether to shuffle the batch before training

        Returns:
            dict with keys: 'mean_loss', 'total_loss', 'count'

        Raises:
            ValueError: If inputs_list and targets_list have different lengths
            RuntimeError: If network has no layers
        """
        ...

    def dump(self) -> None:
        """
        Print debug information about all layers to stdout.

        This displays the layer structure, weights, biases, and current outputs.
        Useful for debugging and understanding network behavior.

        Raises:
            RuntimeError: If network has no layers
        """
        ...

    def __enter__(self) -> CNNNetwork:
        """Enter context manager, returning self."""
        ...

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit context manager. Cleanup handled by __dealloc__."""
        ...


__all__ = ['TinnNetwork', 'GenannNetwork', 'FannNetwork', 'FannNetworkDouble', 'CNNNetwork', 'CNNLayer', 'seed']


class FannNetworkDouble:
    """
    A multi-layer neural network using the FANN library with float64 precision.
    This class wraps the FANN (Fast Artificial Neural Network) C library's
    double precision implementation, providing a Python interface for creating,
    training, and using neural networks with flexible architectures and higher
    numerical precision than FannNetwork (float32).
    """
    def __init__(self, layers: Sequence[int] | None = None, connection_rate: float = 1.0) -> None:
        """
        Create a new neural network with float64 precision.
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
        """List of layer sizes."""
        ...
    @property
    def learning_rate(self) -> float:
        """Learning rate for training."""
        ...
    @learning_rate.setter
    def learning_rate(self, rate: float) -> None:
        ...
    @property
    def learning_momentum(self) -> float:
        """Learning momentum for training."""
        ...
    @learning_momentum.setter
    def learning_momentum(self, momentum: float) -> None:
        ...

    def predict(self, inputs: Sequence[float] | SupportsBufferProtocol) -> list[float]:
        """
        Make a prediction using the network.

        Args:
            inputs: Input values (list, tuple, array, NumPy array, etc.)

        Returns:
            List of output values

        Raises:
            ValueError: If input size doesn't match network's input layer
            RuntimeError: If network not initialized
        """
        ...

    def train(self, inputs: Sequence[float] | SupportsBufferProtocol, targets: Sequence[float] | SupportsBufferProtocol) -> float:
        """
        Train the network on one example.

        Uses the current learning_rate and learning_momentum settings.

        Args:
            inputs: Input values (list, tuple, array, NumPy array, etc.)
            targets: Target output values

        Returns:
            Mean squared error for this training example

        Raises:
            ValueError: If input/target size doesn't match network dimensions
            RuntimeError: If network not initialized
        """
        ...

    def evaluate(
        self,
        inputs: Sequence[float] | SupportsBufferProtocol,
        targets: Sequence[float] | SupportsBufferProtocol
    ) -> float:
        """
        Compute loss without training.

        Args:
            inputs: Input values
            targets: Target output values

        Returns:
            Mean squared error between prediction and targets

        Raises:
            ValueError: If inputs or targets have wrong length
            RuntimeError: If network not initialized
        """
        ...

    def train_batch(
        self,
        inputs_list: list,
        targets_list: list,
        shuffle: bool = False
    ) -> dict[str, float]:
        """
        Train on multiple examples in batch.

        Uses the current learning_rate and learning_momentum settings.

        Args:
            inputs_list: List of input arrays
            targets_list: List of target arrays
            shuffle: Whether to shuffle the batch before training

        Returns:
            dict with keys: 'mean_loss', 'total_loss', 'count'

        Raises:
            ValueError: If inputs_list and targets_list have different lengths
            RuntimeError: If network not initialized
        """
        ...

    def randomize_weights(self, min_weight: float = -0.1, max_weight: float = 0.1) -> None:
        """
        Randomize all network weights.

        Args:
            min_weight: Minimum weight value
            max_weight: Maximum weight value

        Raises:
            RuntimeError: If network not initialized
        """
        ...

    def copy(self) -> FannNetworkDouble:
        """
        Create a deep copy of the network.

        Returns:
            New FannNetworkDouble instance with copied weights and structure

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
    def load(cls, path: str | bytes | PathLike[str] | PathLike[bytes]) -> FannNetworkDouble:
        """
        Load a network from a file.

        Args:
            path: File path from which to load the network

        Returns:
            Loaded FannNetworkDouble instance

        Raises:
            TypeError: If path is not str, bytes, or PathLike
            IOError: If file cannot be opened for reading
        """
        ...

    def __enter__(self) -> FannNetworkDouble:
        """Enter context manager, returning self."""
        ...

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit context manager. Cleanup handled by __dealloc__."""
        ...
