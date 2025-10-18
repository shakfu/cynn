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

__all__ = ['TinnNetwork', 'square', 'seed']
