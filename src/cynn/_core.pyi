"""Type stubs for cynn._core module."""

from typing import ClassVar, Sequence
from os import PathLike

def square(x: float) -> float:
    """Square a float value."""
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
        inputs: Sequence[float],
        targets: Sequence[float],
        rate: float
    ) -> float:
        """
        Train the network on one example using backpropagation.

        Args:
            inputs: Input values (length must match input_size)
            targets: Target output values (length must match output_size)
            rate: Learning rate (typically 0.0 to 1.0)

        Returns:
            Training error for this example

        Raises:
            ValueError: If inputs or targets have wrong length
        """
        ...

    def predict(self, inputs: Sequence[float]) -> list[float]:
        """
        Make a prediction given input values.

        Args:
            inputs: Input values (length must match input_size)

        Returns:
            List of output values (length matches output_size)

        Raises:
            ValueError: If inputs has wrong length
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

__all__ = ['TinnNetwork', 'square']
