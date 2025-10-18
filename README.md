# cynn

A Cython wrapper for the Tinn neural network library - a minimal, dependency-free neural network implementation in C.

## Overview

cynn provides Python bindings to the [Tinn](https://github.com/glouw/tinn) library, a tiny neural network library written in pure C. The project uses Cython to create efficient Python wrappers around the C implementation, allowing you to train and use neural networks with minimal overhead.

## Features

- Simple 3-layer neural network architecture (input, hidden, output)
- Backpropagation training with configurable learning rate
- Save/load trained models to disk
- Minimal dependencies (no NumPy, TensorFlow, etc.)
- Fast C implementation with Python convenience

## Installation

### Requirements

- Python >= 3.13
- uv (recommended) or pip
- CMake >= 3.15
- C compiler

### Build from source

```bash
# Clone the repository
git clone <repository-url>
cd cynn

# Build and install
make build

# Or manually with uv
uv sync
```

## Usage

### Basic Example

```python
from cynn import TinnNetwork

# Create a network: 2 inputs, 4 hidden neurons, 1 output
net = TinnNetwork(2, 4, 1)

# Make a prediction
inputs = [0.5, 0.3]
output = net.predict(inputs)
print(f"Prediction: {output}")

# Train the network
targets = [0.8]
learning_rate = 0.5
loss = net.train(inputs, targets, learning_rate)
print(f"Loss: {loss}")
```

### XOR Problem

```python
from cynn import TinnNetwork
import random

# XOR training data
xor_data = [
    ([0.0, 0.0], [0.0]),
    ([0.0, 1.0], [1.0]),
    ([1.0, 0.0], [1.0]),
    ([1.0, 1.0], [0.0]),
]

# Create network
net = TinnNetwork(2, 4, 1)

# Train with learning rate annealing
rate = 1.0
anneal = 0.99

for epoch in range(100):
    random.shuffle(xor_data)
    total_error = 0.0

    for inputs, targets in xor_data:
        error = net.train(inputs, targets, rate)
        total_error += error

    rate *= anneal
    avg_error = total_error / len(xor_data)

    if epoch % 20 == 0:
        print(f"Epoch {epoch}: avg error = {avg_error:.6f}")

# Test predictions
for inputs, expected in xor_data:
    pred = net.predict(inputs)
    print(f"{inputs} -> {pred[0]:.4f} (expected {expected[0]})")
```

### Save and Load Models

```python
from cynn import TinnNetwork

# Train a network
net = TinnNetwork(2, 4, 1)
# ... training code ...

# Save to disk
net.save("model.tinn")

# Load from disk
loaded_net = TinnNetwork.load("model.tinn")

# Use loaded network
prediction = loaded_net.predict([0.5, 0.3])
```

## Development

### Building

```bash
# Standard build
make build

# CMake build (for debugging)
make cmake

# Clean build artifacts
make clean
```

### Testing

```bash
# Run all tests
make test

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_basic.py -v
```

### Project Structure

```
cynn/
├── src/
│   └── cynn/
│       ├── __init__.py       # Public API
│       ├── _core.pyx         # Cython implementation
│       ├── _core.pyi         # Type stubs
│       ├── nnet.pxd          # C type declarations
│       └── CMakeLists.txt    # Build configuration
├── thirdparty/
│   └── tinn/                 # Vendored Tinn C library
├── tests/                    # pytest test suite
├── CMakeLists.txt            # Root CMake config
├── Makefile                  # Build shortcuts
└── pyproject.toml            # Python package metadata
```

## API Reference

### TinnNetwork

```python
class TinnNetwork:
    def __init__(self, inputs: int, hidden: int, outputs: int)
```

Create a new neural network.

**Parameters:**
- `inputs`: Number of input neurons
- `hidden`: Number of hidden layer neurons
- `outputs`: Number of output neurons

**Properties:**
- `input_size`: Number of inputs
- `hidden_size`: Number of hidden neurons
- `output_size`: Number of outputs
- `shape`: Tuple of (inputs, hidden, outputs)

**Methods:**

#### predict()
```python
def predict(self, inputs: list[float]) -> list[float]
```
Make a prediction given input values.

#### train()
```python
def train(self, inputs: list[float], targets: list[float], rate: float) -> float
```
Train the network on one example. Returns the error for this training step.

#### save()
```python
def save(self, path: str | bytes | os.PathLike) -> None
```
Save the network weights to a file.

#### load()
```python
@classmethod
def load(cls, path: str | bytes | os.PathLike) -> TinnNetwork
```
Load a network from a file.

## Performance Considerations

- The C implementation is fast but operates on single examples (no batch processing)
- For production machine learning, consider TensorFlow, PyTorch, or JAX
- This library is ideal for:
  - Learning neural network fundamentals
  - Embedded systems with limited resources
  - Simple prediction tasks
  - Environments where large ML frameworks aren't available

## Credits

- [Tinn](https://github.com/glouw/tinn) - Original C neural network library by glouw
- Built with [Cython](https://cython.org/)
- Build system uses [scikit-build-core](https://scikit-build-core.readthedocs.io/)

## License

See LICENSE file for details.
