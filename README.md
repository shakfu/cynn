# cynn

A Cython wrapper for minimal, dependency-free neural network libraries in C.

## Overview

cynn provides Python bindings to three lightweight neural network libraries:
- [Tinn](https://github.com/glouw/tinn) - A tiny 3-layer neural network library
- [GENANN](https://github.com/codeplea/genann) - A minimal multi-layer neural network library
- [FANN](https://github.com/libfann/fann) - Fast Artificial Neural Network library

The project uses Cython to create efficient Python wrappers around the C implementations, allowing you to train and use neural networks with minimal overhead.

## Features

- **Three network implementations:**
  - `TinnNetwork`: Simple 3-layer architecture (input, hidden, output) using float32
  - `GenannNetwork`: Flexible multi-layer architecture with arbitrary depth using float64
  - `FannNetwork`: Flexible multi-layer architecture with settable learning parameters using float32
- Backpropagation training with configurable learning rate
- Save/load trained models to disk
- Buffer protocol support - works with lists, tuples, array.array, NumPy arrays, etc.
- **GIL-free execution** - true multithreading support for parallel inference/training
- Fast C implementation with Python convenience
- Zero required dependencies (NumPy is optional)

## Installation

### Requirements

- Python >= 3.13
- uv (recommended) or pip
- CMake >= 3.15
- C compiler

### Build from source

```bash
# Clone the repository
git clone https://github.com/shakfu/cynn
cd cynn

# Build and install
make build

# Or manually with uv
uv sync
```

## Usage

### Basic Example - TinnNetwork

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

### Basic Example - GenannNetwork

```python
from cynn import GenannNetwork

# Create a network: 2 inputs, 2 hidden layers with 4 neurons each, 1 output
net = GenannNetwork(2, 2, 4, 1)

# Make a prediction
inputs = [0.5, 0.3]
output = net.predict(inputs)
print(f"Prediction: {output}")

# Train the network
targets = [0.8]
learning_rate = 0.1
net.train(inputs, targets, learning_rate)

# GenannNetwork has additional features
print(f"Total weights: {net.total_weights}")
print(f"Total neurons: {net.total_neurons}")

# Create a copy of the network
net_copy = net.copy()

# Randomize weights
net.randomize()
```

### Basic Example - FannNetwork

```python
from cynn import FannNetwork

# Create a network: [2 inputs, 4 hidden layer 1, 3 hidden layer 2, 1 output]
net = FannNetwork([2, 4, 3, 1])

# Make a prediction
inputs = [0.5, 0.3]
output = net.predict(inputs)
print(f"Prediction: {output}")

# Adjust learning parameters
net.learning_rate = 0.7
net.learning_momentum = 0.1

# Train the network
targets = [0.8]
net.train(inputs, targets)

# FannNetwork has additional features
print(f"Network layers: {net.layers}")
print(f"Total connections: {net.total_connections}")
print(f"Learning rate: {net.learning_rate}")

# Create a sparse network (50% connectivity)
sparse_net = FannNetwork([2, 8, 1], connection_rate=0.5)

# Create a copy of the network
net_copy = net.copy()

# Randomize weights to specific range
net.randomize_weights(-0.5, 0.5)
```

### XOR Problem

```python
from cynn import TinnNetwork, seed
import random
import time

# Seed random number generators
seed(int(time.time()))
random.seed(int(time.time()))

# XOR training data
xor_data = [
    ([0.0, 0.0], [0.0]),
    ([0.0, 1.0], [1.0]),
    ([1.0, 0.0], [1.0]),
    ([1.0, 1.0], [0.0]),
]

# Create network
net = TinnNetwork(2, 4, 1)

# Train with constant learning rate
rate = 0.5

for epoch in range(3000):
    random.shuffle(xor_data)
    total_error = 0.0

    for inputs, targets in xor_data:
        error = net.train(inputs, targets, rate)
        total_error += error

    avg_error = total_error / len(xor_data)

    if epoch % 500 == 0:
        print(f"Epoch {epoch}: avg error = {avg_error:.6f}")

# Test predictions
for inputs, expected in xor_data:
    pred = net.predict(inputs)
    result = "✓" if abs(pred[0] - expected[0]) < 0.3 else "✗"
    print(f"{result} {inputs} -> {pred[0]:.4f} (expected {expected[0]})")
```

Example output:

```text
% uv run python tests/examples/xor_problem.py
Epoch 0: avg error = 0.129680
Epoch 500: avg error = 0.127645
Epoch 1000: avg error = 0.123747
Epoch 1500: avg error = 0.029109
Epoch 2000: avg error = 0.008168
Epoch 2500: avg error = 0.004388
✓ [0.0, 0.0] -> 0.0893 (expected 0.0)
✓ [1.0, 0.0] -> 0.9285 (expected 1.0)
✓ [0.0, 1.0] -> 0.9284 (expected 1.0)
✓ [1.0, 1.0] -> 0.0721 (expected 0.0)
```


### NumPy Support

The library supports any object implementing the buffer protocol, including NumPy arrays:

```python
import numpy as np
from cynn import TinnNetwork

# Create network
net = TinnNetwork(2, 4, 1)

# Use NumPy arrays (float32 recommended, but float64 works too)
inputs = np.array([0.5, 0.3], dtype=np.float32)
targets = np.array([0.8], dtype=np.float32)

# Train with numpy arrays
loss = net.train(inputs, targets, 0.1)

# Predict with numpy arrays
prediction = net.predict(inputs)

# Batch processing
batch = np.array([
    [0.1, 0.2],
    [0.3, 0.4],
    [0.5, 0.6],
], dtype=np.float32)

predictions = [net.predict(row) for row in batch]
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
│   ├── tinn/                 # Vendored Tinn C library
│   ├── genann/               # Vendored GENANN C library
│   └── fann/                 # Vendored FANN C library
├── tests/                    # pytest test suite
├── CMakeLists.txt            # Root CMake config
├── Makefile                  # Build shortcuts
└── pyproject.toml            # Python package metadata
```

## API Reference

### seed()

```python
def seed(seed_value: int = 0) -> None
```

Seed the C random number generator used for weight initialization. If seed_value is 0 (default), uses current time. Call this before creating networks for reproducible results.

### TinnNetwork

```python
class TinnNetwork:
    def __init__(self, inputs: int, hidden: int, outputs: int)
```

Create a new 3-layer neural network (float32 precision).

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

### GenannNetwork

```python
class GenannNetwork:
    def __init__(self, inputs: int, hidden_layers: int, hidden: int, outputs: int)
```

Create a new multi-layer neural network (float64 precision).

**Parameters:**
- `inputs`: Number of input neurons
- `hidden_layers`: Number of hidden layers
- `hidden`: Number of neurons per hidden layer
- `outputs`: Number of output neurons

**Properties:**
- `input_size`: Number of inputs
- `hidden_layers`: Number of hidden layers
- `hidden_size`: Number of neurons per hidden layer
- `output_size`: Number of outputs
- `shape`: Tuple of (inputs, hidden_layers, hidden, outputs)
- `total_weights`: Total number of weights in the network
- `total_neurons`: Total number of neurons plus inputs

**Methods:**

#### predict()
```python
def predict(self, inputs: list[float]) -> list[float]
```
Make a prediction given input values.

#### train()
```python
def train(self, inputs: list[float], targets: list[float], rate: float) -> None
```
Train the network on one example using backpropagation.

#### randomize()
```python
def randomize(self) -> None
```
Randomize all network weights.

#### copy()
```python
def copy(self) -> GenannNetwork
```
Create a deep copy of the network.

#### save()
```python
def save(self, path: str | bytes | os.PathLike) -> None
```
Save the network weights to a file.

#### load()
```python
@classmethod
def load(cls, path: str | bytes | os.PathLike) -> GenannNetwork
```
Load a network from a file.

### FannNetwork

```python
class FannNetwork:
    def __init__(self, layers: list[int] | None = None, connection_rate: float = 1.0)
```

Create a new multi-layer neural network (float32 precision) using the FANN library.

**Parameters:**
- `layers`: List of layer sizes [input, hidden1, ..., hiddenN, output]. Must have at least 2 layers.
- `connection_rate`: Connection density (0.0 to 1.0). 1.0 = fully connected, < 1.0 = sparse network.

**Properties:**
- `input_size`: Number of inputs
- `output_size`: Number of outputs
- `total_neurons`: Total number of neurons
- `total_connections`: Total number of connections
- `num_layers`: Number of layers
- `layers`: List of neuron counts for each layer
- `learning_rate`: Get or set the learning rate
- `learning_momentum`: Get or set the learning momentum

**Methods:**

#### predict()
```python
def predict(self, inputs: list[float]) -> list[float]
```
Make a prediction given input values.

#### train()
```python
def train(self, inputs: list[float], targets: list[float]) -> None
```
Train the network on one example using backpropagation. Uses current `learning_rate` and `learning_momentum`.

#### randomize_weights()
```python
def randomize_weights(self, min_weight: float = -0.1, max_weight: float = 0.1) -> None
```
Randomize all network weights to values in [min_weight, max_weight].

#### copy()
```python
def copy(self) -> FannNetwork
```
Create a deep copy of the network.

#### save()
```python
def save(self, path: str | bytes | os.PathLike) -> None
```
Save the network to a file (FANN text format).

#### load()
```python
@classmethod
def load(cls, path: str | bytes | os.PathLike) -> FannNetwork
```
Load a network from a file.

## Choosing Between Network Implementations

**Use TinnNetwork when:**
- You need a simple 3-layer network
- Memory efficiency is important (float32 uses less memory)
- You want the training method to return loss values
- You prefer a simpler API with fixed architecture

**Use GenannNetwork when:**
- You need multiple hidden layers (deep networks)
- Higher precision is required (float64)
- You need to copy networks
- You want to randomize weights after creation
- You need to query total weights/neurons
- You prefer the constructor pattern: `GenannNetwork(inputs, hidden_layers, hidden_size, outputs)`

**Use FannNetwork when:**
- You need flexible multi-layer architectures
- You want to control learning rate and momentum during training
- You need sparse networks (partial connectivity)
- You prefer list-based layer specification: `FannNetwork([2, 4, 3, 1])`
- You want settable learning parameters
- Memory efficiency is important (float32 uses less memory than float64)

## Performance Considerations

- The C implementation is fast but operates on single examples (no batch processing)
- **GIL-free execution**: All computational operations (`train`, `predict`, network creation) release the Python GIL, enabling true parallel execution across multiple threads
- Thread-safe: Multiple threads can safely share the same network for predictions
- For production machine learning, consider TensorFlow, PyTorch, or JAX
- This library is ideal for:
  - Learning neural network fundamentals
  - Embedded systems with limited resources
  - Simple prediction tasks
  - Parallel inference workloads
  - Environments where large ML frameworks aren't available

### Multithreading Example

```python
from concurrent.futures import ThreadPoolExecutor
from cynn import TinnNetwork
import numpy as np

# Create a shared network
net = TinnNetwork(100, 50, 10)

def process_batch(batch_data):
    """Process a batch of inputs in parallel."""
    results = []
    for inputs in batch_data:
        pred = net.predict(inputs)
        results.append(pred)
    return results

# Prepare data batches
data = [np.random.rand(100).astype(np.float32) for _ in range(1000)]
batches = [data[i:i+250] for i in range(0, 1000, 250)]

# Process batches in parallel (GIL-free!)
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_batch, batches))
```

## Credits

- [Tinn](https://github.com/glouw/tinn) - Original C neural network library by glouw
- [GENANN](https://github.com/codeplea/genann) - Minimal C neural network library by codeplea
- [FANN](https://github.com/libfann/fann) - Fast Artificial Neural Network library by Steffen Nissen
- Built with [Cython](https://cython.org/)
- Build system uses [scikit-build-core](https://scikit-build-core.readthedocs.io/)

## License

See LICENSE file for details.
