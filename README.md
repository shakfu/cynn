# cynn

A Cython wrapper for minimal, dependency-free neural network libraries in C.

## Overview

cynn provides Python bindings to four lightweight neural network libraries:
- [Tinn](https://github.com/glouw/tinn) - A tiny 3-layer neural network library
- [GENANN](https://github.com/codeplea/genann) - A minimal multi-layer neural network library
- [FANN](https://github.com/libfann/fann) - Fast Artificial Neural Network library
- [nn1](https://github.com/euske/nn1) - Convolutional Neural Network in C

The project uses Cython to create efficient Python wrappers around the C implementations, allowing you to train and use neural networks with minimal overhead.

## Features

- **Five network implementations:**
  - `TinnNetwork`: Simple 3-layer architecture (input, hidden, output) using float32
  - `GenannNetwork`: Flexible multi-layer architecture with arbitrary depth using float64
  - `FannNetwork`: Flexible multi-layer architecture with settable learning parameters using float32
  - `FannNetworkDouble`: Same as FannNetwork but with float64 precision
  - `CNNNetwork`: Layer-based convolutional neural network with input, conv, and fully-connected layers using float64
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
loss = net.train(inputs, targets, learning_rate)
print(f"Loss: {loss}")

# GenannNetwork has additional features
print(f"Total weights: {net.total_weights}")
print(f"Total neurons: {net.total_neurons}")

# Create a copy of the network
net_copy = net.copy()

# Randomize weights
net.randomize()
```

### Basic Example - CNNNetwork

```python
from cynn import CNNNetwork

# Create a convolutional neural network
net = CNNNetwork()
net.create_input_layer(1, 28, 28)  # 28x28 grayscale image input
net.add_conv_layer(8, 24, 24, kernel_size=5, stride=1)  # 8 filters, 5x5 kernel
net.add_conv_layer(16, 12, 12, kernel_size=5, stride=2)  # 16 filters, stride 2
net.add_full_layer(10)  # 10 output classes

# Prepare input (flattened 28x28 image)
inputs = [0.5] * (28 * 28)  # 784 values
targets = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # one-hot encoded

# Train the network
error = net.train(inputs, targets, learning_rate=0.01)
print(f"Training error: {error}")

# Make predictions
outputs = net.predict(inputs)
predicted_class = outputs.index(max(outputs))
print(f"Predicted class: {predicted_class}")

# CNNNetwork has additional features
print(f"Network layers: {net.num_layers}")
print(f"Input shape: {net.input_shape}")
print(f"Output size: {net.output_size}")

# Access individual layers
for layer in net.layers:
    print(f"Layer {layer.layer_id}: type={layer.layer_type}, shape={layer.shape}")
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
loss = net.train(inputs, targets)
print(f"Loss: {loss}")

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

### Batch Training

All network types support batch training for improved efficiency:

```python
from cynn import GenannNetwork

net = GenannNetwork(2, 1, 4, 1)

# Prepare batch data (XOR problem)
inputs_list = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
]
targets_list = [
    [0.0],
    [1.0],
    [1.0],
    [0.0]
]

# Train on entire batch with optional shuffling
stats = net.train_batch(inputs_list, targets_list, rate=0.1, shuffle=True)

print(f"Batch mean loss: {stats['mean_loss']}")
print(f"Batch total loss: {stats['total_loss']}")
print(f"Examples trained: {stats['count']}")

# Train for multiple epochs
for epoch in range(100):
    stats = net.train_batch(inputs_list, targets_list, rate=0.1, shuffle=True)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: loss = {stats['mean_loss']:.4f}")
```

### Evaluating Without Training

Use `evaluate()` to compute loss without updating weights (useful for validation):

```python
from cynn import TinnNetwork

net = TinnNetwork(2, 4, 1)

# Training data
train_inputs = [0.5, 0.3]
train_targets = [0.8]

# Validation data
val_inputs = [0.4, 0.6]
val_targets = [0.7]

# Train on training data
train_loss = net.train(train_inputs, train_targets, rate=0.5)
print(f"Training loss: {train_loss}")

# Evaluate on validation data (no weight updates)
val_loss = net.evaluate(val_inputs, val_targets)
print(f"Validation loss: {val_loss}")

# Verify evaluate doesn't change weights
val_loss2 = net.evaluate(val_inputs, val_targets)
assert val_loss == val_loss2  # Should be identical
```

### Training with Validation

Combine batch training with evaluation for train/validation splits:

```python
from cynn import FannNetwork

net = FannNetwork([2, 8, 1])
net.learning_rate = 0.5

# Split data into train/validation
train_inputs = [[0.0, 0.0], [0.0, 1.0]]
train_targets = [[0.0], [1.0]]

val_inputs = [[1.0, 0.0], [1.0, 1.0]]
val_targets = [[1.0], [0.0]]

for epoch in range(50):
    # Train on training set
    train_stats = net.train_batch(train_inputs, train_targets, shuffle=True)

    # Evaluate on validation set (no weight updates)
    val_losses = [net.evaluate(inp, tgt) for inp, tgt in zip(val_inputs, val_targets)]
    val_loss = sum(val_losses) / len(val_losses)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: train_loss={train_stats['mean_loss']:.4f}, val_loss={val_loss:.4f}")
```

### Context Manager Support

All network types support Python's context manager protocol (`with` statement) for cleaner code:

```python
from cynn import TinnNetwork, GenannNetwork, CNNNetwork

# Automatic resource management with context manager
with TinnNetwork(2, 4, 1) as net:
    output = net.predict([0.5, 0.3])
    loss = net.train([0.5, 0.3], [0.8], rate=0.5)
    print(f"Loss: {loss}")

# Works with all network types
with GenannNetwork(2, 1, 4, 1) as net:
    loss = net.train([0.5, 0.3], [0.8], rate=0.1)

# Useful for temporary networks
with CNNNetwork() as net:
    net.create_input_layer(1, 4, 4)
    net.add_full_layer(2)
    result = net.predict([0.5] * 16)

# Networks remain usable after exiting context
net = TinnNetwork(2, 4, 1)
with net as network:
    network.train([0.5, 0.3], [0.8], rate=0.5)
# Network 'net' is still valid and usable here
output = net.predict([0.5, 0.3])
```

Note: The context manager protocol ensures clean resource handling, but cynn networks already handle cleanup automatically via `__dealloc__`, so using `with` is optional and primarily for code clarity.

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
│       ├── cnn.pxd           # CNN declarations
│       ├── dfann.pxd         # Double Fann declarations
│       ├── ffann.pxd         # Float Fann declarations
│       ├── genann.pxd        # Genann declarations
│       ├── tinn.pxd          # Tinn declarations
│       └── CMakeLists.txt    # Build configuration
├── thirdparty/
│   ├── tinn/                 # Vendored Tinn C library
│   ├── genann/               # Vendored Genann C library
│   ├── fann/                 # Vendored Fann C library
│   └── nn1/                  # Vendored nn1 CNN C library
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
Train the network on one example. Returns the mean squared error for this training step.

#### evaluate()
```python
def evaluate(self, inputs: list[float], targets: list[float]) -> float
```
Compute loss without training. Returns mean squared error between prediction and targets.

#### train_batch()
```python
def train_batch(
    self,
    inputs_list: list,
    targets_list: list,
    rate: float,
    shuffle: bool = False
) -> dict[str, float]
```
Train on multiple examples in batch. Returns dict with keys: `'mean_loss'`, `'total_loss'`, `'count'`.

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

#### \_\_enter\_\_() / \_\_exit\_\_()
```python
def __enter__(self) -> TinnNetwork
def __exit__(self, exc_type, exc_val, exc_tb) -> bool
```
Context manager protocol support. Enables use of `with` statement for cleaner code. The network handles cleanup automatically via `__dealloc__`, so context manager usage is optional.

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
def train(self, inputs: list[float], targets: list[float], rate: float) -> float
```
Train the network on one example using backpropagation. Returns mean squared error.

#### evaluate()
```python
def evaluate(self, inputs: list[float], targets: list[float]) -> float
```
Compute loss without training. Returns mean squared error between prediction and targets.

#### train_batch()
```python
def train_batch(
    self,
    inputs_list: list,
    targets_list: list,
    rate: float,
    shuffle: bool = False
) -> dict[str, float]
```
Train on multiple examples in batch. Returns dict with keys: `'mean_loss'`, `'total_loss'`, `'count'`.

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

#### \_\_enter\_\_() / \_\_exit\_\_()
```python
def __enter__(self) -> GenannNetwork
def __exit__(self, exc_type, exc_val, exc_tb) -> bool
```
Context manager protocol support. Enables use of `with` statement for cleaner code. The network handles cleanup automatically via `__dealloc__`, so context manager usage is optional.

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
def train(self, inputs: list[float], targets: list[float]) -> float
```
Train the network on one example using backpropagation. Uses current `learning_rate` and `learning_momentum`. Returns mean squared error.

#### evaluate()
```python
def evaluate(self, inputs: list[float], targets: list[float]) -> float
```
Compute loss without training. Returns mean squared error between prediction and targets.

#### train_batch()
```python
def train_batch(
    self,
    inputs_list: list,
    targets_list: list,
    shuffle: bool = False
) -> dict[str, float]
```
Train on multiple examples in batch. Uses current `learning_rate` and `learning_momentum`. Returns dict with keys: `'mean_loss'`, `'total_loss'`, `'count'`.

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

#### \_\_enter\_\_() / \_\_exit\_\_()
```python
def __enter__(self) -> FannNetwork
def __exit__(self, exc_type, exc_val, exc_tb) -> bool
```
Context manager protocol support. Enables use of `with` statement for cleaner code. The network handles cleanup automatically via `__dealloc__`, so context manager usage is optional.

### FannNetworkDouble

```python
class FannNetworkDouble:
    def __init__(self, layers: list[int] | None = None, connection_rate: float = 1.0)
```

Create a new multi-layer neural network (float64 precision) using the FANN library. FannNetworkDouble has the identical API to FannNetwork but uses double precision for better numerical stability.

**Parameters:**
- `layers`: List of layer sizes [input, hidden1, ..., hiddenN, output]. Must have at least 2 layers.
- `connection_rate`: Connection density (0.0 to 1.0). 1.0 = fully connected, < 1.0 = sparse network.

**Properties:**
Same as FannNetwork: `input_size`, `output_size`, `total_neurons`, `total_connections`, `num_layers`, `layers`, `learning_rate`, `learning_momentum`

**Methods:**
Same as FannNetwork: `predict()`, `train()`, `evaluate()`, `train_batch()`, `randomize_weights()`, `copy()`, `save()`, `load()`, `__enter__()`, `__exit__()`

**Example:**
```python
from cynn import FannNetworkDouble
import numpy as np

# Create network with float64 precision
net = FannNetworkDouble([2, 4, 3, 1])

# Works seamlessly with NumPy's default float64
inputs = np.array([0.5, 0.3])  # dtype=float64 by default
targets = np.array([0.8])

# Train and predict with higher precision
net.train(inputs, targets)
prediction = net.predict(inputs)
```

### CNNNetwork

```python
class CNNNetwork:
    def __init__(self)
```

Create a new convolutional neural network (float64 precision). Networks are built by adding layers sequentially.

**Properties:**
- `input_shape`: Tuple of (depth, width, height) for the input layer
- `output_size`: Number of output nodes in the final layer
- `num_layers`: Total number of layers in the network
- `layers`: List of CNNLayer wrappers

**Methods:**

#### create_input_layer()
```python
def create_input_layer(self, depth: int, width: int, height: int) -> CNNLayer
```
Create an input layer. Must be called first when building a network.

#### add_conv_layer()
```python
def add_conv_layer(
    self,
    depth: int,
    width: int,
    height: int,
    kernel_size: int,
    padding: int = 0,
    stride: int = 1,
    std: float = 0.1
) -> CNNLayer
```
Add a convolutional layer with specified output dimensions and convolution parameters.

#### add_full_layer()
```python
def add_full_layer(self, num_nodes: int, std: float = 0.1) -> CNNLayer
```
Add a fully-connected layer.

#### predict()
```python
def predict(self, inputs: list[float]) -> list[float]
```
Make a prediction. Input should be a flat array of size depth × width × height.

#### train()
```python
def train(
    self,
    inputs: list[float],
    targets: list[float],
    learning_rate: float
) -> float
```
Train the network on one example. Returns mean squared error.

#### evaluate()
```python
def evaluate(self, inputs: list[float], targets: list[float]) -> float
```
Compute loss without training. Returns mean squared error between prediction and targets.

#### train_batch()
```python
def train_batch(
    self,
    inputs_list: list,
    targets_list: list,
    learning_rate: float,
    shuffle: bool = False
) -> dict[str, float]
```
Train on multiple examples in batch. Returns dict with keys: `'mean_loss'`, `'total_loss'`, `'count'`.

#### dump()
```python
def dump(self) -> None
```
Print debug information about all layers to stdout.

#### \_\_enter\_\_() / \_\_exit\_\_()
```python
def __enter__(self) -> CNNNetwork
def __exit__(self, exc_type, exc_val, exc_tb) -> bool
```
Context manager protocol support. Enables use of `with` statement for cleaner code. The network handles cleanup automatically via `__dealloc__`, so context manager usage is optional.

### CNNLayer

```python
class CNNLayer
```

Represents a single layer in a CNN. Created by CNNNetwork methods, not directly instantiated.

**Properties:**
- `layer_id`: Layer ID in the network
- `shape`: Tuple of (depth, width, height)
- `depth`, `width`, `height`: Individual dimensions
- `num_nodes`: Total nodes (depth × width × height)
- `num_weights`, `num_biases`: Weight and bias counts
- `layer_type`: String ('input', 'conv', or 'full')
- `kernel_size`, `padding`, `stride`: Conv layer parameters (raises ValueError for non-conv layers)

**Methods:**

#### get_outputs()
```python
def get_outputs(self) -> list[float]
```
Get the output values of this layer.

## Choosing Between Network Implementations

| Feature | TinnNetwork | GenannNetwork | FannNetwork | FannNetworkDouble | CNNNetwork |
|---------|-------------|---------------|-------------|-------------------|------------|
| **Precision** | float32 | float64 | float32 | float64 | float64 |
| **Architecture** | Fixed 3-layer | Multi-layer | Flexible | Flexible | Layer-based CNN |
| **Layer Spec** | (in, hid, out) | (in, nlayers, hid, out) | [in, h1, h2, out] | [in, h1, h2, out] | Build API |
| **Learning Rate** | Per-train | Per-train | Settable property | Settable property | Per-train |
| **Momentum** | No | No | Yes | Yes | No |
| **Sparse Networks** | No | No | Yes | Yes | No |
| **Convolutional** | No | No | No | No | Yes |
| **Returns Loss** | Yes | Yes | Yes | Yes | Yes |
| **Memory** | Low | Medium | Low | Medium | High |
| **NumPy Default** | Converts | Native | Converts | Native | Native |

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

**Use FannNetworkDouble when:**
- You need everything FannNetwork offers but with higher precision
- Numerical stability is critical (deep networks, long training sessions)
- You're working primarily with NumPy arrays (which default to float64)
- You need to minimize accumulation of floating-point errors
- You're comparing results with other float64-based implementations
- The extra memory cost (2x per weight) is acceptable

**Use CNNNetwork when:**
- You need convolutional layers for image processing or spatial data
- Building custom CNN architectures (e.g., MNIST, CIFAR-10 style networks)
- You want fine-grained control over layer configuration (kernel size, stride, padding)
- Working with 2D/3D structured input data
- Need to inspect individual layer properties and outputs
- Implementing image classification, object detection, or computer vision tasks
- Higher precision is required (float64)
- You prefer a layer-by-layer building API over fixed architecture

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
- [nn1](https://github.com/euske/nn1) - Convolutional Neural Network in C by euske
- Built with [Cython](https://cython.org/)
- Build system uses [scikit-build-core](https://scikit-build-core.readthedocs.io/)

## License

See LICENSE file for details.
