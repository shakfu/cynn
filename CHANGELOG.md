# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.2]

### Added
- **Batch Training Support** - All network types now support efficient batch training
  - `train_batch(inputs_list, targets_list, [rate], shuffle=False)` method for all networks
  - Returns dict with `'mean_loss'`, `'total_loss'`, and `'count'` statistics
  - Optional shuffling for improved convergence
  - Reduces Python/C overhead compared to training examples individually
  - GIL-free execution per training example
  - Comprehensive tests in `tests/test_batch_training.py` (16 new tests)

- **Loss Evaluation Without Training** - All network types now support validation without weight updates
  - `evaluate(inputs, targets)` method for all networks (TinnNetwork, GenannNetwork, FannNetwork, FannNetworkDouble, CNNNetwork)
  - Computes mean squared error without modifying network weights
  - Perfect for validation sets and monitoring generalization
  - GIL-free execution
  - Returns same loss type as `train()` (float32 or float64 depending on network)

### Changed
- **Standardized Training Interface (BREAKING CHANGE)** - Consistent return values across all network types
  - `GenannNetwork.train()` now returns `float` (mean squared error) instead of `None`
  - `FannNetwork.train()` now returns `float` (mean squared error) instead of `None`
  - `FannNetworkDouble.train()` now returns `float` (mean squared error) instead of `None`
  - `TinnNetwork.train()` and `CNNNetwork.train()` already returned loss (no change)
  - **Migration**: Code ignoring return values continues to work. Only code explicitly checking for `None` needs updating.
  - All `train()` methods now have consistent semantics: train on one example, return loss

- **Enhanced Documentation**
  - Added "Batch Training", "Evaluating Without Training", and "Training with Validation" sections to README.md
  - Updated API Reference with `evaluate()` and `train_batch()` for all network types
  - Updated comparison table to reflect all networks now return loss
  - Added Training API section to CLAUDE.md documenting standardized interface
  - Updated type stubs in `_core.pyi` for all network classes
  - Created `IMPLEMENTATION_SUMMARY.md` with detailed usage examples and migration guide

- **Test Suite Enhancements**
  - Updated 13 existing tests to handle new loss return values
  - All 240 tests passing (224 original + 16 new batch training tests)
  - Tests cover evaluate(), train_batch(), shuffling, edge cases, and consistency

### Fixed
- Removed non-existent `square` function from `__init__.py` exports

## [0.1.1]

### Added
- CNNNetwork class wrapping the nn1 Convolutional Neural Network C library
  - Layer-based architecture with support for input, convolutional, and fully-connected layers
  - Flexible network building API with `create_input_layer()`, `add_conv_layer()`, `add_full_layer()`
  - Support for arbitrary CNN architectures (e.g., MNIST-like, multi-conv networks)
  - Float64 precision for all computations
  - Convolutional layer parameters: kernel_size, padding, stride, std for weight initialization
  - Full layer parameters: num_nodes, std for weight initialization
  - CNNLayer class exposing layer properties (shape, type, weights, biases, conv parameters)
  - Network properties: `input_shape`, `output_size`, `num_layers`, `layers`
  - ReLU activation for convolutional layers, Tanh for hidden fully-connected layers, Softmax for output layer
  - GIL-free execution for parallel training and inference
  - Debug functionality with `dump()` method for network introspection
  - 31 comprehensive tests covering creation, validation, training, prediction, and complex architectures
- GenannNetwork class wrapping the GENANN C library
  - Multi-layer neural network support with arbitrary depth
  - Float64 precision for higher accuracy
  - Support for inputs, hidden_layers, hidden neurons per layer, and outputs
  - Additional properties: `total_weights`, `total_neurons`, `hidden_layers`
  - New methods: `randomize()` for re-randomizing weights, `copy()` for deep copying
  - Full buffer protocol support (lists, tuples, arrays, NumPy arrays)
  - GIL-free execution for parallel training and inference
  - Save/load functionality using FILE* based I/O
- FannNetwork class wrapping the FANN (Fast Artificial Neural Network) C library
  - Flexible multi-layer architecture with arbitrary layer configurations
  - Float32 precision for efficient computation
  - Support for fully connected and sparse networks (via connection_rate parameter)
  - Flexible constructor accepting list of layer sizes (e.g., `[2, 4, 3, 1]`)
  - Properties: `input_size`, `output_size`, `total_neurons`, `total_connections`, `num_layers`, `layers`
  - Settable learning parameters: `learning_rate`, `learning_momentum`
  - New methods: `randomize_weights()` for weight initialization, `copy()` for deep copying
  - Full buffer protocol support (lists, tuples, arrays, NumPy arrays)
  - GIL-free execution for parallel training and inference
  - Native FANN format save/load (text-based .fann files)
- FannNetworkDouble class providing float64 precision alternative to FannNetwork
  - Identical API to FannNetwork but uses double precision (float64)
  - Better numerical stability for deep networks or long training sessions
  - Compatible with NumPy's default float64 arrays without type conversion
  - Same flexible architecture and sparse network support as FannNetwork
  - All features from FannNetwork available in double precision
- Comprehensive type stubs for CNNNetwork, CNNLayer, GenannNetwork, FannNetwork, and FannNetworkDouble in _core.pyi
- nn1, GENANN and FANN library integration in CMake build system
- Created `cnn.pxd` for nn1 CNN C library declarations
- API comparison documentation in README.md

### Changed
- Refactored Cython declarations for better modularity and maintainability
  - Split monolithic `nnet.pxd` into library-specific declaration files
  - Created `tinn.pxd` for Tinn C library declarations
  - Created `genann.pxd` for GENANN C library declarations
  - Created `ffann.pxd` for float32 FANN C library declarations
  - Created `dfann.pxd` for float64 FANN C library declarations
  - Updated `_core.pyx` imports to use new modular declaration files
  - Improved code organization and reduced coupling between neural network backends
- Updated README.md to document TinnNetwork, GenannNetwork, FannNetwork, and FannNetworkDouble
- Project description now mentions Tinn, GENANN, and FANN libraries
- Added "Choosing Between Network Implementations" section to README
- Expanded feature comparison table with precision information

## [0.1.0]

### Added
- Initial release
- TinnNetwork class wrapping the Tinn C library
- Basic 3-layer neural network (input, hidden, output)
- Backpropagation training with configurable learning rate
- Save/load functionality for trained models
- Buffer protocol support for various input types
- GIL-free execution for multithreading
- Comprehensive test suite with 54 tests
- Type stubs for static type checking
- CMake-based build system with scikit-build-core
- Thread-safety tests demonstrating parallel execution
- NumPy compatibility tests
