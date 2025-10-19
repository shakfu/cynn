# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
- Comprehensive type stubs for GenannNetwork, FannNetwork, and FannNetworkDouble in _core.pyi
- GENANN and FANN library integration in CMake build system
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

## [0.1.0] - 2024

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

[Unreleased]: https://github.com/shakfu/cynn/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/shakfu/cynn/releases/tag/v0.1.0
