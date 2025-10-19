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
- Comprehensive type stubs for GenannNetwork in _core.pyi
- GENANN library integration in CMake build system
- API comparison documentation in README.md

### Changed
- Updated README.md to document both TinnNetwork and GenannNetwork
- Project description now mentions both Tinn and GENANN libraries
- Added "Choosing Between TinnNetwork and GenannNetwork" section to README

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
