# TODO

Future enhancements and features for the cynn library.

## High Priority

Low effort, high impact improvements that should be tackled first.

### Batch Training Methods
- [ ] Add `train_batch()` methods to all network classes
- [ ] Process multiple examples in one call to reduce Python/C overhead
- [ ] Return aggregated statistics (mean loss, total loss, etc.)
- [ ] Support for shuffling within batch

### Standardize Training Interface
- [ ] Resolve inconsistency: `TinnNetwork.train()` returns loss, others don't
- [ ] Add `evaluate()` method to all classes for computing loss without training
- [ ] Consider adding `predict_with_loss()` methods
- [ ] Ensure consistent return values across all implementations

### More Examples
- [ ] Regression problem example (housing prices, etc.)
- [ ] Time series prediction example
- [ ] Simple image classification (MNIST subset)
- [ ] Transfer learning example (if applicable)
- [ ] Multi-class classification example

### CI/CD Pipeline
- [ ] Set up GitHub Actions for automated testing
- [ ] Multi-platform testing (Linux, macOS, Windows)
- [ ] Automated wheel building for releases
- [ ] PyPI publishing automation
- [ ] Code coverage reporting

### Context Manager Support
- [ ] Implement `__enter__`/`__exit__` for all network classes
- [ ] Automatic cleanup of resources
- [ ] Support for temporary networks in `with` blocks

## Medium Priority

Medium effort improvements with good value.

### Higher-Level Training API
- [ ] Add `fit(X, y, epochs, batch_size)` method similar to scikit-learn
- [ ] Built-in dataset splitting (train/validation)
- [ ] Automatic shuffling between epochs
- [ ] Progress bar integration (optional tqdm dependency)
- [ ] Validation during training with early stopping

### Model Metrics Utilities
- [ ] Add methods to compute accuracy for classification
- [ ] Add MSE, MAE, RMSE for regression
- [ ] Confusion matrix support
- [ ] ROC/AUC metrics
- [ ] Model comparison utilities

### Documentation Improvements
- [ ] Set up Sphinx or MkDocs for generated documentation
- [ ] Add tutorials for beginners
- [ ] Create migration guide between network types
- [ ] Add architecture decision records (ADRs)
- [ ] Document performance characteristics of each implementation

### Performance Benchmarks
- [ ] Create benchmarking suite comparing all four implementations
- [ ] Memory usage comparisons
- [ ] Speed benchmarks vs NumPy/pure Python implementations
- [ ] Multithreading scalability tests
- [ ] Document when to use each implementation

### Activation Function Control
- [ ] Expose FANN's activation function settings
- [ ] Allow per-layer activation configuration for FannNetwork
- [ ] Add activation function options to network constructors
- [ ] Support for sigmoid, tanh, ReLU, linear, etc.

## Lower Priority

Nice to have features for future consideration.

### Additional Serialization Formats
- [ ] Add JSON serialization support
- [ ] Add pickle support for Python-native serialization
- [ ] Consider ONNX export for interoperability
- [ ] Document trade-offs between formats

### Advanced Optimizer Support
- [ ] Expose FANN's training algorithms (RPROP, quickprop, etc.)
- [ ] Document available training algorithms
- [ ] Add optimizer comparison examples
- [ ] Benchmark different optimizers

### Network Visualization
- [ ] Export network architecture to Graphviz/DOT format
- [ ] Weight histogram plotting utilities
- [ ] Architecture diagram generation
- [ ] Training progress visualization

### Operator Overloading
- [ ] Enable `net(inputs)` syntax as alias for `predict(inputs)`
- [ ] More Pythonic interface
- [ ] Consider other useful operators

### Network Introspection
- [ ] Methods to inspect/export weights directly as arrays
- [ ] Gradient visualization support
- [ ] Layer activation inspection during forward pass
- [ ] Weight statistics (min, max, mean, std)

### Training Callbacks/Hooks
- [ ] Early stopping based on loss threshold
- [ ] Learning rate scheduling
- [ ] Progress monitoring callbacks for long training runs
- [ ] Custom callback interface

### Regularization Support
- [ ] L1/L2 regularization (if FANN supports it)
- [ ] Dropout support (if available in underlying libraries)
- [ ] Weight decay options
- [ ] Document regularization capabilities per implementation

## Quality Assurance

### Testing Improvements
- [ ] Add property-based testing with hypothesis
- [ ] Gradient checking tests
- [ ] Stress testing with random network configurations
- [ ] End-to-end training convergence tests
- [ ] Cross-platform compatibility tests
- [ ] NumPy integration edge cases
- [ ] Memory leak detection tests

### Performance Profiling
- [ ] Profile GIL release effectiveness
- [ ] Identify bottlenecks in Python/C boundary
- [ ] Memory profiling for large networks
- [ ] Optimize hot paths if needed

## Distribution & Packaging

### Pre-built Wheels
- [ ] Build wheels for Linux (manylinux)
- [ ] Build wheels for macOS (x86_64, arm64)
- [ ] Build wheels for Windows
- [ ] Support Python 3.10, 3.11, 3.12, 3.13+
- [ ] Automate wheel building in CI/CD

### Conda Package
- [ ] Create conda-forge recipe
- [ ] Submit to conda-forge
- [ ] Maintain conda package alongside PyPI
- [ ] Document conda installation

## Research & Exploration

### Potential New Features
- [ ] Investigate adding convolutional layer support (if any C library supports it)
- [ ] Research recurrent network support (RNN, LSTM)
- [ ] Evaluate adding ensemble methods
- [ ] Consider model compression techniques

### Upstream Contributions
- [ ] Document any patches to vendored libraries
- [ ] Consider contributing improvements back to Tinn, GENANN, FANN
- [ ] Track upstream changes and updates

## Documentation Tasks

### API Reference
- [ ] Complete docstrings for all public methods
- [ ] Add type hints to all functions
- [ ] Generate API documentation automatically
- [ ] Add usage examples in docstrings

### User Guide
- [ ] Getting started tutorial
- [ ] Common pitfalls and how to avoid them
- [ ] Performance tuning guide
- [ ] Best practices guide

### Developer Documentation
- [ ] Contributing guide
- [ ] Architecture overview
- [ ] Build system documentation
- [ ] Release process documentation

## Completed

Track completed items here by moving them from above sections.

- [x] Initial TinnNetwork implementation
- [x] GenannNetwork implementation
- [x] FannNetwork implementation
- [x] FannNetworkDouble implementation
- [x] Comprehensive test suite
- [x] Buffer protocol support
- [x] GIL-free execution
- [x] Save/load functionality for all network types
- [x] Basic documentation in README.md
- [x] Refactored Cython declarations into separate .pxd files
