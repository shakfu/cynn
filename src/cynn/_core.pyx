# cython: language_level=3

import array
import os
import time

cimport cpython.array as array

from libc.stdlib cimport srand, malloc, free
from libc.stdio cimport FILE, fopen, fclose, stdout

from . cimport tinn
from . cimport genann as genn
from . cimport ffann
from . cimport dfann
from . cimport cnn


def seed(unsigned int seed_value=0):
    """
    Seed the C random number generator used for weight initialization.

    If seed_value is 0 (default), uses current time.
    Call this before creating networks for reproducible results.
    """
    if seed_value == 0:
        seed_value = <unsigned int>time.time()
    srand(seed_value)


cdef bytes _as_bytes_path(object path):
    candidate = path
    if hasattr(candidate, "__fspath__"):
        candidate = os.fspath(candidate)
    if isinstance(candidate, bytes):
        return candidate
    if isinstance(candidate, str):
        return candidate.encode("utf-8")
    raise TypeError("path must be str, bytes, or os.PathLike")


cdef class TinnNetwork:
    cdef tinn.Tinn _impl
    cdef bint _owns_state

    def __cinit__(self, int inputs=0, int hidden=0, int outputs=0):
        self._owns_state = False
        # Only validate and build if parameters are provided (not loading)
        if inputs > 0 or hidden > 0 or outputs > 0:
            if inputs <= 0 or hidden <= 0 or outputs <= 0:
                raise ValueError("network dimensions must be positive")
            # Release GIL during network construction
            with nogil:
                self._impl = tinn.xtbuild(inputs, hidden, outputs)
            self._owns_state = True

    def __dealloc__(self):
        if self._owns_state:
            # Release GIL during cleanup
            with nogil:
                tinn.xtfree(self._impl)
            self._owns_state = False

    @property
    def input_size(self):
        return self._impl.nips

    @property
    def hidden_size(self):
        return self._impl.nhid

    @property
    def output_size(self):
        return self._impl.nops

    @property
    def shape(self):
        return (self._impl.nips, self._impl.nhid, self._impl.nops)

    cpdef float train(self, object inputs, object targets, float rate):
        # Convert to memoryview - handles buffer protocol objects
        cdef float[::1] input_mv
        cdef float[::1] target_mv
        cdef float result

        # Try to create memoryview from input
        try:
            input_mv = inputs
        except (TypeError, ValueError):
            # Not a buffer, convert via array.array
            input_mv = array.array('f', inputs)

        try:
            target_mv = targets
        except (TypeError, ValueError):
            # Not a buffer, convert via array.array
            target_mv = array.array('f', targets)

        if input_mv.shape[0] != self._impl.nips:
            raise ValueError(
                f"expected {self._impl.nips} input values, received {input_mv.shape[0]}"
            )
        if target_mv.shape[0] != self._impl.nops:
            raise ValueError(
                f"expected {self._impl.nops} target values, received {target_mv.shape[0]}"
            )

        # Release GIL during expensive C computation
        with nogil:
            result = tinn.xttrain(self._impl, &input_mv[0], &target_mv[0], rate)

        return result

    cpdef float evaluate(self, object inputs, object targets):
        """
        Compute loss without training.

        Args:
            inputs: Input values
            targets: Target output values

        Returns:
            Mean squared error between prediction and targets
        """
        # Convert to memoryview
        cdef float[::1] input_mv
        cdef float[::1] target_mv
        cdef float* output_ptr
        cdef float error, diff
        cdef int i, nops

        try:
            input_mv = inputs
        except (TypeError, ValueError):
            input_mv = array.array('f', inputs)

        try:
            target_mv = targets
        except (TypeError, ValueError):
            target_mv = array.array('f', targets)

        if input_mv.shape[0] != self._impl.nips:
            raise ValueError(
                f"expected {self._impl.nips} input values, received {input_mv.shape[0]}"
            )
        if target_mv.shape[0] != self._impl.nops:
            raise ValueError(
                f"expected {self._impl.nops} target values, received {target_mv.shape[0]}"
            )

        nops = self._impl.nops

        # Run prediction and compute MSE
        with nogil:
            output_ptr = tinn.xtpredict(self._impl, &input_mv[0])
            error = 0.0
            for i in range(nops):
                diff = output_ptr[i] - target_mv[i]
                error += diff * diff
            error /= nops

        return error

    cpdef dict train_batch(self, list inputs_list, list targets_list, float rate, bint shuffle=False):
        """
        Train on multiple examples in batch.

        Args:
            inputs_list: List of input arrays
            targets_list: List of target arrays
            rate: Learning rate
            shuffle: Whether to shuffle the batch before training

        Returns:
            dict with keys: 'mean_loss', 'total_loss', 'count'
        """
        cdef int batch_size = len(inputs_list)
        cdef float total_loss = 0.0
        cdef float loss
        cdef int i
        cdef list indices

        if len(targets_list) != batch_size:
            raise ValueError(
                f"inputs_list and targets_list must have same length: {batch_size} vs {len(targets_list)}"
            )

        if batch_size == 0:
            return {'mean_loss': 0.0, 'total_loss': 0.0, 'count': 0}

        # Create indices for shuffling
        indices = list(range(batch_size))
        if shuffle:
            import random
            random.shuffle(indices)

        # Train on each example
        for i in indices:
            loss = self.train(inputs_list[i], targets_list[i], rate)
            total_loss += loss

        return {
            'mean_loss': total_loss / batch_size,
            'total_loss': total_loss,
            'count': batch_size
        }

    cpdef list predict(self, object inputs):
        # Convert to memoryview - handles buffer protocol objects
        cdef float[::1] input_mv
        cdef float* output_ptr
        cdef int i, nops

        # Try to create memoryview from input
        try:
            input_mv = inputs
        except (TypeError, ValueError):
            # Not a buffer, convert via array.array
            input_mv = array.array('f', inputs)

        if input_mv.shape[0] != self._impl.nips:
            raise ValueError(
                f"expected {self._impl.nips} input values, received {input_mv.shape[0]}"
            )

        nops = self._impl.nops

        # Release GIL during expensive C computation
        with nogil:
            output_ptr = tinn.xtpredict(self._impl, &input_mv[0])

        # Build output list (requires GIL)
        return [output_ptr[i] for i in range(nops)]

    cpdef void save(self, object path):
        cdef bytes encoded = _as_bytes_path(path)
        cdef const char* c_path = encoded
        # Release GIL during file I/O
        with nogil:
            tinn.xtsave(self._impl, c_path)

    @classmethod
    def load(cls, object path):
        cdef TinnNetwork instance = cls.__new__(cls)
        instance._owns_state = False
        cdef bytes encoded = _as_bytes_path(path)
        cdef const char* c_path = encoded
        # Release GIL during file I/O
        with nogil:
            instance._impl = tinn.xtload(c_path)
        instance._owns_state = True
        return instance

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager. Cleanup handled by __dealloc__."""
        # Return False to propagate exceptions
        return False


cdef class GenannNetwork:
    cdef genn.genann* _impl
    cdef bint _owns_state

    def __cinit__(self, int inputs=0, int hidden_layers=0, int hidden=0, int outputs=0):
        self._owns_state = False
        self._impl = NULL
        # Only validate and build if parameters are provided (not loading)
        if inputs > 0 or hidden_layers > 0 or hidden > 0 or outputs > 0:
            if inputs <= 0 or hidden_layers <= 0 or hidden <= 0 or outputs <= 0:
                raise ValueError("network dimensions must be positive")
            # Release GIL during network construction
            with nogil:
                self._impl = genn.genann_init(inputs, hidden_layers, hidden, outputs)
            if self._impl == NULL:
                raise MemoryError("failed to allocate genann network")
            self._owns_state = True

    def __dealloc__(self):
        if self._owns_state and self._impl != NULL:
            # Release GIL during cleanup
            with nogil:
                genn.genann_free(self._impl)
            self._impl = NULL
            self._owns_state = False

    @property
    def input_size(self):
        if self._impl == NULL:
            raise RuntimeError("network not initialized")
        return self._impl.inputs

    @property
    def hidden_layers(self):
        if self._impl == NULL:
            raise RuntimeError("network not initialized")
        return self._impl.hidden_layers

    @property
    def hidden_size(self):
        if self._impl == NULL:
            raise RuntimeError("network not initialized")
        return self._impl.hidden

    @property
    def output_size(self):
        if self._impl == NULL:
            raise RuntimeError("network not initialized")
        return self._impl.outputs

    @property
    def shape(self):
        if self._impl == NULL:
            raise RuntimeError("network not initialized")
        return (self._impl.inputs, self._impl.hidden_layers, self._impl.hidden, self._impl.outputs)

    @property
    def total_weights(self):
        if self._impl == NULL:
            raise RuntimeError("network not initialized")
        return self._impl.total_weights

    @property
    def total_neurons(self):
        if self._impl == NULL:
            raise RuntimeError("network not initialized")
        return self._impl.total_neurons

    cpdef double train(self, object inputs, object targets, double rate):
        """
        Train the network on one example.

        Args:
            inputs: Input values
            targets: Target output values
            rate: Learning rate

        Returns:
            Mean squared error for this training example
        """
        if self._impl == NULL:
            raise RuntimeError("network not initialized")

        # Convert to memoryview - handles buffer protocol objects
        cdef double[::1] input_mv
        cdef double[::1] target_mv
        cdef const double* output_ptr
        cdef double error, diff
        cdef int i, nops

        # Try to create memoryview from input
        try:
            input_mv = inputs
        except (TypeError, ValueError):
            # Not a buffer, convert via array.array
            input_mv = array.array('d', inputs)

        try:
            target_mv = targets
        except (TypeError, ValueError):
            # Not a buffer, convert via array.array
            target_mv = array.array('d', targets)

        if input_mv.shape[0] != self._impl.inputs:
            raise ValueError(
                f"expected {self._impl.inputs} input values, received {input_mv.shape[0]}"
            )
        if target_mv.shape[0] != self._impl.outputs:
            raise ValueError(
                f"expected {self._impl.outputs} target values, received {target_mv.shape[0]}"
            )

        nops = self._impl.outputs

        # Release GIL during expensive C computation
        # First get outputs to compute error, then train
        with nogil:
            output_ptr = genn.genann_run(self._impl, &input_mv[0])
            error = 0.0
            for i in range(nops):
                diff = output_ptr[i] - target_mv[i]
                error += diff * diff
            error /= nops
            genn.genann_train(self._impl, &input_mv[0], &target_mv[0], rate)

        return error

    cpdef double evaluate(self, object inputs, object targets):
        """
        Compute loss without training.

        Args:
            inputs: Input values
            targets: Target output values

        Returns:
            Mean squared error between prediction and targets
        """
        if self._impl == NULL:
            raise RuntimeError("network not initialized")

        # Convert to memoryview
        cdef double[::1] input_mv
        cdef double[::1] target_mv
        cdef const double* output_ptr
        cdef double error, diff
        cdef int i, nops

        try:
            input_mv = inputs
        except (TypeError, ValueError):
            input_mv = array.array('d', inputs)

        try:
            target_mv = targets
        except (TypeError, ValueError):
            target_mv = array.array('d', targets)

        if input_mv.shape[0] != self._impl.inputs:
            raise ValueError(
                f"expected {self._impl.inputs} input values, received {input_mv.shape[0]}"
            )
        if target_mv.shape[0] != self._impl.outputs:
            raise ValueError(
                f"expected {self._impl.outputs} target values, received {target_mv.shape[0]}"
            )

        nops = self._impl.outputs

        # Run prediction and compute MSE
        with nogil:
            output_ptr = genn.genann_run(self._impl, &input_mv[0])
            error = 0.0
            for i in range(nops):
                diff = output_ptr[i] - target_mv[i]
                error += diff * diff
            error /= nops

        return error

    cpdef dict train_batch(self, list inputs_list, list targets_list, double rate, bint shuffle=False):
        """
        Train on multiple examples in batch.

        Args:
            inputs_list: List of input arrays
            targets_list: List of target arrays
            rate: Learning rate
            shuffle: Whether to shuffle the batch before training

        Returns:
            dict with keys: 'mean_loss', 'total_loss', 'count'
        """
        cdef int batch_size = len(inputs_list)
        cdef double total_loss = 0.0
        cdef double loss
        cdef int i
        cdef list indices

        if len(targets_list) != batch_size:
            raise ValueError(
                f"inputs_list and targets_list must have same length: {batch_size} vs {len(targets_list)}"
            )

        if batch_size == 0:
            return {'mean_loss': 0.0, 'total_loss': 0.0, 'count': 0}

        # Create indices for shuffling
        indices = list(range(batch_size))
        if shuffle:
            import random
            random.shuffle(indices)

        # Train on each example
        for i in indices:
            loss = self.train(inputs_list[i], targets_list[i], rate)
            total_loss += loss

        return {
            'mean_loss': total_loss / batch_size,
            'total_loss': total_loss,
            'count': batch_size
        }

    cpdef list predict(self, object inputs):
        if self._impl == NULL:
            raise RuntimeError("network not initialized")

        # Convert to memoryview - handles buffer protocol objects
        cdef double[::1] input_mv
        cdef const double* output_ptr
        cdef int i, nops

        # Try to create memoryview from input
        try:
            input_mv = inputs
        except (TypeError, ValueError):
            # Not a buffer, convert via array.array
            input_mv = array.array('d', inputs)

        if input_mv.shape[0] != self._impl.inputs:
            raise ValueError(
                f"expected {self._impl.inputs} input values, received {input_mv.shape[0]}"
            )

        nops = self._impl.outputs

        # Release GIL during expensive C computation
        with nogil:
            output_ptr = genn.genann_run(self._impl, &input_mv[0])

        # Build output list (requires GIL)
        return [output_ptr[i] for i in range(nops)]

    cpdef void randomize(self):
        if self._impl == NULL:
            raise RuntimeError("network not initialized")
        # Release GIL during randomization
        with nogil:
            genn.genann_randomize(self._impl)

    def copy(self):
        if self._impl == NULL:
            raise RuntimeError("network not initialized")
        cdef GenannNetwork new_network = GenannNetwork.__new__(GenannNetwork)
        new_network._owns_state = False
        # Release GIL during copy
        with nogil:
            new_network._impl = genn.genann_copy(self._impl)
        if new_network._impl == NULL:
            raise MemoryError("failed to copy genann network")
        new_network._owns_state = True
        return new_network

    cpdef void save(self, object path):
        if self._impl == NULL:
            raise RuntimeError("network not initialized")
        cdef bytes encoded = _as_bytes_path(path)
        cdef const char* c_path = encoded
        # Open file for writing
        cdef FILE* file_ptr
        with nogil:
            file_ptr = fopen(c_path, b"w")
        if file_ptr == NULL:
            raise IOError(f"failed to open file for writing: {path}")
        try:
            with nogil:
                genn.genann_write(self._impl, file_ptr)
        finally:
            with nogil:
                fclose(file_ptr)

    @classmethod
    def load(cls, object path):
        cdef GenannNetwork instance = cls.__new__(cls)
        instance._owns_state = False
        instance._impl = NULL
        cdef bytes encoded = _as_bytes_path(path)
        cdef const char* c_path = encoded
        # Open file for reading
        cdef FILE* file_ptr
        with nogil:
            file_ptr = fopen(c_path, b"r")
        if file_ptr == NULL:
            raise IOError(f"failed to open file for reading: {path}")
        try:
            with nogil:
                instance._impl = genn.genann_read(file_ptr)
        finally:
            with nogil:
                fclose(file_ptr)
        if instance._impl == NULL:
            raise ValueError(f"failed to load genann network from {path}")
        instance._owns_state = True
        return instance

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager. Cleanup handled by __dealloc__."""
        # Return False to propagate exceptions
        return False


cdef class FannNetwork:
    cdef ffann.fann* _impl
    cdef bint _owns_state

    def __cinit__(self, layers=None, connection_rate=1.0):
        cdef unsigned int num_layers
        cdef unsigned int* layer_array
        cdef unsigned int i
        cdef float c_rate
        cdef bint is_fully_connected

        self._owns_state = False
        self._impl = NULL

        if layers is not None:
            if not isinstance(layers, (list, tuple)):
                raise TypeError("layers must be a list or tuple")
            if len(layers) < 2:
                raise ValueError("network must have at least 2 layers (input and output)")
            for size in layers:
                if size <= 0:
                    raise ValueError("all layer sizes must be positive")

            num_layers = len(layers)
            c_rate = connection_rate
            is_fully_connected = (c_rate >= 1.0)
            layer_array = <unsigned int*>malloc(num_layers * sizeof(unsigned int))
            if layer_array == NULL:
                raise MemoryError("failed to allocate layer array")

            try:
                for i in range(num_layers):
                    layer_array[i] = layers[i]

                # Release GIL during network construction
                with nogil:
                    if is_fully_connected:
                        self._impl = ffann.fann_create_standard_array(num_layers, layer_array)
                    else:
                        self._impl = ffann.fann_create_sparse_array(c_rate, num_layers, layer_array)

                if self._impl == NULL:
                    raise MemoryError("failed to allocate FANN network")
                self._owns_state = True
            finally:
                free(layer_array)

    def __dealloc__(self):
        if self._owns_state and self._impl != NULL:
            # Release GIL during cleanup
            with nogil:
                ffann.fann_destroy(self._impl)
            self._impl = NULL
            self._owns_state = False

    @property
    def input_size(self):
        if self._impl == NULL:
            raise RuntimeError("network not initialized")
        return ffann.fann_get_num_input(self._impl)

    @property
    def output_size(self):
        if self._impl == NULL:
            raise RuntimeError("network not initialized")
        return ffann.fann_get_num_output(self._impl)

    @property
    def total_neurons(self):
        if self._impl == NULL:
            raise RuntimeError("network not initialized")
        return ffann.fann_get_total_neurons(self._impl)

    @property
    def total_connections(self):
        if self._impl == NULL:
            raise RuntimeError("network not initialized")
        return ffann.fann_get_total_connections(self._impl)

    @property
    def num_layers(self):
        if self._impl == NULL:
            raise RuntimeError("network not initialized")
        return ffann.fann_get_num_layers(self._impl)

    @property
    def layers(self):
        cdef unsigned int num_layers
        cdef unsigned int* layer_array
        cdef unsigned int i

        if self._impl == NULL:
            raise RuntimeError("network not initialized")

        num_layers = ffann.fann_get_num_layers(self._impl)
        layer_array = <unsigned int*>malloc(num_layers * sizeof(unsigned int))
        if layer_array == NULL:
            raise MemoryError("failed to allocate layer array")
        try:
            ffann.fann_get_layer_array(self._impl, layer_array)
            return [layer_array[i] for i in range(num_layers)]
        finally:
            free(layer_array)

    @property
    def learning_rate(self):
        if self._impl == NULL:
            raise RuntimeError("network not initialized")
        return ffann.fann_get_learning_rate(self._impl)

    @learning_rate.setter
    def learning_rate(self, float rate):
        if self._impl == NULL:
            raise RuntimeError("network not initialized")
        ffann.fann_set_learning_rate(self._impl, rate)

    @property
    def learning_momentum(self):
        if self._impl == NULL:
            raise RuntimeError("network not initialized")
        return ffann.fann_get_learning_momentum(self._impl)

    @learning_momentum.setter
    def learning_momentum(self, float momentum):
        if self._impl == NULL:
            raise RuntimeError("network not initialized")
        ffann.fann_set_learning_momentum(self._impl, momentum)

    cpdef list predict(self, object inputs):
        if self._impl == NULL:
            raise RuntimeError("network not initialized")

        # Convert to memoryview - handles buffer protocol objects
        cdef float[::1] input_mv
        cdef ffann.fann_type* output_ptr
        cdef int i
        cdef unsigned int num_outputs

        # Try to create memoryview from input
        try:
            input_mv = inputs
        except (TypeError, ValueError):
            # Not a buffer, convert via array.array
            input_mv = array.array('f', inputs)

        if input_mv.shape[0] != self._impl.num_input:
            raise ValueError(
                f"expected {self._impl.num_input} input values, received {input_mv.shape[0]}"
            )

        num_outputs = self._impl.num_output

        # Release GIL during expensive C computation
        with nogil:
            output_ptr = ffann.fann_run(self._impl, &input_mv[0])

        # Build output list (requires GIL)
        return [output_ptr[i] for i in range(num_outputs)]

    cpdef float train(self, object inputs, object targets):
        """
        Train the network on one example.

        Args:
            inputs: Input values
            targets: Target output values

        Returns:
            Mean squared error for this training example
        """
        if self._impl == NULL:
            raise RuntimeError("network not initialized")

        # Convert to memoryview - handles buffer protocol objects
        cdef float[::1] input_mv
        cdef float[::1] target_mv
        cdef ffann.fann_type* output_ptr
        cdef float error, diff
        cdef int i
        cdef unsigned int num_outputs

        # Try to create memoryview from input
        try:
            input_mv = inputs
        except (TypeError, ValueError):
            # Not a buffer, convert via array.array
            input_mv = array.array('f', inputs)

        try:
            target_mv = targets
        except (TypeError, ValueError):
            # Not a buffer, convert via array.array
            target_mv = array.array('f', targets)

        if input_mv.shape[0] != self._impl.num_input:
            raise ValueError(
                f"expected {self._impl.num_input} input values, received {input_mv.shape[0]}"
            )
        if target_mv.shape[0] != self._impl.num_output:
            raise ValueError(
                f"expected {self._impl.num_output} target values, received {target_mv.shape[0]}"
            )

        num_outputs = self._impl.num_output

        # Release GIL during expensive C computation
        # First get outputs to compute error, then train
        with nogil:
            output_ptr = ffann.fann_run(self._impl, &input_mv[0])
            error = 0.0
            for i in range(num_outputs):
                diff = output_ptr[i] - target_mv[i]
                error += diff * diff
            error /= num_outputs
            ffann.fann_train(self._impl, &input_mv[0], &target_mv[0])

        return error

    cpdef float evaluate(self, object inputs, object targets):
        """
        Compute loss without training.

        Args:
            inputs: Input values
            targets: Target output values

        Returns:
            Mean squared error between prediction and targets
        """
        if self._impl == NULL:
            raise RuntimeError("network not initialized")

        # Convert to memoryview
        cdef float[::1] input_mv
        cdef float[::1] target_mv
        cdef ffann.fann_type* output_ptr
        cdef float error, diff
        cdef int i
        cdef unsigned int num_outputs

        try:
            input_mv = inputs
        except (TypeError, ValueError):
            input_mv = array.array('f', inputs)

        try:
            target_mv = targets
        except (TypeError, ValueError):
            target_mv = array.array('f', targets)

        if input_mv.shape[0] != self._impl.num_input:
            raise ValueError(
                f"expected {self._impl.num_input} input values, received {input_mv.shape[0]}"
            )
        if target_mv.shape[0] != self._impl.num_output:
            raise ValueError(
                f"expected {self._impl.num_output} target values, received {target_mv.shape[0]}"
            )

        num_outputs = self._impl.num_output

        # Run prediction and compute MSE
        with nogil:
            output_ptr = ffann.fann_run(self._impl, &input_mv[0])
            error = 0.0
            for i in range(num_outputs):
                diff = output_ptr[i] - target_mv[i]
                error += diff * diff
            error /= num_outputs

        return error

    cpdef dict train_batch(self, list inputs_list, list targets_list, bint shuffle=False):
        """
        Train on multiple examples in batch.

        Args:
            inputs_list: List of input arrays
            targets_list: List of target arrays
            shuffle: Whether to shuffle the batch before training

        Returns:
            dict with keys: 'mean_loss', 'total_loss', 'count'
        """
        cdef int batch_size = len(inputs_list)
        cdef float total_loss = 0.0
        cdef float loss
        cdef int i
        cdef list indices

        if len(targets_list) != batch_size:
            raise ValueError(
                f"inputs_list and targets_list must have same length: {batch_size} vs {len(targets_list)}"
            )

        if batch_size == 0:
            return {'mean_loss': 0.0, 'total_loss': 0.0, 'count': 0}

        # Create indices for shuffling
        indices = list(range(batch_size))
        if shuffle:
            import random
            random.shuffle(indices)

        # Train on each example
        for i in indices:
            loss = self.train(inputs_list[i], targets_list[i])
            total_loss += loss

        return {
            'mean_loss': total_loss / batch_size,
            'total_loss': total_loss,
            'count': batch_size
        }

    cpdef void randomize_weights(self, float min_weight=-0.1, float max_weight=0.1):
        if self._impl == NULL:
            raise RuntimeError("network not initialized")
        # Release GIL during randomization
        with nogil:
            ffann.fann_randomize_weights(self._impl, min_weight, max_weight)

    def copy(self):
        if self._impl == NULL:
            raise RuntimeError("network not initialized")
        cdef FannNetwork new_network = FannNetwork.__new__(FannNetwork)
        new_network._owns_state = False
        # Release GIL during copy
        with nogil:
            new_network._impl = ffann.fann_copy(self._impl)
        if new_network._impl == NULL:
            raise MemoryError("failed to copy FANN network")
        new_network._owns_state = True
        return new_network

    cpdef void save(self, object path):
        if self._impl == NULL:
            raise RuntimeError("network not initialized")
        cdef bytes encoded = _as_bytes_path(path)
        cdef const char* c_path = encoded
        cdef int result
        # Release GIL during file I/O
        with nogil:
            result = ffann.fann_save(self._impl, c_path)
        if result == -1:
            raise IOError(f"failed to save FANN network to {path}")

    @classmethod
    def load(cls, object path):
        cdef FannNetwork instance = cls.__new__(cls)
        instance._owns_state = False
        instance._impl = NULL
        cdef bytes encoded = _as_bytes_path(path)
        cdef const char* c_path = encoded
        # Release GIL during file I/O
        with nogil:
            instance._impl = ffann.fann_create_from_file(c_path)
        if instance._impl == NULL:
            raise IOError(f"failed to load FANN network from {path}")
        instance._owns_state = True
        return instance

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager. Cleanup handled by __dealloc__."""
        # Return False to propagate exceptions
        return False


cdef class FannNetworkDouble:
    """Float64 (double precision) FANN neural network implementation."""
    cdef dfann.fann* _impl
    cdef bint _owns_state

    def __cinit__(self, layers=None, connection_rate=1.0):
        cdef unsigned int num_layers
        cdef unsigned int* layer_array
        cdef unsigned int i
        cdef float c_rate
        cdef bint is_fully_connected

        self._owns_state = False
        self._impl = NULL

        if layers is not None:
            if not isinstance(layers, (list, tuple)):
                raise TypeError("layers must be a list or tuple")
            if len(layers) < 2:
                raise ValueError("network must have at least 2 layers (input and output)")
            for size in layers:
                if size <= 0:
                    raise ValueError("all layer sizes must be positive")

            num_layers = len(layers)
            c_rate = connection_rate
            is_fully_connected = (c_rate >= 1.0)
            layer_array = <unsigned int*>malloc(num_layers * sizeof(unsigned int))
            if layer_array == NULL:
                raise MemoryError("failed to allocate layer array")

            try:
                for i in range(num_layers):
                    layer_array[i] = layers[i]

                # Release GIL during network construction
                with nogil:
                    if is_fully_connected:
                        self._impl = dfann.dfann_create_standard_array(num_layers, layer_array)
                    else:
                        self._impl = dfann.dfann_create_sparse_array(c_rate, num_layers, layer_array)

                if self._impl == NULL:
                    raise MemoryError("failed to allocate FANN network")
                self._owns_state = True
            finally:
                free(layer_array)

    def __dealloc__(self):
        if self._owns_state and self._impl != NULL:
            # Release GIL during cleanup
            with nogil:
                dfann.dfann_destroy(self._impl)
            self._impl = NULL
            self._owns_state = False

    @property
    def input_size(self):
        if self._impl == NULL:
            raise RuntimeError("network not initialized")
        return dfann.dfann_get_num_input(self._impl)

    @property
    def output_size(self):
        if self._impl == NULL:
            raise RuntimeError("network not initialized")
        return dfann.dfann_get_num_output(self._impl)

    @property
    def total_neurons(self):
        if self._impl == NULL:
            raise RuntimeError("network not initialized")
        return dfann.dfann_get_total_neurons(self._impl)

    @property
    def total_connections(self):
        if self._impl == NULL:
            raise RuntimeError("network not initialized")
        return dfann.dfann_get_total_connections(self._impl)

    @property
    def num_layers(self):
        if self._impl == NULL:
            raise RuntimeError("network not initialized")
        return dfann.dfann_get_num_layers(self._impl)

    @property
    def layers(self):
        cdef unsigned int num_layers
        cdef unsigned int* layer_array
        cdef unsigned int i

        if self._impl == NULL:
            raise RuntimeError("network not initialized")

        num_layers = dfann.dfann_get_num_layers(self._impl)
        layer_array = <unsigned int*>malloc(num_layers * sizeof(unsigned int))
        if layer_array == NULL:
            raise MemoryError("failed to allocate layer array")
        try:
            dfann.dfann_get_layer_array(self._impl, layer_array)
            return [layer_array[i] for i in range(num_layers)]
        finally:
            free(layer_array)

    @property
    def learning_rate(self):
        if self._impl == NULL:
            raise RuntimeError("network not initialized")
        return dfann.dfann_get_learning_rate(self._impl)

    @learning_rate.setter
    def learning_rate(self, float rate):
        if self._impl == NULL:
            raise RuntimeError("network not initialized")
        dfann.dfann_set_learning_rate(self._impl, rate)

    @property
    def learning_momentum(self):
        if self._impl == NULL:
            raise RuntimeError("network not initialized")
        return dfann.dfann_get_learning_momentum(self._impl)

    @learning_momentum.setter
    def learning_momentum(self, float momentum):
        if self._impl == NULL:
            raise RuntimeError("network not initialized")
        dfann.dfann_set_learning_momentum(self._impl, momentum)

    cpdef list predict(self, object inputs):
        if self._impl == NULL:
            raise RuntimeError("network not initialized")

        # Convert to memoryview - handles buffer protocol objects
        cdef double[::1] input_mv
        cdef double* output_ptr
        cdef int i
        cdef unsigned int num_outputs

        # Try to create memoryview from input
        try:
            input_mv = inputs
        except (TypeError, ValueError):
            # Not a buffer, convert via array.array (double precision)
            input_mv = array.array('d', inputs)

        if input_mv.shape[0] != self._impl.num_input:
            raise ValueError(
                f"expected {self._impl.num_input} input values, received {input_mv.shape[0]}"
            )

        num_outputs = self._impl.num_output

        # Release GIL during expensive C computation
        with nogil:
            output_ptr = dfann.dfann_run(self._impl, &input_mv[0])

        # Build output list (requires GIL)
        return [output_ptr[i] for i in range(num_outputs)]

    cpdef double train(self, object inputs, object targets):
        """
        Train the network on one example.

        Args:
            inputs: Input values
            targets: Target output values

        Returns:
            Mean squared error for this training example
        """
        if self._impl == NULL:
            raise RuntimeError("network not initialized")

        # Convert to memoryview - handles buffer protocol objects
        cdef double[::1] input_mv
        cdef double[::1] target_mv
        cdef double* output_ptr
        cdef double error, diff
        cdef int i
        cdef unsigned int num_outputs

        # Try to create memoryview from input
        try:
            input_mv = inputs
        except (TypeError, ValueError):
            # Not a buffer, convert via array.array (double precision)
            input_mv = array.array('d', inputs)

        try:
            target_mv = targets
        except (TypeError, ValueError):
            # Not a buffer, convert via array.array (double precision)
            target_mv = array.array('d', targets)

        if input_mv.shape[0] != self._impl.num_input:
            raise ValueError(
                f"expected {self._impl.num_input} input values, received {input_mv.shape[0]}"
            )
        if target_mv.shape[0] != self._impl.num_output:
            raise ValueError(
                f"expected {self._impl.num_output} target values, received {target_mv.shape[0]}"
            )

        num_outputs = self._impl.num_output

        # Release GIL during expensive C computation
        # First get outputs to compute error, then train
        with nogil:
            output_ptr = dfann.dfann_run(self._impl, &input_mv[0])
            error = 0.0
            for i in range(num_outputs):
                diff = output_ptr[i] - target_mv[i]
                error += diff * diff
            error /= num_outputs
            dfann.dfann_train(self._impl, &input_mv[0], &target_mv[0])

        return error

    cpdef double evaluate(self, object inputs, object targets):
        """
        Compute loss without training.

        Args:
            inputs: Input values
            targets: Target output values

        Returns:
            Mean squared error between prediction and targets
        """
        if self._impl == NULL:
            raise RuntimeError("network not initialized")

        # Convert to memoryview
        cdef double[::1] input_mv
        cdef double[::1] target_mv
        cdef double* output_ptr
        cdef double error, diff
        cdef int i
        cdef unsigned int num_outputs

        try:
            input_mv = inputs
        except (TypeError, ValueError):
            input_mv = array.array('d', inputs)

        try:
            target_mv = targets
        except (TypeError, ValueError):
            target_mv = array.array('d', targets)

        if input_mv.shape[0] != self._impl.num_input:
            raise ValueError(
                f"expected {self._impl.num_input} input values, received {input_mv.shape[0]}"
            )
        if target_mv.shape[0] != self._impl.num_output:
            raise ValueError(
                f"expected {self._impl.num_output} target values, received {target_mv.shape[0]}"
            )

        num_outputs = self._impl.num_output

        # Run prediction and compute MSE
        with nogil:
            output_ptr = dfann.dfann_run(self._impl, &input_mv[0])
            error = 0.0
            for i in range(num_outputs):
                diff = output_ptr[i] - target_mv[i]
                error += diff * diff
            error /= num_outputs

        return error

    cpdef dict train_batch(self, list inputs_list, list targets_list, bint shuffle=False):
        """
        Train on multiple examples in batch.

        Args:
            inputs_list: List of input arrays
            targets_list: List of target arrays
            shuffle: Whether to shuffle the batch before training

        Returns:
            dict with keys: 'mean_loss', 'total_loss', 'count'
        """
        cdef int batch_size = len(inputs_list)
        cdef double total_loss = 0.0
        cdef double loss
        cdef int i
        cdef list indices

        if len(targets_list) != batch_size:
            raise ValueError(
                f"inputs_list and targets_list must have same length: {batch_size} vs {len(targets_list)}"
            )

        if batch_size == 0:
            return {'mean_loss': 0.0, 'total_loss': 0.0, 'count': 0}

        # Create indices for shuffling
        indices = list(range(batch_size))
        if shuffle:
            import random
            random.shuffle(indices)

        # Train on each example
        for i in indices:
            loss = self.train(inputs_list[i], targets_list[i])
            total_loss += loss

        return {
            'mean_loss': total_loss / batch_size,
            'total_loss': total_loss,
            'count': batch_size
        }

    cpdef void randomize_weights(self, double min_weight=-0.1, double max_weight=0.1):
        if self._impl == NULL:
            raise RuntimeError("network not initialized")
        # Release GIL during randomization
        with nogil:
            dfann.dfann_randomize_weights(self._impl, min_weight, max_weight)

    def copy(self):
        if self._impl == NULL:
            raise RuntimeError("network not initialized")
        cdef FannNetworkDouble new_network = FannNetworkDouble.__new__(FannNetworkDouble)
        new_network._owns_state = False
        # Release GIL during copy
        with nogil:
            new_network._impl = dfann.dfann_copy(self._impl)
        if new_network._impl == NULL:
            raise MemoryError("failed to copy FANN network")
        new_network._owns_state = True
        return new_network

    cpdef void save(self, object path):
        if self._impl == NULL:
            raise RuntimeError("network not initialized")
        cdef bytes encoded = _as_bytes_path(path)
        cdef const char* c_path = encoded
        cdef int result
        # Release GIL during file I/O
        with nogil:
            result = dfann.dfann_save(self._impl, c_path)
        if result == -1:
            raise IOError(f"failed to save FANN network to {path}")

    @classmethod
    def load(cls, object path):
        cdef FannNetworkDouble instance = cls.__new__(cls)
        instance._owns_state = False
        instance._impl = NULL
        cdef bytes encoded = _as_bytes_path(path)
        cdef const char* c_path = encoded
        # Release GIL during file I/O
        with nogil:
            instance._impl = dfann.dfann_create_from_file(c_path)
        if instance._impl == NULL:
            raise IOError(f"failed to load FANN network from {path}")
        instance._owns_state = True
        return instance

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager. Cleanup handled by __dealloc__."""
        # Return False to propagate exceptions
        return False


cdef class CNNLayer:
    """
    Represents a single layer in a convolutional neural network.

    This class wraps the C Layer structure and should not be instantiated directly.
    Use CNNNetwork methods to build layers.
    """
    cdef cnn.Layer* _impl
    cdef bint _owns_state

    def __cinit__(self):
        self._impl = NULL
        self._owns_state = False

    def __dealloc__(self):
        # Layers are owned and destroyed by the network, not individual layer objects
        # Only destroy if this layer explicitly owns its state
        if self._owns_state and self._impl != NULL:
            with nogil:
                cnn.Layer_destroy(self._impl)
            self._impl = NULL
            self._owns_state = False

    @property
    def layer_id(self):
        """Layer ID in the network."""
        if self._impl == NULL:
            raise RuntimeError("layer not initialized")
        return self._impl.lid

    @property
    def shape(self):
        """Layer shape as (depth, width, height)."""
        if self._impl == NULL:
            raise RuntimeError("layer not initialized")
        return (self._impl.depth, self._impl.width, self._impl.height)

    @property
    def depth(self):
        """Layer depth dimension."""
        if self._impl == NULL:
            raise RuntimeError("layer not initialized")
        return self._impl.depth

    @property
    def width(self):
        """Layer width dimension."""
        if self._impl == NULL:
            raise RuntimeError("layer not initialized")
        return self._impl.width

    @property
    def height(self):
        """Layer height dimension."""
        if self._impl == NULL:
            raise RuntimeError("layer not initialized")
        return self._impl.height

    @property
    def num_nodes(self):
        """Total number of nodes (depth * width * height)."""
        if self._impl == NULL:
            raise RuntimeError("layer not initialized")
        return self._impl.nnodes

    @property
    def num_weights(self):
        """Number of weights in this layer."""
        if self._impl == NULL:
            raise RuntimeError("layer not initialized")
        return self._impl.nweights

    @property
    def num_biases(self):
        """Number of biases in this layer."""
        if self._impl == NULL:
            raise RuntimeError("layer not initialized")
        return self._impl.nbiases

    @property
    def layer_type(self):
        """Layer type as string: 'input', 'full', or 'conv'."""
        if self._impl == NULL:
            raise RuntimeError("layer not initialized")
        if self._impl.ltype == cnn.LAYER_INPUT:
            return 'input'
        elif self._impl.ltype == cnn.LAYER_FULL:
            return 'full'
        elif self._impl.ltype == cnn.LAYER_CONV:
            return 'conv'
        return 'unknown'

    @property
    def kernel_size(self):
        """Kernel size for convolutional layers (raises error for non-conv layers)."""
        if self._impl == NULL:
            raise RuntimeError("layer not initialized")
        if self._impl.ltype != cnn.LAYER_CONV:
            raise ValueError("kernel_size only available for convolutional layers")
        return self._impl.conv.kernsize

    @property
    def padding(self):
        """Padding for convolutional layers (raises error for non-conv layers)."""
        if self._impl == NULL:
            raise RuntimeError("layer not initialized")
        if self._impl.ltype != cnn.LAYER_CONV:
            raise ValueError("padding only available for convolutional layers")
        return self._impl.conv.padding

    @property
    def stride(self):
        """Stride for convolutional layers (raises error for non-conv layers)."""
        if self._impl == NULL:
            raise RuntimeError("layer not initialized")
        if self._impl.ltype != cnn.LAYER_CONV:
            raise ValueError("stride only available for convolutional layers")
        return self._impl.conv.stride

    def get_outputs(self):
        """Get the output values of this layer as a list."""
        if self._impl == NULL:
            raise RuntimeError("layer not initialized")
        cdef int i
        return [self._impl.outputs[i] for i in range(self._impl.nnodes)]


cdef class CNNNetwork:
    """
    A convolutional neural network with support for input, convolutional, and fully-connected layers.

    This class wraps the nn1 CNN C library. Networks are built by chaining layers together
    starting from an input layer.
    """
    cdef cnn.Layer* _input_layer
    cdef cnn.Layer* _output_layer
    cdef bint _owns_state
    cdef list _layer_refs  # Keep Python references to prevent GC

    def __cinit__(self):
        self._input_layer = NULL
        self._output_layer = NULL
        self._owns_state = False
        self._layer_refs = []

    def __dealloc__(self):
        cdef cnn.Layer* current
        cdef cnn.Layer* next_layer

        if self._owns_state and self._input_layer != NULL:
            # Destroy all layers starting from the input
            current = self._input_layer
            with nogil:
                while current != NULL:
                    next_layer = current.lnext
                    cnn.Layer_destroy(current)
                    current = next_layer
            self._input_layer = NULL
            self._output_layer = NULL
            self._owns_state = False
        self._layer_refs.clear()

    def create_input_layer(self, int depth, int width, int height):
        """
        Create an input layer. This must be called first to start building a network.

        Args:
            depth: Input depth (number of channels)
            width: Input width
            height: Input height

        Returns:
            CNNLayer wrapper for the created layer

        Raises:
            ValueError: If network already has an input layer or dimensions invalid
        """
        if self._input_layer != NULL:
            raise ValueError("network already has an input layer")
        if depth <= 0 or width <= 0 or height <= 0:
            raise ValueError("all dimensions must be positive")

        cdef cnn.Layer* layer
        with nogil:
            layer = cnn.Layer_create_input(depth, width, height)

        if layer == NULL:
            raise MemoryError("failed to create input layer")

        self._input_layer = layer
        self._output_layer = layer
        self._owns_state = True

        # Create Python wrapper
        cdef CNNLayer py_layer = CNNLayer.__new__(CNNLayer)
        py_layer._impl = layer
        py_layer._owns_state = False  # Network owns the C layer
        self._layer_refs.append(py_layer)

        return py_layer

    def add_conv_layer(self, int depth, int width, int height,
                       int kernel_size, int padding=0, int stride=1, double std=0.1):
        """
        Add a convolutional layer to the network.

        Args:
            depth: Number of output channels (filters)
            width: Output width
            height: Output height
            kernel_size: Size of the convolution kernel (must be odd)
            padding: Padding size (default: 0)
            stride: Stride for convolution (default: 1)
            std: Standard deviation for weight initialization (default: 0.1)

        Returns:
            CNNLayer wrapper for the created layer

        Raises:
            ValueError: If network has no input layer or parameters are invalid
            MemoryError: If layer creation fails
        """
        if self._output_layer == NULL:
            raise ValueError("must create input layer first")
        if depth <= 0 or width <= 0 or height <= 0:
            raise ValueError("all dimensions must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be positive and odd")
        if stride <= 0:
            raise ValueError("stride must be positive")

        cdef cnn.Layer* layer
        with nogil:
            layer = cnn.Layer_create_conv(
                self._output_layer, depth, width, height,
                kernel_size, padding, stride, std)

        if layer == NULL:
            raise MemoryError("failed to create convolutional layer")

        self._output_layer = layer

        # Create Python wrapper
        cdef CNNLayer py_layer = CNNLayer.__new__(CNNLayer)
        py_layer._impl = layer
        py_layer._owns_state = False  # Network owns the C layer
        self._layer_refs.append(py_layer)

        return py_layer

    def add_full_layer(self, int num_nodes, double std=0.1):
        """
        Add a fully-connected layer to the network.

        Args:
            num_nodes: Number of nodes in this layer
            std: Standard deviation for weight initialization (default: 0.1)

        Returns:
            CNNLayer wrapper for the created layer

        Raises:
            ValueError: If network has no input layer or num_nodes invalid
            MemoryError: If layer creation fails
        """
        if self._output_layer == NULL:
            raise ValueError("must create input layer first")
        if num_nodes <= 0:
            raise ValueError("num_nodes must be positive")

        cdef cnn.Layer* layer
        with nogil:
            layer = cnn.Layer_create_full(self._output_layer, num_nodes, std)

        if layer == NULL:
            raise MemoryError("failed to create fully-connected layer")

        self._output_layer = layer

        # Create Python wrapper
        cdef CNNLayer py_layer = CNNLayer.__new__(CNNLayer)
        py_layer._impl = layer
        py_layer._owns_state = False  # Network owns the C layer
        self._layer_refs.append(py_layer)

        return py_layer

    @property
    def input_shape(self):
        """Shape of the input layer as (depth, width, height)."""
        if self._input_layer == NULL:
            raise RuntimeError("network has no input layer")
        return (self._input_layer.depth, self._input_layer.width, self._input_layer.height)

    @property
    def output_size(self):
        """Number of output nodes in the final layer."""
        if self._output_layer == NULL:
            raise RuntimeError("network has no layers")
        return self._output_layer.nnodes

    @property
    def num_layers(self):
        """Total number of layers in the network."""
        if self._input_layer == NULL:
            return 0
        return len(self._layer_refs)

    @property
    def layers(self):
        """List of all layer wrappers in the network."""
        return list(self._layer_refs)

    cpdef list predict(self, object inputs):
        """
        Make a prediction given input values.

        Args:
            inputs: Input values (length must match input layer size: depth * width * height)

        Returns:
            List of output values from the final layer

        Raises:
            RuntimeError: If network has no layers
            ValueError: If input size doesn't match
        """
        if self._input_layer == NULL:
            raise RuntimeError("network has no input layer")
        if self._output_layer == NULL:
            raise RuntimeError("network has no output layer")

        # Convert to memoryview
        cdef double[::1] input_mv
        try:
            input_mv = inputs
        except (TypeError, ValueError):
            input_mv = array.array('d', inputs)

        cdef int expected_size = self._input_layer.nnodes
        if input_mv.shape[0] != expected_size:
            raise ValueError(
                f"expected {expected_size} input values, received {input_mv.shape[0]}"
            )

        # Set inputs and perform forward pass
        with nogil:
            cnn.Layer_setInputs(self._input_layer, &input_mv[0])

        # Get outputs from final layer
        cdef int output_size = self._output_layer.nnodes
        cdef int i
        return [self._output_layer.outputs[i] for i in range(output_size)]

    cpdef double train(self, object inputs, object targets, double learning_rate):
        """
        Train the network on one example using backpropagation.

        Args:
            inputs: Input values (length must match input layer size)
            targets: Target output values (length must match output layer size)
            learning_rate: Learning rate for weight updates

        Returns:
            Total squared error for this training example

        Raises:
            RuntimeError: If network has no layers
            ValueError: If input or target size doesn't match
        """
        if self._input_layer == NULL:
            raise RuntimeError("network has no input layer")
        if self._output_layer == NULL:
            raise RuntimeError("network has no output layer")

        # Convert inputs to memoryview
        cdef double[::1] input_mv
        try:
            input_mv = inputs
        except (TypeError, ValueError):
            input_mv = array.array('d', inputs)

        # Convert targets to memoryview
        cdef double[::1] target_mv
        try:
            target_mv = targets
        except (TypeError, ValueError):
            target_mv = array.array('d', targets)

        cdef int expected_input_size = self._input_layer.nnodes
        if input_mv.shape[0] != expected_input_size:
            raise ValueError(
                f"expected {expected_input_size} input values, received {input_mv.shape[0]}"
            )

        cdef int expected_output_size = self._output_layer.nnodes
        if target_mv.shape[0] != expected_output_size:
            raise ValueError(
                f"expected {expected_output_size} target values, received {target_mv.shape[0]}"
            )

        cdef double error
        # Perform forward pass, backpropagation, and weight update
        with nogil:
            cnn.Layer_setInputs(self._input_layer, &input_mv[0])
            cnn.Layer_learnOutputs(self._output_layer, &target_mv[0])
            error = cnn.Layer_getErrorTotal(self._output_layer)
            cnn.Layer_update(self._output_layer, learning_rate)

        return error

    cpdef double evaluate(self, object inputs, object targets):
        """
        Compute loss without training.

        Args:
            inputs: Input values (length must match input layer size)
            targets: Target output values (length must match output layer size)

        Returns:
            Mean squared error between prediction and targets

        Raises:
            RuntimeError: If network has no layers
            ValueError: If input or target size doesn't match
        """
        if self._input_layer == NULL:
            raise RuntimeError("network has no input layer")
        if self._output_layer == NULL:
            raise RuntimeError("network has no output layer")

        # Convert inputs to memoryview
        cdef double[::1] input_mv
        try:
            input_mv = inputs
        except (TypeError, ValueError):
            input_mv = array.array('d', inputs)

        # Convert targets to memoryview
        cdef double[::1] target_mv
        try:
            target_mv = targets
        except (TypeError, ValueError):
            target_mv = array.array('d', targets)

        cdef int expected_input_size = self._input_layer.nnodes
        if input_mv.shape[0] != expected_input_size:
            raise ValueError(
                f"expected {expected_input_size} input values, received {input_mv.shape[0]}"
            )

        cdef int expected_output_size = self._output_layer.nnodes
        if target_mv.shape[0] != expected_output_size:
            raise ValueError(
                f"expected {expected_output_size} target values, received {target_mv.shape[0]}"
            )

        cdef double error, diff
        cdef int i
        # Perform forward pass only and compute error manually
        with nogil:
            cnn.Layer_setInputs(self._input_layer, &input_mv[0])

        # Compute MSE manually
        error = 0.0
        for i in range(expected_output_size):
            diff = self._output_layer.outputs[i] - target_mv[i]
            error += diff * diff
        error /= expected_output_size

        return error

    cpdef dict train_batch(self, list inputs_list, list targets_list, double learning_rate, bint shuffle=False):
        """
        Train on multiple examples in batch.

        Args:
            inputs_list: List of input arrays
            targets_list: List of target arrays
            learning_rate: Learning rate for weight updates
            shuffle: Whether to shuffle the batch before training

        Returns:
            dict with keys: 'mean_loss', 'total_loss', 'count'
        """
        cdef int batch_size = len(inputs_list)
        cdef double total_loss = 0.0
        cdef double loss
        cdef int i
        cdef list indices

        if len(targets_list) != batch_size:
            raise ValueError(
                f"inputs_list and targets_list must have same length: {batch_size} vs {len(targets_list)}"
            )

        if batch_size == 0:
            return {'mean_loss': 0.0, 'total_loss': 0.0, 'count': 0}

        # Create indices for shuffling
        indices = list(range(batch_size))
        if shuffle:
            import random
            random.shuffle(indices)

        # Train on each example
        for i in indices:
            loss = self.train(inputs_list[i], targets_list[i], learning_rate)
            total_loss += loss

        return {
            'mean_loss': total_loss / batch_size,
            'total_loss': total_loss,
            'count': batch_size
        }

    def dump(self):
        """
        Print debug information about all layers to stdout.

        Raises:
            RuntimeError: If network has no layers
        """
        if self._input_layer == NULL:
            raise RuntimeError("network has no input layer")

        cdef cnn.Layer* current = self._input_layer
        with nogil:
            while current != NULL:
                cnn.Layer_dump(current, stdout)
                current = current.lnext

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager. Cleanup handled by __dealloc__."""
        # Return False to propagate exceptions
        return False
