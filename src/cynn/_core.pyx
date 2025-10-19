# cython: language_level=3

import array
import os
import time

cimport cpython.array as array

from libc.stdlib cimport srand
from libc.stdio cimport FILE, fopen, fclose

from . cimport nnet


def square(float x):
    return x * x


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
    cdef nnet.Tinn _impl
    cdef bint _owns_state

    def __cinit__(self, int inputs=0, int hidden=0, int outputs=0):
        self._owns_state = False
        # Only validate and build if parameters are provided (not loading)
        if inputs > 0 or hidden > 0 or outputs > 0:
            if inputs <= 0 or hidden <= 0 or outputs <= 0:
                raise ValueError("network dimensions must be positive")
            # Release GIL during network construction
            with nogil:
                self._impl = nnet.xtbuild(inputs, hidden, outputs)
            self._owns_state = True

    def __dealloc__(self):
        if self._owns_state:
            # Release GIL during cleanup
            with nogil:
                nnet.xtfree(self._impl)
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
            result = nnet.xttrain(self._impl, &input_mv[0], &target_mv[0], rate)

        return result

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
            output_ptr = nnet.xtpredict(self._impl, &input_mv[0])

        # Build output list (requires GIL)
        return [output_ptr[i] for i in range(nops)]

    cpdef void save(self, object path):
        cdef bytes encoded = _as_bytes_path(path)
        cdef const char* c_path = encoded
        # Release GIL during file I/O
        with nogil:
            nnet.xtsave(self._impl, c_path)

    @classmethod
    def load(cls, object path):
        cdef TinnNetwork instance = cls.__new__(cls)
        instance._owns_state = False
        cdef bytes encoded = _as_bytes_path(path)
        cdef const char* c_path = encoded
        # Release GIL during file I/O
        with nogil:
            instance._impl = nnet.xtload(c_path)
        instance._owns_state = True
        return instance


cdef class GenannNetwork:
    cdef nnet.genann* _impl
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
                self._impl = nnet.genann_init(inputs, hidden_layers, hidden, outputs)
            if self._impl == NULL:
                raise MemoryError("failed to allocate genann network")
            self._owns_state = True

    def __dealloc__(self):
        if self._owns_state and self._impl != NULL:
            # Release GIL during cleanup
            with nogil:
                nnet.genann_free(self._impl)
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

    cpdef void train(self, object inputs, object targets, double rate):
        if self._impl == NULL:
            raise RuntimeError("network not initialized")

        # Convert to memoryview - handles buffer protocol objects
        cdef double[::1] input_mv
        cdef double[::1] target_mv

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

        # Release GIL during expensive C computation
        with nogil:
            nnet.genann_train(self._impl, &input_mv[0], &target_mv[0], rate)

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
            output_ptr = nnet.genann_run(self._impl, &input_mv[0])

        # Build output list (requires GIL)
        return [output_ptr[i] for i in range(nops)]

    cpdef void randomize(self):
        if self._impl == NULL:
            raise RuntimeError("network not initialized")
        # Release GIL during randomization
        with nogil:
            nnet.genann_randomize(self._impl)

    def copy(self):
        if self._impl == NULL:
            raise RuntimeError("network not initialized")
        cdef GenannNetwork new_network = GenannNetwork.__new__(GenannNetwork)
        new_network._owns_state = False
        # Release GIL during copy
        with nogil:
            new_network._impl = nnet.genann_copy(self._impl)
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
                nnet.genann_write(self._impl, file_ptr)
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
                instance._impl = nnet.genann_read(file_ptr)
        finally:
            with nogil:
                fclose(file_ptr)
        if instance._impl == NULL:
            raise ValueError(f"failed to load genann network from {path}")
        instance._owns_state = True
        return instance
