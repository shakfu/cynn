# cython: language_level=3

import array
import os
import time

cimport cpython.array as array

from libc.stdlib cimport srand

from .nnet cimport (
    Tinn,
    xtpredict,
    xttrain,
    xtbuild,
    xtsave,
    xtload,
    xtfree,
)


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
    cdef Tinn _impl
    cdef bint _owns_state

    def __cinit__(self, int inputs=0, int hidden=0, int outputs=0):
        self._owns_state = False
        # Only validate and build if parameters are provided (not loading)
        if inputs > 0 or hidden > 0 or outputs > 0:
            if inputs <= 0 or hidden <= 0 or outputs <= 0:
                raise ValueError("network dimensions must be positive")
            # Release GIL during network construction
            with nogil:
                self._impl = xtbuild(inputs, hidden, outputs)
            self._owns_state = True

    def __dealloc__(self):
        if self._owns_state:
            # Release GIL during cleanup
            with nogil:
                xtfree(self._impl)
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
            result = xttrain(self._impl, &input_mv[0], &target_mv[0], rate)

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
            output_ptr = xtpredict(self._impl, &input_mv[0])

        # Build output list (requires GIL)
        return [output_ptr[i] for i in range(nops)]

    cpdef void save(self, object path):
        cdef bytes encoded = _as_bytes_path(path)
        cdef const char* c_path = encoded
        # Release GIL during file I/O
        with nogil:
            xtsave(self._impl, c_path)

    @classmethod
    def load(cls, object path):
        cdef TinnNetwork instance = cls.__new__(cls)
        instance._owns_state = False
        cdef bytes encoded = _as_bytes_path(path)
        cdef const char* c_path = encoded
        # Release GIL during file I/O
        with nogil:
            instance._impl = xtload(c_path)
        instance._owns_state = True
        return instance
