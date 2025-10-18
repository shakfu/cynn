# cython: language_level=3

import array
import os

cimport cpython.array as array

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
            self._impl = xtbuild(inputs, hidden, outputs)
            self._owns_state = True

    def __dealloc__(self):
        if self._owns_state:
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
        cdef array.array input_buf = array.array("f", inputs)
        cdef array.array target_buf = array.array("f", targets)
        if len(input_buf) != self._impl.nips:
            raise ValueError(
                f"expected {self._impl.nips} input values, received {len(input_buf)}"
            )
        if len(target_buf) != self._impl.nops:
            raise ValueError(
                f"expected {self._impl.nops} target values, received {len(target_buf)}"
            )
        cdef float[::1] input_mv = input_buf
        cdef float[::1] target_mv = target_buf
        return xttrain(self._impl, &input_mv[0], &target_mv[0], rate)

    cpdef list predict(self, object inputs):
        cdef array.array input_buf = array.array("f", inputs)
        if len(input_buf) != self._impl.nips:
            raise ValueError(
                f"expected {self._impl.nips} input values, received {len(input_buf)}"
            )
        cdef float[::1] input_mv = input_buf
        cdef float* output_ptr = xtpredict(self._impl, &input_mv[0])
        return [output_ptr[i] for i in range(self._impl.nops)]

    cpdef void save(self, object path):
        cdef bytes encoded = _as_bytes_path(path)
        xtsave(self._impl, encoded)

    @classmethod
    def load(cls, object path):
        cdef TinnNetwork instance = cls.__new__(cls)
        instance._owns_state = False
        cdef bytes encoded = _as_bytes_path(path)
        instance._impl = xtload(encoded)
        instance._owns_state = True
        return instance
