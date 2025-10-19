# Declarations for doublefann library wrapper
# Separate from nnet.pxd to avoid function name conflicts with floatfann
# Uses dfann_wrapper.h which provides renamed functions

cdef extern from "dfann_wrapper.h":

    cdef struct fann:
        # Basic network properties
        unsigned int num_input
        unsigned int num_output
        unsigned int total_neurons
        unsigned int total_connections
        # Learning parameters
        float learning_rate
        float learning_momentum
        float connection_rate
        # Internal pointers (not accessed directly from Python)
        double* weights
        double* output

    # Network creation and destruction
    cdef fann* dfann_create_standard_array(unsigned int num_layers, const unsigned int* layers) nogil
    cdef fann* dfann_create_sparse_array(float connection_rate, unsigned int num_layers, const unsigned int* layers) nogil
    cdef fann* dfann_copy(fann* ann) nogil
    cdef void dfann_destroy(fann* ann) nogil

    # Training and prediction
    cdef double* dfann_run(fann* ann, double* input) nogil
    cdef void dfann_train(fann* ann, double* input, double* desired_output) nogil
    cdef void dfann_randomize_weights(fann* ann, double min_weight, double max_weight) nogil

    # File I/O
    cdef fann* dfann_create_from_file(const char* configuration_file) nogil
    cdef int dfann_save(fann* ann, const char* configuration_file) nogil

    # Network queries
    cdef unsigned int dfann_get_num_input(fann* ann) nogil
    cdef unsigned int dfann_get_num_output(fann* ann) nogil
    cdef unsigned int dfann_get_total_neurons(fann* ann) nogil
    cdef unsigned int dfann_get_total_connections(fann* ann) nogil
    cdef unsigned int dfann_get_num_layers(fann* ann) nogil
    cdef void dfann_get_layer_array(fann* ann, unsigned int* layers) nogil

    # Learning parameter accessors
    cdef float dfann_get_learning_rate(fann* ann) nogil
    cdef void dfann_set_learning_rate(fann* ann, float learning_rate) nogil
    cdef float dfann_get_learning_momentum(fann* ann) nogil
    cdef void dfann_set_learning_momentum(fann* ann, float learning_momentum) nogil
