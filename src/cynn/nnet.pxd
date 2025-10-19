cdef extern from "Tinn.h":

    ctypedef struct Tinn:
        # All the weights.
        float* w
        # Hidden to output layer weights.
        float* x
        # Biases.
        float* b
        # Hidden layer.
        float* h
        # Output layer.
        float* o
        # Number of biases - always two - Tinn only supports a single hidden layer.
        int nb
        # Number of weights.
        int nw
        # Number of inputs.
        int nips
        # Number of hidden neurons.
        int nhid
        # Number of outputs.
        int nops

    cdef float* xtpredict(Tinn, const float* in_) nogil
    cdef float xttrain(Tinn, const float* in_, const float* tg, float rate) nogil
    cdef Tinn xtbuild(int nips, int nhid, int nops) nogil
    cdef void xtsave(Tinn, const char* path) nogil
    cdef Tinn xtload(const char* path) nogil
    cdef void xtfree(Tinn) nogil
    cdef void xtprint(const float* arr, const int size) nogil


cdef extern from "stdio.h":
    ctypedef struct FILE:
        pass


cdef extern from "floatfann.h":

    ctypedef float fann_type

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
        fann_type* weights
        fann_type* output

    # Network creation and destruction
    cdef fann* fann_create_standard_array(unsigned int num_layers, const unsigned int* layers) nogil
    cdef fann* fann_create_sparse_array(float connection_rate, unsigned int num_layers, const unsigned int* layers) nogil
    cdef fann* fann_copy(fann* ann) nogil
    cdef void fann_destroy(fann* ann) nogil

    # Training and prediction
    cdef fann_type* fann_run(fann* ann, fann_type* input) nogil
    cdef void fann_train(fann* ann, fann_type* input, fann_type* desired_output) nogil
    cdef void fann_randomize_weights(fann* ann, fann_type min_weight, fann_type max_weight) nogil

    # File I/O
    cdef fann* fann_create_from_file(const char* configuration_file) nogil
    cdef int fann_save(fann* ann, const char* configuration_file) nogil

    # Network queries
    cdef unsigned int fann_get_num_input(fann* ann) nogil
    cdef unsigned int fann_get_num_output(fann* ann) nogil
    cdef unsigned int fann_get_total_neurons(fann* ann) nogil
    cdef unsigned int fann_get_total_connections(fann* ann) nogil
    cdef unsigned int fann_get_num_layers(fann* ann) nogil
    cdef void fann_get_layer_array(fann* ann, unsigned int* layers) nogil

    # Learning parameter accessors
    cdef float fann_get_learning_rate(fann* ann) nogil
    cdef void fann_set_learning_rate(fann* ann, float learning_rate) nogil
    cdef float fann_get_learning_momentum(fann* ann) nogil
    cdef void fann_set_learning_momentum(fann* ann, float learning_momentum) nogil


cdef extern from "genann.h":

    ctypedef struct genann:
        # How many inputs, outputs, and hidden neurons.
        int inputs
        int hidden_layers
        int hidden
        int outputs
        # Total number of weights, and size of weights buffer.
        int total_weights
        # Total number of neurons + inputs and size of output buffer.
        int total_neurons
        # All weights (total_weights long).
        double* weight
        # Stores input array and output of each neuron (total_neurons long).
        double* output
        # Stores delta of each hidden and output neuron (total_neurons - inputs long).
        double* delta

    cdef genann* genann_init(int inputs, int hidden_layers, int hidden, int outputs) nogil
    cdef void genann_free(genann* ann) nogil
    cdef const double* genann_run(const genann* ann, const double* inputs) nogil
    cdef void genann_train(const genann* ann, const double* inputs, const double* desired_outputs, double learning_rate) nogil
    cdef void genann_randomize(genann* ann) nogil
    cdef genann* genann_copy(const genann* ann) nogil
    cdef void genann_write(const genann* ann, FILE* out) nogil
    cdef genann* genann_read(FILE* in_file) nogil
