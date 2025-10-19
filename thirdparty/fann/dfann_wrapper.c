/* Wrapper implementation for doublefann */
#include "doublefann.h"
#include "dfann_wrapper.h"

/* Network creation and destruction */
struct fann* dfann_create_standard_array(unsigned int num_layers, const unsigned int* layers) {
    return fann_create_standard_array(num_layers, layers);
}

struct fann* dfann_create_sparse_array(float connection_rate, unsigned int num_layers, const unsigned int* layers) {
    return fann_create_sparse_array(connection_rate, num_layers, layers);
}

struct fann* dfann_copy(struct fann* ann) {
    return fann_copy(ann);
}

void dfann_destroy(struct fann* ann) {
    fann_destroy(ann);
}

/* Training and prediction */
double* dfann_run(struct fann* ann, double* input) {
    return (double*)fann_run(ann, input);
}

void dfann_train(struct fann* ann, double* input, double* desired_output) {
    fann_train(ann, input, desired_output);
}

void dfann_randomize_weights(struct fann* ann, double min_weight, double max_weight) {
    fann_randomize_weights(ann, min_weight, max_weight);
}

/* File I/O */
struct fann* dfann_create_from_file(const char* configuration_file) {
    return fann_create_from_file(configuration_file);
}

int dfann_save(struct fann* ann, const char* configuration_file) {
    return fann_save(ann, configuration_file);
}

/* Network queries */
unsigned int dfann_get_num_input(struct fann* ann) {
    return fann_get_num_input(ann);
}

unsigned int dfann_get_num_output(struct fann* ann) {
    return fann_get_num_output(ann);
}

unsigned int dfann_get_total_neurons(struct fann* ann) {
    return fann_get_total_neurons(ann);
}

unsigned int dfann_get_total_connections(struct fann* ann) {
    return fann_get_total_connections(ann);
}

unsigned int dfann_get_num_layers(struct fann* ann) {
    return fann_get_num_layers(ann);
}

void dfann_get_layer_array(struct fann* ann, unsigned int* layers) {
    fann_get_layer_array(ann, layers);
}

/* Learning parameter accessors */
float dfann_get_learning_rate(struct fann* ann) {
    return fann_get_learning_rate(ann);
}

void dfann_set_learning_rate(struct fann* ann, float learning_rate) {
    fann_set_learning_rate(ann, learning_rate);
}

float dfann_get_learning_momentum(struct fann* ann) {
    return fann_get_learning_momentum(ann);
}

void dfann_set_learning_momentum(struct fann* ann, float learning_momentum) {
    fann_set_learning_momentum(ann, learning_momentum);
}
