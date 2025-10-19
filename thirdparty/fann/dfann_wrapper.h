#ifndef DFANN_WRAPPER_H
#define DFANN_WRAPPER_H

/* Wrapper for doublefann to avoid naming conflicts with floatfann */

#ifdef __cplusplus
extern "C" {
#endif

struct fann;  /* Forward declaration */

/* Network creation and destruction */
struct fann* dfann_create_standard_array(unsigned int num_layers, const unsigned int* layers);
struct fann* dfann_create_sparse_array(float connection_rate, unsigned int num_layers, const unsigned int* layers);
struct fann* dfann_copy(struct fann* ann);
void dfann_destroy(struct fann* ann);

/* Training and prediction */
double* dfann_run(struct fann* ann, double* input);
void dfann_train(struct fann* ann, double* input, double* desired_output);
void dfann_randomize_weights(struct fann* ann, double min_weight, double max_weight);

/* File I/O */
struct fann* dfann_create_from_file(const char* configuration_file);
int dfann_save(struct fann* ann, const char* configuration_file);

/* Network queries */
unsigned int dfann_get_num_input(struct fann* ann);
unsigned int dfann_get_num_output(struct fann* ann);
unsigned int dfann_get_total_neurons(struct fann* ann);
unsigned int dfann_get_total_connections(struct fann* ann);
unsigned int dfann_get_num_layers(struct fann* ann);
void dfann_get_layer_array(struct fann* ann, unsigned int* layers);

/* Learning parameter accessors */
float dfann_get_learning_rate(struct fann* ann);
void dfann_set_learning_rate(struct fann* ann, float learning_rate);
float dfann_get_learning_momentum(struct fann* ann);
void dfann_set_learning_momentum(struct fann* ann, float learning_momentum);

#ifdef __cplusplus
}
#endif

#endif /* DFANN_WRAPPER_H */
