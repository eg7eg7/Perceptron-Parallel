
#include "vectors.h"
#include <omp.h>

void mult_scalar_with_vector(double* vector, int dim, double scalar, double* result_vector) {
#ifdef _PARALLEL_VECTORS
#pragma omp parallel for
#endif _PARALLEL_VECTORS
	for (int i = 0; i < dim; i++)
		result_vector[i] = vector[i] * scalar;
}
double mult_vector_with_vector(double* vector1, double* vector2, int dim) {
	double result = 0;
#ifdef _PARALLEL_VECTORS
#pragma omp parallel for reduction( + : result )
#endif _PARALLEL_VECTORS
	for (int i = 0; i < dim; i++)
		result += vector1[i] * vector2[i];
	return result;
}

void add_vector_to_vector(double* vector1, double* vector2, int dim, double* result_vector) {
#ifdef _PARALLEL_VECTORS
#pragma omp parallel for
#endif _PARALLEL_VECTORS
	for (int i = 0; i < dim; i++)
		result_vector[i] = vector1[i] + vector2[i];
}

void copy_vector(double* target, double* source, int dim) {
#ifdef _PARALLEL_VECTORS
#pragma omp parallel for
#endif _PARALLEL_VECTORS
	for (int i = 0; i < dim; i++) 
		target[i] = source[i];
}