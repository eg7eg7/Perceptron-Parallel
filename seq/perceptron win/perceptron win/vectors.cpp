//#include "pch.h"
#include "stdafx.h"
#include "vectors.h"

void mult_scalar_with_vector(double* vector, int dim, double scalar, double* result_vector) {
	for (int i = 0; i < dim; i++)
		result_vector[i] = vector[i] * scalar;
}
double mult_vector_with_vector(double* vector1, double* vector2, int dim) {
	double result = vector1[0] * vector2[0];
	for (int i = 1; i < dim; i++)
		result += vector1[i] * vector2[i];
	return result;
}
void add_vector_to_vector(double* vector1, double* vector2, int dim, double* result_vector) {
	for (int i = 0; i < dim; i++)
		result_vector[i] = vector1[i] + vector2[i];
}