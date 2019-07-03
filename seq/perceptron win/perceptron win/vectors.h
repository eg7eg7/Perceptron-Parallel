#pragma once
//#include "pch.h"
#ifndef Perceptron_vectors
#define Perceptron_vectors

void mult_scalar_with_vector(double* vector, int dim, double scalar, double* result_vector);
double mult_vector_with_vector(double* vector1, double* vector2, int dim);
void add_vector_to_vector(double* vector1, double* vector2, int dim, double* result_vector);

#endif