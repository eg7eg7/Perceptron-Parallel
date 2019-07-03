#pragma once
#ifndef Perceptron_vectors
#define Perceptron_vectors
#include "pch.h"

void mult_scalar_with_vector(double* vector, int dim, double scalar, double* result_vector);
double mult_vector_with_vector(double* vector1, double* vector2, int dim);
void add_vector_to_vector(double* vector1, double* vector2, int dim, double* result_vector);
void copy_vector(double* target, double* source, int dim);
#endif