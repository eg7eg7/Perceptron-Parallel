#pragma once

#ifndef MYAPP_H
#define myAPP_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include "myApp.h"

#include <omp.h>
#include <math.h>
#include "myMacro.h"
cudaError_t resultWithCuda(int *array, int arraysize, int *result);
#define MASTER 0
//Helper function
void printArray(int* arr, int size);
void readArrayFromFile(int **array, int *size);
int calcWithOpenMP(int arraySize, int A[]);
double f(int i);

//Initializes array from file/random, changes pointer to array and size
void initArray(int **arr, int *size);

//Optional - Create 100k-500k size random array
void randNumbers(int **arr, int *size);
int sequentialCalculation(int *A, int array_size);
int test_mult_scalar_with_vector();
int test_mult_vector_with_vector();
int test_add_vector_to_vector();
int test_adjust_W();
void runTests();
#endif

