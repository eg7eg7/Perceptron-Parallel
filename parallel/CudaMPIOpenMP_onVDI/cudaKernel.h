#pragma once
#ifndef CUDAKERNEL_H
#define CUDAKERNEL_H
#define CHECK_ERRORS(status, msg, retValue) if ((status) != cudaSuccess) { fprintf(stderr, (msg));return (retValue); }
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "Perceptron.h"
#define NUM_CUDA_CORES 1000

// Sum array elements to smaller array
__global__ void sumResultsKernel(int *result, int *sum_results, int size);
__device__ void add_vector_to_vector_device(double* vector1, double* vector2, int dim, double* result_vector);
__device__ void mult_scalar_with_vector_device(double* vector, int dim, double scalar, double* result_vector);

__global__ void adjustW_with_faulty_point(int *faulted_points, int size, Point* points, double* W,double* temp_vector, int K, double alpha);

// calculate f function on GPU for each thread
__global__ void fOnGPUKernel(int *result, Point* points, double* W, int N, int K, double* result_debug);
__device__ double mult_vector_with_vector_device(double* vector1, double* vector2, int dim);
__device__ void device_adjustW(double* W, double* temp_vector, Point* point, int K,double alpha);
// Main function

cudaError_t CopyPointsToDevice(Point* points, Point** dev_points,double*** dev_x_points, int N, int K);
cudaError_t freePointsFromDevice(Point** dev_points, double*** dev_x_points, int N);
cudaError_t get_quality_with_alpha_GPU(Point* points, double alpha, double* W, int N, int K, int LIMIT);
#endif