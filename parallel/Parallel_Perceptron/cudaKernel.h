#pragma once
#ifndef CUDAKERNEL_H
#define CUDAKERNEL_H
#define CHECK_ERRORS(status, msg, retValue) if ((status) != cudaSuccess) { fprintf(stderr, (msg));return (retValue); }
#define CHECK_AND_SYNC_ERRORS(msg) cudaStatus = cudaGetLastError();CHECK_ERRORS(cudaStatus, msg, cudaErrorUnknown) cudaStatus = cudaDeviceSynchronize();CHECK_ERRORS(cudaStatus, "Cuda sync failed\n", cudaErrorUnknown)

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "Perceptron.h"

#define NUM_CUDA_CORES 1000
#define MALLOC_FLAG 1
#define FREE_MALLOC_FLAG 0

// Sum array elements to smaller array
__device__ void add_vector_to_vector_device(double* vector1, double* vector2, int dim, double* result_vector);
__device__ void mult_scalar_with_vector_device(double* vector, int dim, double scalar, double* result_vector);

//counts number of correct points with W
__global__ void count_correct_points_kernel(int *result, int *sum_results, int size);

//sums count of correct points
__global__ void sum_count_results_kernel(int *sum_results, int size);

// calculate f function on GPU for each thread
__global__ void f_on_GPU_kernel(int *result, Point* points, double* W, int N, int K);
__device__ double mult_vector_with_vector_device(double* vector1, double* vector2, int dim);


cudaError_t memcpy_double_array_to_device(double **dest, double **src, int n);
cudaError_t memcpy_point_array_to_device(Point **dest, Point **src, int n);

//malloc double array in GPU
cudaError_t cuda_malloc_double_by_size(double** arr, int arr_size);

//malloc Point array in GPU
cudaError_t cuda_malloc_point_by_size(Point** arr, int arr_size);

//will malloc if empty pointers, will free if not empty
cudaError_t cuda_malloc_and_free_pointers_from_quality_function(int N, int K, int num_blocks, double** W_dev, int** device_results, int** sum_results, int malloc_flag);

cudaError_t get_quality_with_GPU(Point* points, double* W, int N, int K, double* q);

cudaError_t set_device();

cudaError_t free_cuda_point_array(Point** dev_points);
#endif