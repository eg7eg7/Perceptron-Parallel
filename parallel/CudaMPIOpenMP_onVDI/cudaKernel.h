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
__global__ void findFirstIncorrectPointInBlockKernel(int *result, int *sum_results, int size);
__device__ void add_vector_to_vector_device(double* vector1, double* vector2, int dim, double* result_vector);
__device__ void mult_scalar_with_vector_device(double* vector, int dim, double scalar, double* result_vector);

//finds first faulty points, if exists adjust W
__global__ void adjustW_with_faulty_point(int *faulted_points, int size, Point* points, double* W,double* temp_vector, int K, double alpha);

//counts number of correct points with W
__global__ void countCorrectPointsKernel(int *result, int *sum_results, int size);

//sums count of correct points
__global__ void sumCountResultsKernel(int *sum_results, int size);

// calculate f function on GPU for each thread
__global__ void fOnGPUKernel(int *result, Point* points, double* W, int N, int K);
__device__ double mult_vector_with_vector_device(double* vector1, double* vector2, int dim);
__device__ void device_adjustW(double* W, double* temp_vector, Point* point, int K,double alpha);


cudaError_t memcpyDoubleArrayToDevice(double **dest, double **src, int n);
cudaError_t memcpyPointArrayToDevice(Point **dest, Point **src, int n);

//malloc double array in GPU
cudaError_t cudaMallocDoubleBySize(double** arr, int arr_size);

//malloc Point array in GPU
cudaError_t cudaMallocPointBySize(Point** arr, int arr_size);

//will malloc if empty pointers, will free if not empty
cudaError_t cudaMallocAndFreePointersFromQualityFunction(int N, int K, int num_blocks, double** W_dev, double** W_dev_temp, int** device_results, int** sum_results, int malloc_flag);
cudaError_t syncAndCheckErrors(const char* msg);
cudaError_t get_quality_with_GPU(Point* points, double* W, int N, int K, double* q);
cudaError_t setDevice();
cudaError_t freeCudaPointArray(Point** dev_points);
#endif