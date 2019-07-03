#pragma once
#ifndef CUDAKERNEL_H
#define CUDAKERNEL_H
#define CHECK_ERRORS(status, msg, retValue) if ((status) != cudaSuccess) { fprintf(stderr, (msg)); return (retValue); }
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__device__ double fOnGPU(int i);

// Sum array elements to smaller array
__global__ void sumResultsKernel(int *result, int *sum_results, int *size);

// sum small array into an integer (in result[0])
__global__ void sumAllKernel(int *result);

// calculate f function on GPU for each thread
__global__ void fOnGPUKernel(int *result, int *array, int *size);

// Main function
cudaError_t resultWithCuda(int *array, int arraysize, int *result);

#endif