
#include <stdio.h>
#include <omp.h>
#include "myMacro.h"
#include "myApp.h"
#include <math.h>
#include "cudaKernel.h"
#include "Perceptron.h"

__device__ double fOnGPU(int i) {

	return 1;
}


__global__ void sumResultsKernel(int *result, int *sum_results, int size) {
	int i, index = threadIdx.x;
	sum_results[index] = POINT_CORRECT;
	int chunk_size = NUM_CUDA_CORES;
	int start_index = threadIdx.x * chunk_size;

	for (i = start_index; i < start_index + chunk_size; i++) {
		if (i >= size)
			break;
		if (result[i] != POINT_CORRECT) {
			sum_results[index] = result[i];
			break;
		}
	}

}

__device__ void mult_scalar_with_vector_device(double* vector, int dim, double scalar, double* result_vector) {
	for (int i = 0; i < dim; i++)
		result_vector[i] = vector[i] * scalar;
}

__device__ void add_vector_to_vector_device(double* vector1, double* vector2, int dim, double* result_vector) {
	for (int i = 0; i < dim; i++)
		result_vector[i] = vector1[i] + vector2[i];
}

__device__ void device_adjustW(double* W, double* temp_vector, Point* point, int K, double alpha) {
	double val = mult_vector_with_vector_device((*point).x, W, K + 1);
	int sign;
	if (val >= 0)
		sign = SET_A;
	else
		sign = SET_B;

	mult_scalar_with_vector_device((*point).x, K + 1, alpha*(-sign), temp_vector);
	add_vector_to_vector_device(W, temp_vector, K + 1, W);

}
//return 1 if all points correct, return -1 if W is adjusted
__global__ void adjustW_with_faulty_point(int *faulty_points,int size,Point* points, double* W,double* temp_vector,int K,double alpha) {
	int index;
	for (int i = 0; i < size; i++)
	{
		index = faulty_points[i];
		if (index != POINT_CORRECT)
		{
			//ADJUST W and return 
			device_adjustW(W,temp_vector, &(points[index]),K,alpha);
			faulty_points[0] = W_ADJUSTED;
			return;
		}
	}
	faulty_points[0] = ALL_POINTS_CORRECT;
}

__device__ double mult_vector_with_vector_device(double* vector1, double* vector2, int dim) {
	double result = vector1[0] * vector2[0];
	for (int i = 1; i < dim; i++)
		result += vector1[i] * vector2[i];
	return result;
}

__global__ void fOnGPUKernel(int *result, Point* points,double* W, int N,int K) {
	int index = threadIdx.x + blockIdx.x * NUM_CUDA_CORES;
	if (index >= N)
		return;
	double val = mult_vector_with_vector_device(points[index].x, W, K+1);
	if (val*points[index].set < 0)
		result[index] = index;
	else
		result[index] = POINT_CORRECT;

}


cudaError_t CopyPointsToDevice(Point* points, Point** dev_points,double*** dev_x_points, int N, int K) {
	*dev_x_points = (double**)malloc(sizeof(double*)*N);
	
	cudaError_t cudaStatus = cudaSuccess;
	cudaStatus = cudaSetDevice(0);
	CHECK_ERRORS(cudaStatus, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?", cudaErrorUnknown)

		
	cudaStatus = cudaMalloc((void**)dev_points, N * sizeof(Point));
	CHECK_ERRORS(cudaStatus, "cudaMalloc failed!", cudaErrorUnknown)

#pragma omp for
		for (int i = 0; i < N; i++)
		{
			cudaStatus = cudaMalloc((void**)&((*dev_x_points)[i]), (K + 1) * sizeof(double));
			CHECK_ERRORS(cudaStatus, "cudaMalloc failed!", cudaErrorUnknown);
			cudaMemcpy((*dev_x_points)[i], points[i].x, sizeof(double)*(K+1), cudaMemcpyHostToDevice);
			Point pt;
			pt.x = (*dev_x_points)[i];
			pt.set = points[i].set;
			cudaMemcpy(&(*dev_points)[i], &pt, sizeof(Point), cudaMemcpyHostToDevice);
		}
	return cudaStatus;
}

cudaError_t freePointsFromDevice(Point** dev_points, double*** dev_x_points, int N) {
	cudaError_t cudaStatus = cudaSuccess;
	cudaStatus = cudaSetDevice(0);
	CHECK_ERRORS(cudaStatus, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?", cudaErrorUnknown)
		// Allocate GPU buffer for temporary results - one member for each thread.
		for (int i = 0; i < N; i++)
		{
			cudaStatus = cudaFree((*dev_x_points)[i]);
			CHECK_ERRORS(cudaStatus, "cudaFree failed!", cudaErrorUnknown)
		}
	cudaStatus = cudaFree(*dev_points);
	CHECK_ERRORS(cudaStatus, "cudaFree failed!", cudaErrorUnknown)
	free(*dev_x_points);
	return cudaStatus;
}

cudaError_t get_quality_with_alpha_GPU(Point* points, double alpha, double* W, int N, int K, int LIMIT) {
	int* device_results;
	int* sum_results;
	double t1, t2;
	cudaError_t cudaStatus = cudaSuccess;
	double *W_dev,*W_dev_temp;
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	CHECK_ERRORS(cudaStatus, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?", cudaErrorUnknown)
	cudaStatus = cudaMalloc((void**)&W_dev, sizeof(double)*(K + 1));
	CHECK_ERRORS(cudaStatus, "cudaMalloc failed!", cudaErrorUnknown)
	cudaMemcpy(W_dev, W, sizeof(double)*(K + 1), cudaMemcpyHostToDevice);

	cudaStatus = cudaMalloc((void**)&W_dev_temp, sizeof(double)*(K + 1));
	CHECK_ERRORS(cudaStatus, "cudaMalloc failed!", cudaErrorUnknown)

	cudaStatus = cudaMalloc((void**)&device_results, sizeof(int)*N);
	CHECK_ERRORS(cudaStatus, "cudaMalloc failed!", cudaErrorUnknown)
	
	t1 = omp_get_wtime();
	int num_blocks = (int) ceil(N / (double) NUM_CUDA_CORES);
	cudaStatus = cudaMalloc((void**)&sum_results, sizeof(int)*num_blocks);
	CHECK_ERRORS(cudaStatus, "cudaMalloc failed!", cudaErrorUnknown)


	for (int i = 0;i < LIMIT; i++)
	{
		/*
		do f on all points
		*/
	fOnGPUKernel <<<num_blocks, NUM_CUDA_CORES>>> (device_results,points, W_dev, N,K);
	cudaStatus = cudaGetLastError();
	CHECK_ERRORS(cudaStatus, "fOnGPUKernel launch failed", cudaErrorUnknown)
	cudaStatus = cudaDeviceSynchronize();
	CHECK_ERRORS(cudaStatus, "Cuda sync failed", cudaErrorUnknown)
	/*
	find first point to fail
	*/
	sumResultsKernel <<<1, num_blocks >>> (device_results, sum_results,N);
	CHECK_ERRORS(cudaStatus, "sumResultsKernel launch failed", cudaErrorUnknown)
	cudaStatus = cudaDeviceSynchronize();
	CHECK_ERRORS(cudaStatus, "Cuda sync failed", cudaErrorUnknown)
	/*
	adjust W if fault found
	*/
	adjustW_with_faulty_point<<<1,1>>>(sum_results, num_blocks, points, W_dev, W_dev_temp, K, alpha);
	cudaStatus = cudaDeviceSynchronize();
	CHECK_ERRORS(cudaStatus, "Cuda sync failed", cudaErrorUnknown)
		if (sum_results[0] == ALL_POINTS_CORRECT)
			break;
	}
	
	t2 = omp_get_wtime();
	cudaMemcpy(W_dev, W, sizeof(double)*(K + 1), cudaMemcpyDeviceToHost);
	cudaFree(W_dev);
	cudaFree(W_dev_temp);
	cudaFree(device_results);
	cudaFree(sum_results);
	printf("\nGPU time = %f\n", t2 - t1);
	return cudaStatus;
}