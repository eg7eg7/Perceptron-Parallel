
#include <stdio.h>
#include <omp.h>
#include "myMacro.h"
#include "myApp.h"
#include <math.h>
#include "cudaKernel.h"


__device__ double fOnGPU(int i) {
	int j;
	double value;
	double result = 0;

	for (j = 1; j < HEAVY; j++) {
		value = (i + 1)*(j % 10);
		result += cos(value);
	}
	return cos(result);

}


__global__ void sumResultsKernel(int *result, int *sum_results, int *size) {
	int arraysize = *size;
	int i;
	int index = threadIdx.x;
	sum_results[index] = 0;
	int chunk_size = (arraysize / 25);
	int start_index = threadIdx.x * chunk_size;

	for (i = start_index; i < start_index + chunk_size; i++) {
		if (i >= arraysize)
			break;
		if (result[i] > 0) {
			sum_results[index]++;
		}
	}

}


__global__ void sumAllKernel(int *result) {

	for (int i = 1; i < 25; i++)
	{
		result[0] += result[i];
	}
}


__global__ void fOnGPUKernel(int *result, int *array, int *size) {

	//calculate index
	int arraysize = *size;
	int index = threadIdx.x + blockIdx.x * 1000;

	if (index < arraysize)
	{
		if (fOnGPU(array[index]) > 0)
			result[index] = 1;
		else
			result[index] = 0;
	}

}

cudaError_t resultWithCuda(int *array, int arraysize, int *result)
{
	int num_blocks = ((int)(arraysize) / 1000) + 1;
	int *device_results = 0;
	cudaError_t cudaStatus = cudaSuccess;
	double t1, t2;
	int *array_device;
	int *arraysize_device;
	int *sum_results;

	t1 = omp_get_wtime();
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	CHECK_ERRORS(cudaStatus, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?", cudaErrorUnknown)

	// Allocate GPU buffer for temporary results - one member for each thread.
	cudaStatus = cudaMalloc((void**)&device_results, (arraysize) * sizeof(int));
	CHECK_ERRORS(cudaStatus, "cudaMalloc failed!", cudaErrorUnknown)

	cudaStatus = cudaMalloc((void**)&array_device, arraysize * sizeof(int));
	CHECK_ERRORS(cudaStatus, "cudaMalloc failed!", cudaErrorUnknown)

	cudaStatus = cudaMemcpy(array_device, array, arraysize * sizeof(int), cudaMemcpyHostToDevice);
	CHECK_ERRORS(cudaStatus, "CudaMemCpy failed!", cudaErrorUnknown)

	cudaStatus = cudaMalloc((void**)&arraysize_device, sizeof(int));
	CHECK_ERRORS(cudaStatus, "cudaMalloc failed!", cudaErrorUnknown)

	cudaStatus = cudaMemcpy(arraysize_device, &arraysize, sizeof(int), cudaMemcpyHostToDevice);
	CHECK_ERRORS(cudaStatus, "CudaMemCpy failed!", cudaErrorUnknown)

	/***********************************************************************************/
	//perform f on gpu using number of blocks with 1000 threads
	fOnGPUKernel << <num_blocks, 1000 >> > (device_results, array_device, arraysize_device);


	cudaStatus = cudaGetLastError();
	CHECK_ERRORS(cudaStatus, "fOnGPUKernel launch failed", cudaErrorUnknown)

	cudaStatus = cudaDeviceSynchronize();
	CHECK_ERRORS(cudaStatus, "Cuda sync failed", cudaErrorUnknown)

	cudaStatus = cudaMalloc((void**)&sum_results, 25 * sizeof(int));
	CHECK_ERRORS(cudaStatus, "CudaMalloc failed", cudaErrorUnknown)

	// save results in 25 threads
	sumResultsKernel << < 1, 25 >> > (device_results, sum_results, arraysize_device);
	cudaStatus = cudaDeviceSynchronize();
	CHECK_ERRORS(cudaStatus, "Cuda Sync launch failed", cudaErrorUnknown)

	cudaStatus = cudaGetLastError();
	CHECK_ERRORS(cudaStatus, "Cuda Sum results", cudaErrorUnknown)


	// get final result of calculations to sum_results
	sumAllKernel << < 1, 1 >> > (sum_results);


	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	CHECK_ERRORS(cudaStatus, "Cuda Sum results kernel failed", cudaErrorUnknown)

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	CHECK_ERRORS(cudaStatus, "Cuda Sync returned an error", cudaErrorUnknown)

	// Copy devResults[0] from GPU buffer to host memory.
	cudaStatus = cudaMemcpy((void *)result, (void *)(sum_results),
		1 * sizeof(int), cudaMemcpyDeviceToHost);
	CHECK_ERRORS(cudaStatus, "Cuda MemCopy failed", cudaErrorUnknown)

	cudaFree(array_device);
	cudaFree(arraysize_device);
	cudaFree(device_results);
	t2 = omp_get_wtime();
	printf("GPU time = %f=================\n", t2 - t1);
	return cudaStatus;
}
