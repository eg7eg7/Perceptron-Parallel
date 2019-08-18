
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include "cudaKernel.h"
#include "Perceptron.h"

cudaError_t memcpyDoubleArrayToHost(double **dest, double **src, int n) {
	cudaError_t cudaStatus = cudaSuccess;
	cudaStatus = cudaMemcpy(*dest, *src, n * sizeof(double), cudaMemcpyDeviceToHost);
	CHECK_ERRORS(cudaStatus, "cudaMemcpy - double failed\n", cudaErrorUnknown);
		return cudaStatus;
}

cudaError_t memcpyDoubleArrayToDevice(double **dest, double **src, int n) {
	cudaError_t cudaStatus = cudaSuccess;
	cudaStatus = cudaMemcpy(*dest, *src, n * sizeof(double), cudaMemcpyHostToDevice);
	CHECK_ERRORS(cudaStatus, "cudaMemcpy - double failed\n", cudaErrorUnknown);
	return cudaStatus;
}

cudaError_t memcpyPointArrayToDevice(Point **dest, Point **src, int n) {
	cudaError_t cudaStatus = cudaSuccess;
	cudaStatus = cudaMemcpy(*dest, *src, n * sizeof(Point), cudaMemcpyHostToDevice);
	
	CHECK_ERRORS(cudaStatus, "cudaMemcpy - Point failed\n", cudaErrorUnknown);
	return cudaStatus;
}


__global__ void countCorrectPointsKernel(int *result, int *sum_results, int size) {
	int i, index = threadIdx.x;
	sum_results[index] = 0;
	int chunk_size = NUM_CUDA_CORES;
	int start_index = threadIdx.x * chunk_size;

	for (i = start_index; i < start_index + chunk_size; i++) {
		if (i >= size)
			break;
		if (result[i] != POINT_CORRECT) {
			sum_results[index]++;
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
__device__ int sign_device(double val)
{
	if (val >= 0)
		return SET_A;
	return SET_B;
}
__device__ void device_adjustW(double* W, double* temp_vector, Point* point, int K, double alpha) {
	double val = mult_vector_with_vector_device((*point).x, W, K + 1);
	int sign = sign_device(val);
	mult_scalar_with_vector_device((*point).x, K + 1, alpha*(-sign), temp_vector);
	add_vector_to_vector_device(W, temp_vector, K + 1, W);

}
__global__ void sumCountResultsKernel(int *sum_results, int size) {
	int sum=0;
	for (int i = 0; i < size; i++)
	{
		sum += sum_results[i];
	}
	sum_results[0] = sum;
}

__global__ void adjustW_with_faulty_point(int *faulty_points,int size,Point* points, double* W,double* temp_vector,int K,double alpha) {

	int index;
	for (int i = 0; i < size; i++)
	{
		index = faulty_points[i];
		if (index != POINT_CORRECT)
		{
			//adjust W and return 
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

__global__ void f_on_GPU_kernel(int *result, Point* points,double* W, int N,int K) {

	int index = threadIdx.x + blockIdx.x * NUM_CUDA_CORES;
	
	if (index >= N)
		return;
	double val = mult_vector_with_vector_device(points[index].x, W, K+1);
	if (sign_device(val) != points[index].set)
		result[index] = index;
	else
		result[index] = POINT_CORRECT;

}
cudaError_t set_device()
{
	cudaError_t cudaStatus = cudaSuccess;
	cudaStatus = cudaSetDevice(0);
	CHECK_ERRORS(cudaStatus, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?", cudaErrorUnknown);
	return cudaStatus;
}
cudaError_t cuda_malloc_double_by_size(double** arr, int arr_size)
{
	set_device();
	cudaError_t cudaStatus = cudaSuccess;
	cudaStatus = cudaMalloc((void**)arr, arr_size * sizeof(double));
	CHECK_ERRORS(cudaStatus, "cudaMalloc failed!", cudaErrorUnknown);
	return cudaStatus;
}
cudaError_t cuda_malloc_point_by_size(Point** arr, int arr_size)
{
	set_device();
	cudaError_t cudaStatus = cudaSuccess;
	cudaStatus = cudaMalloc((void**)arr, arr_size * sizeof(Point));
	CHECK_ERRORS(cudaStatus, "cudaMalloc failed!", cudaErrorUnknown);
	return cudaStatus;
}

cudaError_t free_cuda_point_array(Point** dev_points) {
	cudaError_t cudaStatus = cudaSuccess;
	set_device();
	Point point0;
	cudaStatus = cudaMemcpy(&point0, (*dev_points), sizeof(Point), cudaMemcpyDeviceToHost);
	CHECK_ERRORS(cudaStatus, "cudaMemCpy failed!", cudaErrorUnknown);

	//freeing dev_points[0].x will free the rest of the points memory as well
	cudaStatus = cudaFree(point0.x);
	CHECK_ERRORS(cudaStatus, "cudaFree failed!", cudaErrorUnknown);
	cudaStatus = cudaFree(*dev_points);
	CHECK_ERRORS(cudaStatus, "cudaFree failed!", cudaErrorUnknown);
	return cudaStatus;
}
cudaError_t cuda_malloc_and_free_pointers_from_quality_function(int N, int K, int num_blocks, double** W_dev, double** W_dev_temp,int** device_results, int** sum_results, int malloc_flag)
{
	static int is_last_malloc_flag = FREE_MALLOC_FLAG;
	static double *W_dev_p = 0, *W_dev_temp_p = 0;
	static int *device_results_p=0,*sum_results_p=0;
	cudaError_t cudaStatus = cudaSuccess;
	
	set_device();
	if (!is_last_malloc_flag && malloc_flag==MALLOC_FLAG)
	{
	cudaStatus = cudaMalloc((void**)W_dev, sizeof(double)*(K + 1));
	CHECK_ERRORS(cudaStatus, "cudaMalloc failed!\n", cudaErrorUnknown);
	cudaStatus = cudaMalloc((void**)W_dev_temp, sizeof(double)*(K + 1));
	CHECK_ERRORS(cudaStatus, "cudaMalloc failed!\n", cudaErrorUnknown);
	cudaStatus = cudaMalloc((void**)device_results, sizeof(int)*N);
	CHECK_ERRORS(cudaStatus, "cudaMalloc failed!\n", cudaErrorUnknown);
	cudaStatus = cudaMalloc((void**)sum_results, sizeof(int)*num_blocks);
	CHECK_ERRORS(cudaStatus, "cudaMalloc failed!\n", cudaErrorUnknown);

	W_dev_p = *W_dev;
	W_dev_temp_p = *W_dev_temp;
	device_results_p = *device_results;
	sum_results_p = *sum_results;
	is_last_malloc_flag = MALLOC_FLAG;
	}
	else if(is_last_malloc_flag && malloc_flag==FREE_MALLOC_FLAG)
	{
		cudaStatus = cudaFree(W_dev_p);
		CHECK_ERRORS(cudaStatus, "cudaFree failed!\n", cudaErrorUnknown);
		cudaStatus = cudaFree(W_dev_temp_p);
		CHECK_ERRORS(cudaStatus, "cudaFree failed!\n", cudaErrorUnknown);
		cudaStatus = cudaFree(device_results_p);
		CHECK_ERRORS(cudaStatus, "cudaFree failed!\n", cudaErrorUnknown);
		cudaStatus = cudaFree(sum_results_p);
		CHECK_ERRORS(cudaStatus, "cudaFree failed!\n", cudaErrorUnknown);
		is_last_malloc_flag = FREE_MALLOC_FLAG;
	}
	return cudaStatus;
}


cudaError_t get_quality_with_GPU(Point* points, double* W, int N, int K, double* q) {
	static int *device_results,*sum_results;
	static double *W_dev, *W_dev_temp;

	int count;
	int num_blocks = (int)ceil(N / (double)NUM_CUDA_CORES);
	cudaError_t cudaStatus = cudaSuccess;
	
	cuda_malloc_and_free_pointers_from_quality_function(N,K,num_blocks,&W_dev,&W_dev_temp,&device_results,&sum_results,MALLOC_FLAG);

	memcpyDoubleArrayToDevice(&W_dev, &W, K + 1);
	
	/*Do f on all points with adjusted W*/
	f_on_GPU_kernel <<<num_blocks, NUM_CUDA_CORES >>> (device_results, points, W_dev, N, K);
	CHECK_AND_SYNC_ERRORS("fOnGPUKernel launch failed\n");
	

	/*count number of correct points in each block*/
	countCorrectPointsKernel <<<1, num_blocks >>> (device_results, sum_results, N);
	CHECK_AND_SYNC_ERRORS("sumResultsKernel launch failed\n");
	/*count of incorrect points in sum_results[0]*/
	sumCountResultsKernel <<<1, 1>>>(sum_results, num_blocks);
	CHECK_AND_SYNC_ERRORS("adjustW_with_faulty_point launch failed\n");

	cudaStatus = cudaMemcpy(&count, &(sum_results[0]), sizeof(int), cudaMemcpyDeviceToHost);
	CHECK_ERRORS(cudaStatus, "Cudamemcpy failed\n", cudaErrorUnknown);

	*q = (count / (double) N);
	return cudaStatus;
}