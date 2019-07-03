#include "myApp.h"
#include <mpi.h>

int main(int argc, char *argv[])
{
	cudaError_t cudaStatus;
	time_t t;
	
	int array_size;
	int *A = 0; /*array of numbers to calculate*/
	int parallel_result = 0; /*result in parallel calculation*/
	long sequential_result = 0; /*result in sequential calculation*/
	int positiveCudaResults = 0;
	double t1, t2;

	int rank;
	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	srand((unsigned)time_t(&t));


	if (rank == 0)
	{
		initArray(&A, &array_size); 
		// Sequential solution
		t1 = omp_get_wtime();
		sequential_result = sequentialCalculation(A, array_size);
		t2 = omp_get_wtime();
		printf("===== Sequential time = %f, result = %ld==--\n", t2 - t1, sequential_result);

		//send array to host 1
		MPI_Send(&array_size, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
		MPI_Send(A + (array_size / 2), array_size / 2, MPI_INT, 1, 0, MPI_COMM_WORLD);

	}
	else if (rank == 1)
	{
		//receive size and array from host 0
		MPI_Recv(&array_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		A = (int*)malloc(array_size * sizeof(int));
		MPI_Recv(A, array_size / 2, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

	}


	t1 = omp_get_wtime();
	//Calculate 1/4 of array with OpenMP
	parallel_result += calcWithOpenMP(array_size, A);

	//Calculate 1/4 of array with CUDA
	cudaStatus = resultWithCuda(A + array_size / 4, array_size / 4, &positiveCudaResults);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "resultWithCuda failed!");
		return 1;
	}
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}


	parallel_result += positiveCudaResults;

	if (rank == 0)
	{
		//Receive results from host 1 and print
		int otherPositives;
		MPI_Recv(&otherPositives, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &status);
		parallel_result += otherPositives;
		t2 = omp_get_wtime();
#ifdef DEBUG_VERB_PRINT
		printf("received from process 1 result=%d\n", otherPositives);
#endif //DEBUG_TOK

		printf("===== END ===== Parallel Result = %d, time %f\n", parallel_result, t2 - t1);
	}
	else if (rank == 1)
	{
		//Send result to host 0
#ifdef DEBUG_VERB_PRINT
		printf("sending to process 0 result=%d\n", result);
#endif //DEBUG_TOK
		MPI_Send(&parallel_result, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
	}
	free(A);
	MPI_Finalize();
	return 0;
}

int calcWithOpenMP(int arraySize, int A[])
{
	int positiveOpenMPResults = 0;
#pragma omp parallel for reduction(+ : positiveOpenMPResults)
	for (int i = 0; i < arraySize / 4; i++)
	{
		if (f(A[i]) > 0)
			positiveOpenMPResults++;
	}
	return positiveOpenMPResults;
}

// Heavy function that runs on CPU 
double f(int i) {
	int j;
	double value;
	double result = 0;
	for (j = 1; j < HEAVY; j++)
	{
		value = (i + 1)*(j % 10);
		result += cos(value);
	}
	return cos(result);

}



int sequentialCalculation(int *A, int array_size) {
	int sequential_result = 0;
	for (int i = 0; i < array_size; i++)
	{
		if (f(A[i]) > 0) {
			sequential_result++;
		}
	}
	return sequential_result;
}

void readArrayFromFile(int **array, int *size) {
	const int line_size = 20;
	char* line = (char*)malloc(line_size);
	int num_elements_file = 0;
	errno_t err;
	FILE *file;
	err = fopen_s(&file, PATH, "r");
	if (file == NULL)
	{
		printf("Failed to open file.\n");
		exit(EXIT_FAILURE);
	}


	if (fgets(line, line_size, file) != NULL)
	{
		num_elements_file = atoi(line);
#ifdef DEBUG_FIXED_SIZE_FILE
		num_elements_file = FIXED_SIZE;
#endif //DEBUG_FIXED_SIZE_FILE
#ifdef DEBUG_VERB_PRINT
		printf("reading %d elements from file\n", num_elements_file);
#endif // DEBUG_TOK
		(*size) = num_elements_file;
		(*array) = (int*)malloc((*size) * sizeof(int));
	}
	for (int i = 0; i < num_elements_file && fgets(line, line_size, file) != NULL; i++)
	{
		(*array)[i] = atoi(line);
	}
}


void randNumbers(int **arr, int *size) {
	*size = (100 + rand() % 400) * 1000;
#ifdef DEBUG_RAND_FIXED_SIZE
	*size = FIXED_SIZE;
#endif //DEBUG_TOK_NOT_RAND
	(*arr) = (int*)malloc((*size) * sizeof(int));
	if (*arr == 0)
	{
		printf("malloc error");
		return;
	}
	printf("arr size = %d\n", *size);
	for (int i = 0; i < *size; i++) {
		(*arr)[i] = rand();
	}
}


void initArray(int **arr, int *size) {
#ifdef DEBUG_GENERATE_RANDOM
#ifdef DEBUG_VERB_PRINT
	printf("**Generating random numbers\n");
#endif // DEBUG_VERB_PRINT
	randNumbers(arr, size);
#else 
#ifdef DEBUG_VERB_PRINT
	printf("**Reading from file in path\n");
#endif // DEBUG_VERB_PRINT
	readArrayFromFile(arr, size);
#endif // DEBUG_GENERATE_RANDOM

	printf("array size = %d\n", *size);

#ifdef DEBUG_PRINT_ARRAY
	printArray(*arr, *size);
#endif // DEBUG_PRINT_ARRAY
}


void printArray(int* arr, int size) {
	for (int i = 0; i < size; i++) {
		printf("%d\n", arr[i]);
	}
}