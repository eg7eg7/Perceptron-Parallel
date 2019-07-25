#pragma warning( disable : 4996)
#include "Perceptron.h"

static const char INPUT_PATH[] = "C:\\Users\\cudauser\\Desktop\\Parallel-Binary-Classification-Perceptron\\dataset.txt";
//change to input.txt
static const char OUTPUT_PATH[] = "C:\\Users\\cudauser\\Desktop\\Parallel-Binary-Classification-Perceptron\\output.txt";
int main(int argc, char *argv[])
{
	
	//cudaError_t cudaStatus;
	//time_t t;

	int rank,size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	int N, K, LIMIT;
	double alpha_zero, alpha_max, QC;
	double* W = 0;
	Point* points = 0;
	
	Perceptron_readDataset(INPUT_PATH, rank, MPI_COMM_WORLD,&N, &K, &alpha_zero, &alpha_max, &LIMIT, &QC, &points);
	//printPointArray(points, N, K,rank);
	if (size < 2)
	{
		printf("N=%d K=%d alpha zero = %f alpha_max = %f LIMIT=%d QC = %f\n", N, K, alpha_zero, alpha_max, LIMIT, QC);
		run_perceptron_sequential(OUTPUT_PATH, N, K, alpha_zero, alpha_max, LIMIT, QC, points, W);
	}
	else
	{
		if (rank == MASTER)
			printf("Running in parallel with %d hosts.\n", size);
		run_perceptron_parallel(OUTPUT_PATH, rank, size,MPI_COMM_WORLD, N, K, alpha_zero, alpha_max, LIMIT, QC, points, W);
	}
	freePointArray(&points, N);
	MPI_Finalize();
	return 0;
	
}

