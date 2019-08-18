#pragma warning( disable : 4996)
#include "Perceptron.h"

static const char INPUT_PATH[] = "C:\\input.txt";

static const char OUTPUT_PATH[] = "C:\\output.txt";

int main(int argc, char *argv[])
{
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	int N, K, LIMIT;
	double alpha_zero, alpha_max, QC, t1, t2;
	Point* dev_points = 0;
	Point* points = 0;
	
	if (size < 2)
	{
		omp_set_num_threads(1);
	}
	int max_threads = omp_get_max_threads();
	if(rank == 0)
		printf("\nRunning with max threads = %d\n", max_threads);
	t1 = omp_get_wtime();
	perceptron_read_dataset(INPUT_PATH, rank, MPI_COMM_WORLD, &N, &K, &alpha_zero, &alpha_max, &LIMIT, &QC, &points, &dev_points);
	t2 = omp_get_wtime();
	//printf("\nRank %d read data time - %f seconds\n", rank, t2 - t1);
	//printf("\nN=%d K=%d alpha zero = %f alpha_max = %f LIMIT=%d QC = %f\n", N, K, alpha_zero, alpha_max, LIMIT, QC);

	if (size < 2)
		run_perceptron_sequential(OUTPUT_PATH, N, K, alpha_zero, alpha_max, LIMIT, QC, points);
	else
		run_perceptron_parallel(OUTPUT_PATH, rank, size, MPI_COMM_WORLD, N, K, alpha_zero, alpha_max, LIMIT, QC, points,dev_points);
	
	freePointArray(&points,&dev_points, N);
	MPI_Finalize();
	return 0;

}

