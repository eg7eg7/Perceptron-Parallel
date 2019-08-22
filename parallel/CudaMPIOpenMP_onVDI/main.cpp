#pragma warning( disable : 4996)
#include "Perceptron.h"
<<<<<<< HEAD
/* EDEN DUPONT 204808596 */
<<<<<<< HEAD
static const char INPUT_PATH[] = "input\\input.txt";
=======

static const char INPUT_PATH[] = "C:\\input.txt";
>>>>>>> parent of c00690c... reliable
=======
static const char INPUT_PATH[] = "C:\\input.txt";
>>>>>>> parent of 10f9ebb... refactoring and comments

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
<<<<<<< HEAD
<<<<<<< HEAD
=======
	omp_set_nested(true);
>>>>>>> parent of c00690c... reliable
=======
	//omp_set_nested(true);
>>>>>>> parent of 10f9ebb... refactoring and comments
	int num_threads = omp_get_max_threads();


	t1 = omp_get_wtime();
	perceptron_read_dataset(INPUT_PATH, rank, MPI_COMM_WORLD, &N, &K, &alpha_zero, &alpha_max, &LIMIT, &QC, &points, &dev_points);
	t2 = omp_get_wtime();
<<<<<<< HEAD
<<<<<<< HEAD
	printf("Read/receive data time - %f seconds, rank %d\n", t2 - t1, rank);

	if (size > 1 || num_threads > 1)
	{
		if (rank == MASTER)
			printf("\n\nBegin parallel Algorithm (%d Max Thread, %d World size)\n", num_threads, size);

=======
	
	double read_data_time = t2 - t1;
	printf("num threads %d", num_threads);
	if (size > 1 || num_threads > 1)
	{
>>>>>>> parent of c00690c... reliable
=======
	double read_data_time = t2 - t1;
	printf("Read/receive data time - %f seconds, rank %d\n", read_data_time, rank);
	
	if (size > 1 || num_threads > 1)
	{
		if(rank == MASTER)
			printf("\n\nBegin parallel Algorithm (%d Max Thread, %d World size)\n", num_threads,size);
		
>>>>>>> parent of 10f9ebb... refactoring and comments
		run_perceptron_parallel(OUTPUT_PATH, rank, size, MPI_COMM_WORLD, N, K, alpha_zero, alpha_max, LIMIT, QC, points, dev_points);
		
	}
		
	if (rank == MASTER)
	{
		t1 = omp_get_wtime();
<<<<<<< HEAD
<<<<<<< HEAD
		run_perceptron_sequential(OUTPUT_PATH, N, K, alpha_zero, alpha_max, LIMIT, QC, points);
=======
		//omp_set_num_threads(1);
		//	run_perceptron_sequential(OUTPUT_PATH, N, K, alpha_zero, alpha_max, LIMIT, QC, points);
>>>>>>> parent of c00690c... reliable
=======
		//	run_perceptron_sequential(OUTPUT_PATH, N, K, alpha_zero, alpha_max, LIMIT, QC, points);
>>>>>>> parent of 10f9ebb... refactoring and comments
		t2 = omp_get_wtime();
	}
<<<<<<< HEAD
<<<<<<< HEAD
	printf("\n\nEnd of program - Rank %d\n", rank);
=======
	printf("\n\nEnd of program.\nRank %d read data time - %f seconds\n", rank, read_data_time);
>>>>>>> parent of c00690c... reliable
=======
	printf("\n\nEnd of program - Rank %d\n\nEden Dupont 204808596 Afeka 2019\n", rank);
>>>>>>> parent of 10f9ebb... refactoring and comments
	freePointArray(&points, &dev_points, N);


	MPI_Finalize();
	return 0;

}

