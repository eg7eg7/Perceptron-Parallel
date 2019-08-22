
#include "Perceptron.h"
#include "cudaKernel.h"
#include <stdlib.h>

Alpha* alpha_array;
int alpha_array_size;

int isInteger(char* string, int* the_num) {
	int num;

	num = atoi(string);

	if (num == 0 && string[0] != '0')
		return 0;
	*the_num = num;
	return 1;
}
int isDouble(char* string, double* the_num) {
	double num;

	num = (double)atof(string);

	if (num == 0 && string[0] != '0')
		return 0;
	*the_num = num;
	return 1;
}
void fileReadFailure() {
	printf("Failed to open file.\n");
	exit(EXIT_FAILURE);
}
void fileReadFailure(char* error) {
	printf("%s\n", error);
	fileReadFailure();
}
void perceptron_read_dataset(const char* path, int rank, MPI_Comm comm, int* N, int* K, double* alpha_zero, double* alpha_max, int* LIMIT, double* QC, Point** point_array, Point** device_point_array) {
	const int line_size = 1000;
	int set;
	char delim[] = " ";
	char* line;
	FILE *file;
	char *token;
	Point* temp_point_array;
	double* x_point_array;
	double* dev_x_point_array;

	if (rank == MASTER)
	{
		line = (char*)malloc(line_size);
		fopen_s(&file, path, "r");
		if (file == NULL)
			fileReadFailure("Could not find or open file.\n");
		//reading first line of the dataset file
		if (fgets(line, line_size, file) != NULL)
		{
			//reading N
			token = strtok(line, delim);
			if (token == NULL)
				fileReadFailure("could not token N");
			if (!isInteger(token, N))
				fileReadFailure("wrong token - N");

			//reading K
			token = strtok(NULL, delim);
			if (token == NULL)
				fileReadFailure("could not token K");
			if (!isInteger(token, K))
				fileReadFailure("wrong token - K");

			//reading alpha_zero
			token = strtok(NULL, delim);
			if (token == NULL)
				fileReadFailure();
			if (!isDouble(token, alpha_zero))
				fileReadFailure();

			//reading alpha_max
			token = strtok(NULL, delim);
			if (token == NULL)
				fileReadFailure();
			if (!isDouble(token, alpha_max))
				fileReadFailure();

			//reading LIMIT
			token = strtok(NULL, delim);
			if (token == NULL)
				fileReadFailure();
			if (!isInteger(token, LIMIT))
				fileReadFailure();

			//reading QC
			token = strtok(NULL, delim);
			if (token == NULL)
				fileReadFailure();
			if (!isDouble(token, QC))
				fileReadFailure();

		}
	}

	MPI_Bcast(N, 1, MPI_INT, MASTER, comm);
	MPI_Bcast(K, 1, MPI_INT, MASTER, comm);
	MPI_Bcast(alpha_zero, 1, MPI_DOUBLE, MASTER, comm);
	MPI_Bcast(alpha_max, 1, MPI_DOUBLE, MASTER, comm);
	MPI_Bcast(LIMIT, 1, MPI_INT, MASTER, comm);
	MPI_Bcast(QC, 1, MPI_DOUBLE, MASTER, comm);
	init_point_array(point_array, *N, *K);

	int arr_size = (*K + 1);

	temp_point_array = (Point*)malloc(sizeof(Point)*(*N));
	x_point_array = (double*)malloc(sizeof(double)*arr_size*(*N));
	cuda_malloc_double_by_size(&dev_x_point_array, (*N)*arr_size);
	cuda_malloc_point_by_size(device_point_array, (*N));


	for (int i = 0; i < (*N); i++)
	{
		if (rank == MASTER) {
			if (fgets(line, line_size, file) != NULL)
				if (line == NULL)
					fileReadFailure("line is NULL");
			token = strtok(line, delim);

			for (int j = 0; j < (*K) && token != NULL; j++)
			{
				isDouble(token, &(((*point_array)[i]).x[j]));
				token = strtok(NULL, delim);
			}
			((*point_array)[i]).x[*K] = 1;
			isInteger(token, &set);
			if (set == SET_A)
				(*point_array)[i].set = SET_A;
			else
				(*point_array)[i].set = SET_B;
		}
		MPI_Bcast((*point_array)[i].x, arr_size, MPI_DOUBLE, MASTER, comm);
		MPI_Bcast(&(*point_array)[i].set, 1, MPI_INT, MASTER, comm);

		copy_vector(&x_point_array[i*arr_size], (*point_array)[i].x, arr_size);
		temp_point_array[i].x = dev_x_point_array + i*arr_size;
		temp_point_array[i].set = (*point_array)[i].set;


	}
	if (rank == MASTER)
	{
		free(line);
		fclose(file);
	}

	memcpyDoubleArrayToDevice(&dev_x_point_array, &x_point_array, (*N)*arr_size);
	memcpyPointArrayToDevice(device_point_array, &temp_point_array, (*N));
	free(x_point_array);
	free(temp_point_array);

}
double f(double* x, double* W, int dim) {
	return mult_vector_with_vector(x, W, dim + 1);
}
int init_W(double** W, int K) {
	*W = (double*)malloc((K + 1) * sizeof(double));
	if (*W == NULL)
		return 0;
	return 1;
}
void init_point_array(Point** points, int N, int K) {
	if (N <= 0 || K <= 0)
	{
		printf("invalid parameters N=%d and K=%d\n", N, K);
		return;
	}
	*points = (Point*)malloc(sizeof(Point)*(N));
#pragma omp parallel for
	for (int i = 0; i < N; i++)
	{
		(*points)[i].x = (double*)malloc(sizeof(double)*(K + 1));
	}
}

void freePointArray(Point** points, Point** dev_points, int size) {
#pragma omp parallel for
	for (int i = 0; i < size; i++)
	{
		free((*points)[i].x);
	}
	free(*points);

	if (*dev_points != 0)
		free_cuda_point_array(dev_points);
}

void printPointArray(Point* points, int size, int dim, int rank) {
	for (int i = 0; i < size; i++) {
		printf("rank %d - point %d: ", rank, i + 1);
		for (int j = 0; j < dim; j++)
			printf("%.2f ", points[i].x[j]);
		printf("set %d\n", points[i].set);
	}
}

double get_quality(Point* points, double* W, int N, int K) {
	int N_mis = 0;
	double val;
	for (int i = 0; i < N; i++) {
		val = f(points[i].x, W, K);
		if (sign(val) != points[i].set)
		{
			N_mis += 1;
		}
	}
	return (double)((double)N_mis) / N;
}

int sign(double a)
{
	if (a >= 0)
		return SET_A;
	else
		return SET_B;
}
void adjustW(double* W, double* temp_result, int dim, double* p_xi, double f_p_scalar, double alpha) {
	mult_scalar_with_vector(p_xi, dim + 1, alpha*(-sign(f_p_scalar)), temp_result);
	add_vector_to_vector(W, temp_result, dim + 1, W);
}
void zero_W(double* W, int K) {
	mult_scalar_with_vector(W, K + 1, 0, W);
}
void run_perceptron_sequential(const char* output_path, int N, int K, double alpha_zero, double alpha_max, int LIMIT, double QC, Point* points) {
	double  t1, t2, *W, *temp_result, alpha, current_q = MAX_QC;
	int fault_flag = FAULT;

	init_W(&temp_result, K);
	t1 = omp_get_wtime();
	if (!init_W(&W, K))
	{
		printf("malloc assignment error");
		return;
	}
	//alpha=alpha zero  1, alpha=alpha+alpha_zero 9
	for (alpha = alpha_zero; alpha <= alpha_max; alpha += alpha_zero)
	{
		//2
		zero_W(W, K);
		//5 - loop through 3 and 4 til all points are properly classified or limit reached

		check_points_and_adjustW(points, W, temp_result, N, K, LIMIT, alpha);

		//6 find q
		current_q = get_quality(points, W, N, K);
		//7+8
		if (current_q <= QC)
			break;
	}
	t2 = omp_get_wtime();
	print_perceptron_output(output_path, W, K, alpha, current_q, QC, t2 - t1);
	free(W);
	free(temp_result);
}
void print_perceptron_output(const char* path, double* W, int K, double alpha, double q, double QC, double time) {


	FILE* file = NULL;
	fopen_s(&file, path, "w");

	if (q > QC)
	{
		printf("Alpha is not found\n");
		fprintf(file, "Alpha is not found");
	}
	else
	{
		printf("Alpha minimum = %f q=%f\n", alpha, q);
		print_arr(W, K + 1);
		fprintf(file, "Alpha minimum = %f q=%f\n", alpha, q);
		for (int i = 0; i <= K; i++)
			fprintf(file, "%f\n", W[i]);
	}
	printf("\nTotal time - %f seconds \n", time);
	fclose(file);
}

void free_alpha_array() {
#pragma omp parallel for
	for (int i = 0; i < alpha_array_size; i++)
		free(alpha_array[i].W);
	free(alpha_array);
}
void init_alpha_array(double alpha_max, double alpha_zero, int dim) {
	alpha_array_size = (int)floor(alpha_max / alpha_zero);

	alpha_array = (Alpha*)malloc(sizeof(Alpha)*	alpha_array_size);

#pragma omp parallel for
	for (int i = 0; i < alpha_array_size; i++)
	{
		alpha_array[i].q = Q_NOT_CHECKED;
		alpha_array[i].W = (double*)malloc(sizeof(double)*dim);
		alpha_array[i].value = (i + 1)*alpha_zero;
	}

}
void send_first_alphas_to_world(const double alpha_max, const double alpha_zero, double& alpha, const int world_size, int& num_workers, MPI_Comm comm)
{
	//Send first alpha values to hosts
#pragma omp parallel for
	for (int dst = 1; dst < world_size; dst++)
	{
		if (alpha <= alpha_max) {
			MPI_Send(&alpha, 1, MPI_DOUBLE, dst, START_TASK_TAG, comm);
#pragma omp critical
			{
				num_workers++;
				alpha += alpha_zero;
			}
		}

	}
}
void send_finish_tag_to_world(int world_size, MPI_Comm comm)
{
	double a;
#pragma omp parallel for
	for (int dst = 1; dst < world_size; dst++)
		MPI_Send(&a, 1, MPI_DOUBLE, dst, FINISH_PROCESS_TAG, comm);
}
void pack_buffer(char* buffer, double& alpha, double& q, double* W, int W_size, const MPI_Comm comm)
{
	int position = 0;
	MPI_Pack(&alpha, 1, MPI_DOUBLE, buffer, BUFFER_SIZE, &position, comm);
	MPI_Pack(&q, 1, MPI_DOUBLE, buffer, BUFFER_SIZE, &position, comm);
	MPI_Pack(W, W_size, MPI_DOUBLE, buffer, BUFFER_SIZE, &position, comm);
}
void unpack_buffer(char* buffer, double& alpha, double& q, double* W, int W_size, const MPI_Comm comm)
{
	int position = 0;
	MPI_Unpack(buffer, BUFFER_SIZE, &position, &alpha, 1, MPI_DOUBLE, comm);
	MPI_Unpack(buffer, BUFFER_SIZE, &position, &q, 1, MPI_DOUBLE, comm);
	MPI_Unpack(buffer, BUFFER_SIZE, &position, W, W_size, MPI_DOUBLE, comm);
}

void sendNextAlpha(double& alpha, const double alpha_max, const double alpha_zero, int dest, int& num_workers, MPI_Comm comm)
{
	MPI_Send(&alpha, 1, MPI_DOUBLE, dest, START_TASK_TAG, comm);
	num_workers++;
	alpha += alpha_zero;
}
<<<<<<< HEAD
<<<<<<< HEAD
int send_alpha_to_second_process(omp_lock_t& lock, int& PROCESS_2_STATUS_SHARED, double& alpha_2, double& alpha, double& alpha_zero, double* W, int K)
=======
int send_alpha_to_second_process(omp_lock_t& lock, int& PROCESS_2_STATUS_SHARED, double& alpha_2, double& alpha, const double& alpha_zero)
>>>>>>> parent of c00690c... reliable
=======
int send_alpha_to_second_process(omp_lock_t& lock, int& PROCESS_2_STATUS_SHARED, double& alpha_2, double& alpha, const double& alpha_zero, double* W, const int K)
>>>>>>> parent of 10f9ebb... refactoring and comments
{
	int status;
	omp_set_lock(&lock);
	if ((status = PROCESS_2_STATUS_SHARED) == PROCESS_WAITING)
	{
		alpha_2 = alpha += alpha_zero;
		status = PROCESS_2_STATUS_SHARED = PROCESS_BUSY;
	}
	omp_unset_lock(&lock);
	return status;
}
<<<<<<< HEAD

int get_value_thread_safe(omp_lock_t& lock, int& var)
{
	int val;
	omp_set_lock(&lock);
	val = var;
	omp_unset_lock(&lock);
	return val;
}
<<<<<<< HEAD
void master_dynamic_alpha_sending(int N, int K, double alpha_zero, double alpha_max, int LIMIT, double QC, MPI_Comm comm, int world_size, char* buffer, const char* output_path, Point* points, Point* points_device)
=======
void master_dynamic_alpha_sending(const int N, const int K, const double alpha_zero, const double alpha_max, const int LIMIT, const double QC, MPI_Comm comm, int world_size, char* buffer, const char* output_path, Point* points, Point* points_device)
>>>>>>> parent of c00690c... reliable
=======
void master_dynamic_alpha_sending(const int N, const int K, const double alpha_zero, const double alpha_max, const int LIMIT, const double QC, MPI_Comm comm, int world_size, char* buffer, const char* output_path, Point* points, Point* points_device)
>>>>>>> parent of 10f9ebb... refactoring and comments
{
	MPI_Status status;
	double alpha, alpha_2, q_2, returned_alpha, returned_q, *W, *W_2, *temp_result, t1, t2;
	int num_workers = 0, alpha_found_state = ALPHA_NOT_FOUND, data_src, RECEIVED_SOLUTION_FLAG = NO_SOLUTION;
	init_W(&W, K);
	init_W(&W_2, K);
	init_W(&temp_result, K);
	init_alpha_array(alpha_max, alpha_zero, K + 1);
	alpha = alpha_zero;
	int PROCESS_2_STATUS_SHARED = PROCESS_WAITING;
	int PROCESS_2_STATUS_PRIVATE = PROCESS_WAITING;
	omp_lock_t lock;
	omp_init_lock(&lock);
#pragma omp parallel num_threads(2) shared(lock,PROCESS_2_STATUS_SHARED, q_2,alpha_2,W_2) private(PROCESS_2_STATUS_PRIVATE)
	{
		if (omp_get_thread_num() == 0)
		{
			t1 = omp_get_wtime();
			PROCESS_2_STATUS_PRIVATE = send_alpha_to_second_process(lock, PROCESS_2_STATUS_SHARED, alpha_2, alpha, alpha_zero);
			send_first_alphas_to_world(alpha_max, alpha_zero, alpha, world_size, num_workers, comm);

			//send new alphas to hosts that finish
			while (num_workers > 0 || PROCESS_2_STATUS_SHARED == PROCESS_BUSY)
			{
				omp_set_lock(&lock);
				PROCESS_2_STATUS_PRIVATE = PROCESS_2_STATUS_SHARED;

				if (PROCESS_2_STATUS_PRIVATE != PROCESS_HAS_SOLUTION)
					omp_unset_lock(&lock);
				if (PROCESS_2_STATUS_PRIVATE == PROCESS_HAS_SOLUTION)
				{
					returned_alpha = alpha_2;
					returned_q = q_2;
					copy_vector(W, W_2, K + 1);
					PROCESS_2_STATUS_PRIVATE = PROCESS_2_STATUS_SHARED = PROCESS_WAITING;
					omp_unset_lock(&lock);
					data_src = MASTER;
					RECEIVED_SOLUTION_FLAG = 1;
				}
				else if (num_workers > 0)
				{
					MPI_Recv(buffer, BUFFER_SIZE, MPI_PACKED, MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &status);
					data_src = status.MPI_SOURCE;
					if (alpha_found_state != ALPHA_FOUND)
						unpack_buffer(buffer, returned_alpha, returned_q, W, K + 1, comm);
					--num_workers;
					RECEIVED_SOLUTION_FLAG = 1;
				}
				if (alpha_found_state != ALPHA_FOUND && RECEIVED_SOLUTION_FLAG == HAVE_SOLUTION)
				{
					printf("received alpha %f, q=%f solution from %d W[0] is %f\n", returned_alpha, returned_q, data_src, W[0]);

					alpha_found_state = check_lowest_alpha(&returned_alpha, &returned_q, QC, W, K + 1);
					if (alpha_found_state == ALPHA_FOUND)
					{
						// not doing break, need to wait for hosts to send their calculations.
						t2 = omp_get_wtime();
						printf("Alpha found by rank %d\n", data_src);
						print_perceptron_output(output_path, W, K, returned_alpha, returned_q, QC, t2 - t1);

					}
					RECEIVED_SOLUTION_FLAG = NO_SOLUTION;
				}
				//send new alpha
				if (alpha_found_state == ALPHA_NOT_FOUND && alpha <= alpha_max) {
					if (data_src == MASTER)
						send_alpha_to_second_process(lock, PROCESS_2_STATUS_SHARED, alpha_2, alpha, alpha_zero);
					else if (world_size > 1)
						sendNextAlpha(alpha, alpha_max, alpha_zero, status.MPI_SOURCE, num_workers, comm);
				}
			}

			if (alpha_found_state != ALPHA_FOUND)
			{
				t2 = omp_get_wtime();
				print_perceptron_output(output_path, W, K, returned_alpha, returned_q, QC, t2 - t1);
			}
			//send hosts the finish tag
			send_finish_tag_to_world(world_size, comm);
			omp_set_lock(&lock);
			PROCESS_2_STATUS_SHARED = FINISH_PROCESS;
			omp_unset_lock(&lock);
		}
		else // PROCESS 2, aid with alpha
		{
			int set_solution;
			while (PROCESS_2_STATUS_SHARED != FINISH_PROCESS)
			{
				set_solution = 0;
				if (PROCESS_2_STATUS_SHARED == PROCESS_BUSY)
				{
<<<<<<< HEAD
					omp_set_lock(&var_lock);
<<<<<<< HEAD
=======
					zero_W(W_2, K);
>>>>>>> parent of c00690c... reliable
=======
					printf("MASTER receive alpha %f\n", alpha_2);
>>>>>>> parent of 10f9ebb... refactoring and comments
					check_points_and_adjustW(points, W_2, temp_result, N, K, LIMIT, alpha_2);
					get_quality_with_GPU(points_device, W_2, N, K, &q_2);
					omp_set_lock(&lock);
					PROCESS_2_STATUS_PRIVATE = PROCESS_2_STATUS_SHARED = PROCESS_HAS_SOLUTION;
					set_solution = 1;
					omp_unset_lock(&lock);
				}
				if (!set_solution)
<<<<<<< HEAD
<<<<<<< HEAD
=======
				{
>>>>>>> parent of 10f9ebb... refactoring and comments
					PROCESS_2_STATUS_PRIVATE = get_value_thread_safe(lock, PROCESS_2_STATUS_SHARED);
				}
			}
<<<<<<< HEAD
			cuda_malloc_and_free_pointers_from_quality_function(0, 0, 0, 0, 0, 0, FREE_MALLOC_FLAG);
=======
				{
					omp_set_lock(&lock);
					PROCESS_2_STATUS_PRIVATE = PROCESS_2_STATUS_SHARED;
					omp_unset_lock(&lock);
				}
			}
>>>>>>> parent of c00690c... reliable
=======
			cuda_malloc_and_free_pointers_from_quality_function(0, 0, 0, 0, 0, 0, 0, FREE_MALLOC_FLAG);
>>>>>>> parent of 10f9ebb... refactoring and comments
		}
	}
	free_alpha_array();
	free(W);
	free(W_2);
	free(temp_result);
	omp_destroy_lock(&lock);
}



void run_perceptron_parallel(const char* output_path, int rank, int world_size, MPI_Comm comm, int N, int K, double alpha_zero, double alpha_max, int LIMIT, double QC, Point* points, Point* points_device)
{
	char buffer[BUFFER_SIZE];

	if (rank == MASTER) {
		master_dynamic_alpha_sending(N, K, alpha_zero, alpha_max, LIMIT, QC, comm, world_size, buffer, output_path, points, points_device);
	}
	else //host is not MASTER
	{
<<<<<<< HEAD
		get_alphas_and_calc_q(rank, buffer, N, K, LIMIT, points, points_device, comm);
<<<<<<< HEAD
		cuda_malloc_and_free_pointers_from_quality_function(0, 0, 0, 0, 0, 0, FREE_MALLOC_FLAG);
=======
		get_alphas_and_calc_q(buffer, N, K, LIMIT, points, points_device, comm);
		cuda_malloc_and_free_pointers_from_quality_function(0, 0, 0, 0, 0, 0, 0, FREE_MALLOC_FLAG);
>>>>>>> parent of c00690c... reliable
=======
		cuda_malloc_and_free_pointers_from_quality_function(0, 0, 0, 0, 0, 0, 0, FREE_MALLOC_FLAG);
>>>>>>> parent of 10f9ebb... refactoring and comments
	}
}
void get_alphas_and_calc_q(char* buffer, int N, int K, int LIMIT, Point* points, Point* points_device, MPI_Comm comm) {
	double alpha, q, *W, *temp_result;
	MPI_Status status;
	init_W(&W, K);
	init_W(&temp_result, K);

	while (1) {
		MPI_Recv(&alpha, 1, MPI_DOUBLE, MASTER, MPI_ANY_TAG, comm, &status);
		if (status.MPI_TAG == FINISH_PROCESS_TAG)
			break;
		zero_W(W, K);
		check_points_and_adjustW(points, W, temp_result, N, K, LIMIT, alpha);
		get_quality_with_GPU(points_device, W, N, K, &q);
		printf("rank %d receive alpha %f and return %f\n", rank, alpha, q);
		pack_buffer(buffer, alpha, q, W, K + 1, comm);
		MPI_Send(buffer, BUFFER_SIZE, MPI_PACKED, MASTER, FINISH_TASK_TAG, comm);
	}
	free(W);
	free(temp_result);
}
void check_points_and_adjustW(Point *points, double *W, double *temp_arr, int N, int K, int LIMIT, double alpha)
{

	int fault_flag = FAULT;
	double val;

	for (int loop = 0; loop < LIMIT && fault_flag == FAULT; loop++)
	{
		fault_flag = NO_FAULT;
		//3
		for (int i = 0; i < N; i++) {
			val = f(points[i].x, W, K);
			if (sign(val) != points[i].set)
			{
				fault_flag = FAULT;
				//4
				adjustW(W, temp_arr, K, points[i].x, val, alpha);
				break;
			}
		}
	}
}

int check_lowest_alpha(double* returned_alpha, double* returned_q, double QC, double* W, int dim) {
	static int alpha_array_state = ALPHA_NOT_FOUND;
	static int min_index = 0;

	if (*returned_q <= QC)
		alpha_array_state = ALPHA_POTENTIALLY_FOUND;
	int index = (int)(((*returned_alpha) / alpha_array[0].value) - 1);
	alpha_array[index].q = *returned_q;
<<<<<<< HEAD
	copy_vector(alpha_array[index].W, W, dim);

	//order really matters!

	for (int i = min_index; i < alpha_array_size; i++)
=======

	copy_vector(alpha_array[index].W, W, dim);
	//DO NOT USE OpenMP here - order really matters!
	for (int i = min_index; i < alpha_array_size && alpha_array_state == ALPHA_POTENTIALLY_FOUND; i++)
>>>>>>> parent of c00690c... reliable
	{

		printf("num workers %d\n", omp_get_num_threads());
		if (alpha_array[index].q == Q_NOT_CHECKED)
			return alpha_array_state;

		if (alpha_array[index].q <= QC)
		{
			*returned_alpha = alpha_array[index].value;
			*returned_q = alpha_array[index].q;
			copy_vector(W, alpha_array[index].W, dim);
			alpha_array_state = ALPHA_FOUND;
			printf("my index=%d ,alpha %f is found correct - return %d\n", index, *returned_alpha, alpha_array_state);
			return alpha_array_state;
		}
		min_index = i;
<<<<<<< HEAD

<<<<<<< HEAD
=======
>>>>>>> parent of c00690c... reliable
	}
=======
		}
>>>>>>> parent of 10f9ebb... refactoring and comments
	return alpha_array_state;
}


void print_arr(double* W, int dim) {
	for (int i = 0; i < dim; i++)
		printf("%f\n", W[i]);
}

