
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
	/* Read order : N_K_ALPHA_ZERO_ALPHA_MAX_LIMIT_QC
				N1 -K1 K2 K3 ... G
				N2 -K1 K2 K3 ... G
				N3 -K1 K2 K3 ... G
				 .
				 .
				 .
				where G is 1 or -1					*/

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
			fileReadFailure("Could not find or open file. \n");
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

	memcpy_double_array_to_device(&dev_x_point_array, &x_point_array, (*N)*arr_size);
	memcpy_point_array_to_device(device_point_array, &temp_point_array, (*N));
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
	double *W, *temp_result, alpha, current_q = MAX_QC,t1,t2;
	int fault_flag = FAULT;
	t1 = omp_get_wtime();
	init_W(&temp_result, K);
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
	print_perceptron_output(output_path, W, K, alpha, current_q, QC,t2-t1);
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
		print_array(W, K + 1);
		fprintf(file, "Alpha minimum = %f q=%f\n", alpha, q);
		for (int i = 0; i <= K; i++)
			fprintf(file, "%f\n", W[i]);
	}
#ifdef _DEBUG_PRINTS
	fprintf(file, "DEBUG : Time = %f\n", time);
	printf("DEBUG : Time = %f\n", time);
#endif
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
void send_first_alphas_to_world(double alpha_max, double alpha_zero, double& alpha, int world_size, int& num_workers, MPI_Comm comm)
{
	//Send first alpha values to hosts
#pragma omp parallel for
	for (int dst = 1; dst < world_size; ++dst)
	{
		if (alpha <= alpha_max) {
			MPI_Send(&alpha, 1, MPI_DOUBLE, dst, START_TASK_TAG, comm);
#pragma omp critical
			{
				++num_workers;
				alpha += alpha_zero;
			}
		}
	}
}
void send_finish_tag_to_world(int world_size, MPI_Comm comm)
{
	double junk_val;
#pragma omp parallel for
	for (int dst = 1; dst < world_size; dst++)
		MPI_Send(&junk_val, 1, MPI_DOUBLE, dst, FINISH_PROCESS_TAG, comm);
}
void pack_buffer(char* buffer, double& alpha, double& q, double* W, int W_size, MPI_Comm comm)
{
	int position = 0;
	MPI_Pack(&alpha, 1, MPI_DOUBLE, buffer, BUFFER_SIZE, &position, comm);
	MPI_Pack(&q, 1, MPI_DOUBLE, buffer, BUFFER_SIZE, &position, comm);
	MPI_Pack(W, W_size, MPI_DOUBLE, buffer, BUFFER_SIZE, &position, comm);
}
void unpack_buffer(char* buffer, double& alpha, double& q, double* W, int W_size, MPI_Comm comm)
{
	int position = 0;
	MPI_Unpack(buffer, BUFFER_SIZE, &position, &alpha, 1, MPI_DOUBLE, comm);
	MPI_Unpack(buffer, BUFFER_SIZE, &position, &q, 1, MPI_DOUBLE, comm);
	MPI_Unpack(buffer, BUFFER_SIZE, &position, W, W_size, MPI_DOUBLE, comm);
}

void sendNextAlpha(double& alpha, double alpha_max, double alpha_zero, int dest, int& num_workers, MPI_Comm comm)
{
	MPI_Send(&alpha, 1, MPI_DOUBLE, dest, START_TASK_TAG, comm);
		++num_workers;
		alpha += alpha_zero;
}
int send_alpha_to_second_process(omp_lock_t& lock, omp_lock_t& var_lock, int& PROCESS_2_STATUS_SHARED, double& alpha_2, double& alpha, double& alpha_zero, double* W, int K)
{
	int status;
	omp_set_lock(&lock);
	if ((status = PROCESS_2_STATUS_SHARED) == CORE_WAITING)
	{
		omp_set_lock(&var_lock);
		alpha_2 = alpha;
		zero_W(W, K);
		omp_unset_lock(&var_lock);
		status = PROCESS_2_STATUS_SHARED = CORE_BUSY;
		alpha += alpha_zero;
	}
	omp_unset_lock(&lock);
	return status;
}

int get_value_thread_safe(omp_lock_t& lock, int& var)
{
	int val;
	omp_set_lock(&lock);
	val = var;
	omp_unset_lock(&lock);
	return val;
}

void master_dynamic_alpha_sending(int N, int K, double alpha_zero, double alpha_max, int LIMIT, double QC, MPI_Comm comm, int world_size, char* buffer, const char* output_path, Point* points, Point* points_device)
{
	MPI_Status status;
	double alpha, alpha_2, q_2, returned_alpha, returned_q, *W, *W_2, *temp_result, t1, t2,t3,t4;
	int num_workers = 0, alpha_found_state = ALPHA_NOT_FOUND, data_src, RECEIVED_SOLUTION_FLAG = NO_SOLUTION;
	init_W(&W, K);
	init_W(&W_2, K);
	init_W(&temp_result, K);
	init_alpha_array(alpha_max, alpha_zero, K + 1);
	alpha = alpha_zero;
	int CORE_2_STATUS_SHARED = CORE_WAITING;
	int CORE_2_STATUS_PRIVATE = CORE_WAITING;
	omp_lock_t lock;
	omp_lock_t var_lock;
	omp_init_lock(&var_lock);
	omp_init_lock(&lock);
	t1 = omp_get_wtime();
	/*Master host has two processes working,
	1. for dynamic scheduling for alphas
	2. Calculating for received alpha - helping with the load
	*/
#pragma omp parallel num_threads(2) shared(lock,var_lock,CORE_2_STATUS_SHARED, q_2,alpha_2,W_2) private(CORE_2_STATUS_PRIVATE,t3,t4)
	{
		if (omp_get_thread_num() == 0)
		{
			send_first_alphas_to_world(alpha_max, alpha_zero, alpha, world_size, num_workers, comm);
			CORE_2_STATUS_PRIVATE = send_alpha_to_second_process(lock,var_lock, CORE_2_STATUS_SHARED, alpha_2, alpha, alpha_zero, W_2, K);
			//send new alphas to hosts that finish
			while (num_workers > 0 || CORE_2_STATUS_PRIVATE == CORE_BUSY)
			{
				CORE_2_STATUS_PRIVATE = get_value_thread_safe(lock, CORE_2_STATUS_SHARED);
				if (CORE_2_STATUS_PRIVATE == CORE_HAS_SOLUTION)
				{
					//Receive solution from other process
					t3 = omp_get_wtime();
					omp_set_lock(&var_lock);
					returned_alpha = alpha_2;
					returned_q = q_2;
					copy_vector(W, W_2, K + 1);
					omp_unset_lock(&var_lock);
					
					omp_set_lock(&lock);
					CORE_2_STATUS_PRIVATE = CORE_2_STATUS_SHARED = CORE_WAITING;
					omp_unset_lock(&lock);
					data_src = MASTER;
					RECEIVED_SOLUTION_FLAG = 1;
					t4 = omp_get_wtime();
#ifdef _DEBUG_PRINTS
					printf("time to receive locks and get vars from process 2 is %f\n", t4 - t3);
#endif
				}
				else if (num_workers > 0)
				{
					//Receive solution from other host
					MPI_Recv(buffer, BUFFER_SIZE, MPI_PACKED, MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &status);
					data_src = status.MPI_SOURCE;
					unpack_buffer(buffer, returned_alpha, returned_q, W, K + 1, comm);
					--num_workers;
					RECEIVED_SOLUTION_FLAG = 1;

				}

				if (alpha_found_state != ALPHA_FOUND && RECEIVED_SOLUTION_FLAG == HAVE_SOLUTION)
				{
					t3 = omp_get_wtime();
					alpha_found_state = check_lowest_alpha(&returned_alpha, &returned_q, QC, W, K + 1);
					t4 = omp_get_wtime();
#ifdef _DEBUG_PRINTS
					printf("time to check lowest alpha %f\n", t4 - t3);
#endif
					if (alpha_found_state == ALPHA_FOUND)
					{
						// not doing break, need to wait for hosts to send their calculations.
						t2 = omp_get_wtime();
						print_perceptron_output(output_path, W, K, returned_alpha, returned_q, QC,t2-t1);
					}
				}
				//send new alpha
				if (alpha_found_state == ALPHA_NOT_FOUND && alpha <= alpha_max && RECEIVED_SOLUTION_FLAG== HAVE_SOLUTION) {
					if (data_src == MASTER)
						send_alpha_to_second_process(lock,var_lock, CORE_2_STATUS_SHARED, alpha_2, alpha, alpha_zero, W_2, K);
					else if (world_size > 1)
						sendNextAlpha(alpha, alpha_max, alpha_zero, status.MPI_SOURCE, num_workers, comm);
				}
				//reset flag
				RECEIVED_SOLUTION_FLAG = NO_SOLUTION;
			}
			if (alpha_found_state != ALPHA_FOUND)
			{
				t2 = omp_get_wtime();
				print_perceptron_output(output_path, W, K, returned_alpha, returned_q, QC,t2-t1);
			}
			//send hosts the finish tag
			printf("\nTotal parallel time - %f seconds \n", t2 - t1);
			send_finish_tag_to_world(world_size, comm);
			omp_set_lock(&lock);
			CORE_2_STATUS_SHARED = FINISH_CORE;
			omp_unset_lock(&lock);
		}
		else // CORE 2, aid with alpha
		{
			int set_solution;
			CORE_2_STATUS_PRIVATE = get_value_thread_safe(lock, CORE_2_STATUS_SHARED);
			while (CORE_2_STATUS_PRIVATE != FINISH_CORE)
			{
				set_solution = 0;
				if (CORE_2_STATUS_PRIVATE == CORE_BUSY)
				{
					t3 = omp_get_wtime();
					omp_set_lock(&var_lock);
					check_points_and_adjustW(points, W_2, temp_result, N, K, LIMIT, alpha_2);
					get_quality_with_GPU(points_device, W_2, N, K, &q_2);
					t4 = omp_get_wtime();
					omp_unset_lock(&var_lock);
#ifdef _DEBUG_PRINTS
					printf("core 2 receive alpha %f, return q %f and w0 %f w1 %f w2 %f - time %f\n", alpha_2, q_2, W_2[0], W_2[1], W_2[2],t4-t3);
#endif
					omp_set_lock(&lock);
					CORE_2_STATUS_PRIVATE = CORE_2_STATUS_SHARED = CORE_HAS_SOLUTION;
					set_solution = 1;
					omp_unset_lock(&lock);
				}
				if (!set_solution)
					CORE_2_STATUS_PRIVATE = get_value_thread_safe(lock, CORE_2_STATUS_SHARED);
			} //end while
			cuda_malloc_and_free_pointers_from_quality_function(0, 0, 0, 0, 0, 0, FREE_MALLOC_FLAG);
		} //end process 2 procedure
	} // end pragma omp

	free_alpha_array();
	free(W);
	free(W_2);
	free(temp_result);
	omp_destroy_lock(&lock);
	omp_destroy_lock(&var_lock);
}



void run_perceptron_parallel(const char* output_path, int rank, int world_size, MPI_Comm comm, int N, int K, double alpha_zero, double alpha_max, int LIMIT, double QC, Point* points, Point* points_device)
{
	char buffer[BUFFER_SIZE];

	if (rank == MASTER) {
		master_dynamic_alpha_sending(N, K, alpha_zero, alpha_max, LIMIT, QC, comm, world_size, buffer, output_path, points, points_device);
	}
	else //host is not MASTER
	{
		get_alphas_and_calc_q(rank, buffer, N, K, LIMIT, points, points_device, comm);
		cuda_malloc_and_free_pointers_from_quality_function(0, 0, 0, 0, 0, 0, FREE_MALLOC_FLAG);
	}
}
void get_alphas_and_calc_q(int rank, char* buffer, int N, int K, int LIMIT, Point* points, Point* points_device, MPI_Comm comm) {
	double alpha, q, *W, *temp_result,t1,t2,t3;
	MPI_Status status;
	init_W(&W, K);
	init_W(&temp_result, K);

	while (1) {
		MPI_Recv(&alpha, 1, MPI_DOUBLE, MASTER, MPI_ANY_TAG, comm, &status);
		if (status.MPI_TAG == FINISH_PROCESS_TAG)
			break;
		zero_W(W, K);
		t1 = omp_get_wtime();
		check_points_and_adjustW(points, W, temp_result, N, K, LIMIT, alpha);
		t2 = omp_get_wtime();
		get_quality_with_GPU(points_device, W, N, K, &q);
		t3 = omp_get_wtime();
#ifdef _DEBUG_PRINTS
		printf("Rank %d - receive alpha %f return q %f w0 %f w1 %f w2 %f time - %f (adjusting w %f, getting q %f)\n", rank, alpha, q,W[0],W[1],W[2],t3-t1,t2-t1,t3-t2);
#endif
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
	copy_vector(alpha_array[index].W, W, dim);

	//order really matters! - no omp
	for (int i = min_index; i < alpha_array_size; ++i)
	{
		if (alpha_array[i].q == Q_NOT_CHECKED)
		{
#ifdef _DEBUG_PRINTS
			printf("alpha %f checked, q is %f - invalid\n",alpha_array[i].value, alpha_array[i].q);
#endif
			return alpha_array_state;
		}

		if (alpha_array[i].q <= QC)
		{

			*returned_alpha = alpha_array[i].value;
			*returned_q = alpha_array[i].q;
			copy_vector(W, alpha_array[i].W, dim);
			alpha_array_state = ALPHA_FOUND;
			return alpha_array_state;
		}
		min_index = i;

	}
	return alpha_array_state;
}


void print_array(double* W, int dim) {
	for (int i = 0; i < dim; i++)
		printf("%f\n", W[i]);
}

