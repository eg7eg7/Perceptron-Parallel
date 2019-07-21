
#include "Perceptron.h"
#include "cudaKernel.h"
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
void Perceptron_readDataset(const char* path, int rank, MPI_Comm comm, int* N, int* K, double* alpha_zero, double* alpha_max, int* LIMIT, double* QC, Point** point_array) {
	//MPI_Status status;
	const int line_size = 1000;
	char delim[] = " ";
	char* line;
	FILE *file;
	char *token;
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
	initPointArray(point_array, *N, *K);

	//if (rank == 1) { MPI_Finalize(); }
	int set;

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
		MPI_Bcast((*point_array)[i].x, (*K) + 1, MPI_DOUBLE, MASTER, comm);
		MPI_Bcast(&(*point_array)[i].set, 1, MPI_INT, MASTER, comm);
	}

	if (rank == MASTER)
	{
		free(line);
		fclose(file);
	}

}

double f(double* x, double* W, int dim) {
	return mult_vector_with_vector(x, W, dim + 1);
}

int init_W(double** W, int K) {
	//W is initialized to 0
	*W = (double*)malloc((K + 1) * sizeof(double));
	if (*W == NULL)
		return 0;
	return 1;
}

void initPointArray(Point** points, int N, int K) {
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

void freePointArray(Point** points, int size) {
#pragma omp parallel for
	for (int i = 0; i < size; i++)
	{
		free((*points)[i].x);
	}
	free(*points);
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
#pragma omp parallel for
	for (int i = 0; i < N; i++) {
		double val = f(points[i].x, W, K);
		if (sign(val) != points[i].set)
		{
#pragma omp critical
			{
				N_mis++;
			}
			
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
void adjustW(double* W, int dim, double* p_xi, double f_p_scalar, double alpha) {
	double *temp_result = (double*)malloc(sizeof(double)*(dim + 1));

	mult_scalar_with_vector(p_xi, dim + 1, alpha*(-sign(f_p_scalar)), temp_result);
	add_vector_to_vector(W, temp_result, dim + 1, W);

	free(temp_result);
}
void zero_W(double* W, int K) {
	for (int i = 0; i <= K; i++)
		W[i] = 0.0;
}
void run_perceptron_sequential(const char* output_path, int N, int K, double alpha_zero, double alpha_max, int LIMIT, double QC, Point* points, double* W) {
	double t1, t2;
	t1 = omp_get_wtime();
	double val;
	int fault_flag = FAULT;
	double alpha, current_q = MAX_QC, best_q = MAX_QC, best_alpha;
	int loop;
	int faulty_point;
	double* min_W;
	if (!init_W(&W, K))
	{
		printf("malloc assignment error");
		return;
	}

	init_W(&min_W, K);

	//alpha=alpha zero  1, alpha=alpha+alpha_zero 9
	for (alpha = alpha_zero; alpha <= alpha_max; alpha += alpha_zero)
	{
		//2
		zero_W(W, K);
		//5 - loop through 3 and 4 til all points are properly classified or limit reached
		for (loop = 0; loop < LIMIT && fault_flag == FAULT; loop++)
		{
			fault_flag = NO_FAULT;
			//3
			for (int i = 0; i < N; i++) {
				val = f(points[i].x, W, K);
				if (sign(val) != points[i].set)
				{
					fault_flag = FAULT;
					faulty_point = i;
					break;
				}
			}
			if (fault_flag == FAULT) {
				//4
				adjustW(W, K, points[faulty_point].x, val, alpha);
			}
		}
		//6 find q
		current_q = get_quality(points, W, N, K);
		//7+8
		if (current_q < best_q)
		{
			copy_vector(min_W, W, K + 1);
			best_q = current_q;
			best_alpha = alpha;
		}
		if (current_q <= QC)
			break;
	}
	if (alpha > alpha_max)
		printf("\tstop reason : alpha value exceeded. alpha max %f\n", alpha_max);
	if (current_q < QC)
		printf("\tstop reason : desired quality reached. q %f of QC=%f\n", current_q, QC);

	printf("q=%f,alpha=%f\n\nW = [", current_q, alpha);
	for (int i = 0; i < K + 1; i++) {
		printf("%f ", W[i]);
	}
	printf("]\n");
	printf("Best W is W= [");
	for (int i = 0; i < K + 1; i++) {
		printf("%f ", min_W[i]);
	}
	printf("] with q=%f,alpha=%f\n", best_q, best_alpha);

	t2 = omp_get_wtime();
	printf("Perceptron sequential time is %f\n", t2 - t1);
	printPerceptronOutput(output_path, W, K, alpha, current_q, QC);

	free(W);
	free(min_W);
}
void printPerceptronOutput(const char* path, double* W, int K, double alpha, double q, double QC) {
#ifdef PRINT
	if (q > QC)
		printf("Alpha is not found");
	else
	{
		printf("Alpha minimum = %f q=%f\n", alpha, q);
		print_arr(W, K + 1);

	}
#else 

	FILE* file;
	fopen_s(&file, path, "w");
	if (q > QC)
		fprintf(file, "Alpha is not found");
	else
	{
		fprintf(file, "Alpha minimum = %f q=%f\n", alpha, q);
		for (int i = 0; i <= K; i++)
			fprintf(file, "%f\n", W[i]);

	}

	fclose(file);
#endif

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
void run_perceptron_parallel(const char* output_path, int rank, int world_size, MPI_Comm comm, int N, int K, double alpha_zero, double alpha_max, int LIMIT, double QC, Point* points, double* W)
{
	if (rank == MASTER)
		init_alpha_array(alpha_max, alpha_zero, K + 1);
	int alpha_found = ALPHA_NOT_FOUND;
	int position;
	MPI_Status status;
	char buffer[BUFFER_SIZE];
	init_W(&W, K);
	double alpha;
	int printed_output = 0;
	double returned_alpha;
	double returned_q;
	if (rank == MASTER) {
		int num_workers = 0;
		alpha = alpha_zero;
//#pragma omp parallel for
				//TODO pragma ordered?
		for (int dst = 1; dst < world_size; dst++)
		{
			if (alpha <= alpha_max) {
				MPI_Send(&alpha, 1, MPI_DOUBLE, dst, START_TASK_TAG, comm);
				//#pragma omp critical
				//{
					num_workers++;
				//}
			}
			alpha += alpha_zero;
		}
		while (num_workers > 0)
		{
			position = 0;
			MPI_Recv(buffer, BUFFER_SIZE, MPI_PACKED, MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &status);
			num_workers--;
			if (alpha_found != ALPHA_FOUND)
			{
				MPI_Unpack(buffer, BUFFER_SIZE, &position, &returned_alpha, 1, MPI_DOUBLE, comm);
				MPI_Unpack(buffer, BUFFER_SIZE, &position, &returned_q, 1, MPI_DOUBLE, comm);
				MPI_Unpack(buffer, BUFFER_SIZE, &position, W, K + 1, MPI_DOUBLE, comm);
				alpha_found = check_lowest_alpha(&returned_alpha, &returned_q, QC, W, K + 1);
				if (alpha_found == ALPHA_FOUND)
					printf("Alpha found by rank %d\n", status.MPI_SOURCE);
			}
			
			if (alpha <= alpha_max && alpha_found == ALPHA_NOT_FOUND)
			{
				MPI_Send(&alpha, 1, MPI_DOUBLE, status.MPI_SOURCE, START_TASK_TAG, comm);
				num_workers++;
				alpha += alpha_zero;
			}

		}
		printPerceptronOutput(output_path, W, K, returned_alpha, returned_q, QC);
		//send to hosts finish tag
#pragma omp parallel for
		for (int dst = 1; dst < world_size; dst++)
			MPI_Send(&alpha, 1, MPI_DOUBLE, dst, FINISH_PROCESS_TAG, comm);

	}
	else //host is not MASTER
	{
		Point* dev_points;
		CopyPointsToDevice(points, &dev_points, N, K);
		double q;
		int position;
		while (1) {
			position = 0;
			MPI_Recv(&alpha, 1, MPI_DOUBLE, MASTER, MPI_ANY_TAG, comm, &status);
			if(status.MPI_TAG == START_TASK_TAG)
				printf("Rank %d received alpha %f from Master\n", rank, alpha);
			else
				printf("Rank %d finish\n", rank);
			if (status.MPI_TAG == FINISH_PROCESS_TAG)
				break;
			zero_W(W, K);
			get_quality_with_alpha_GPU(points, alpha, W, N, K, LIMIT);
			q = get_quality(points, W, N, K);
			MPI_Pack(&alpha, 1, MPI_DOUBLE, buffer, BUFFER_SIZE, &position, comm);
			MPI_Pack(&q, 1, MPI_DOUBLE, buffer, BUFFER_SIZE, &position, comm);
			MPI_Pack(W, K+1, MPI_DOUBLE, buffer, BUFFER_SIZE, &position, comm);
			MPI_Send(buffer, BUFFER_SIZE, MPI_PACKED, MASTER, FINISH_TASK_TAG, comm);
		}
		freePointsFromDevice(&dev_points, N);
	}

	free(W);
	if (rank == MASTER)
		free_alpha_array();
}

int check_lowest_alpha(double* returned_alpha, double* returned_q, double QC, double* W, int dim) {
	static int alpha_array_state = ALPHA_NOT_FOUND;
	if (*returned_q <= QC)
		alpha_array_state = ALPHA_POTENTIALLY_FOUND;
	int index = (int)(((*returned_alpha) / alpha_array[0].value) - 1);
	alpha_array[index].q = *returned_q;
	copy_vector(alpha_array[index].W, W, dim);
	
	for (int i = 0; i < alpha_array_size; i++)
	{
		if (alpha_array[index].q == Q_NOT_CHECKED)
			return alpha_array_state;
		if (alpha_array[index].q <= QC)
		{
			*returned_alpha = alpha_array[index].value;
			*returned_q = alpha_array[index].q;
			copy_vector(W, alpha_array[index].W, dim);
			alpha_array_state = ALPHA_FOUND;
			return alpha_array_state;
		}
	}
	return alpha_array_state;
}

double get_quality_with_alpha(Point* points,double alpha,double* W,int N,int K,int LIMIT) {
	int fault_flag = FAULT,faulty_point;
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
				faulty_point = i;
				break;
			}
		}
		if (fault_flag == FAULT) {
			//4
			adjustW(W, K, points[faulty_point].x, val, alpha);
		}
	}
	//6 find q
	return get_quality(points, W, N, K);
}

void print_arr(double* W, int dim) {
	for (int i = 0; i < dim; i++)
		printf("%f\n", W[i]);
}

