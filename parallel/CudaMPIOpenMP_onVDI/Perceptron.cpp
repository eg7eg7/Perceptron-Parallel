
#include "Perceptron.h"

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

	const int line_size = 1000;
	char delim[] = " ";
	char* line;
	FILE *file;
	char *token;
	if (rank == MASTER)
	{
		line = (char*)malloc(line_size);
		file = fopen(path, "r");
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
	int set;
	for (int i = 0; i < (*N); i++)
	{
		if (rank == MASTER) {
			if (fgets(line, line_size, file) != NULL)
				if (line == NULL)
					fileReadFailure("line is NULL");
			token = strtok(line, delim);
		}
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
	for (int i = 0; i < N; i++) {
		double val = f(points[i].x, W, K);
		if (sign(val) != points[i].set)
			N_mis++;
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
void run_perceptron_sequential(int N, int K, double alpha_zero, double alpha_max, int LIMIT, double QC, Point* points, double* W) {
	double val;
	int fault_flag = FAULT;
	double alpha, current_q = MAX_QC, best_q = MAX_QC, best_alpha;
	int loop;
	int faulty_point;

	if (!init_W(&W, K))
	{
		printf("malloc assignment error");
		return;
	}
	double* min_W;
	init_W(&min_W, K);

	//alpha=alpha zero  1, alpha=alpha+alpha_zero 9
	for (alpha = alpha_zero; alpha <= alpha_max; alpha += alpha_zero)
	{
		//2
		zero_W(W, K);
		//5 - loop through 3 and 4 til all points are properly classified or limit reached
		for (loop = 1; loop < LIMIT && fault_flag == FAULT; loop++)
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

	printf("finished sequential.\n");
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
	free(W);
	free(min_W);
}