#pragma once
#pragma warning( disable : 4996)
#ifndef PERCEPTRON
#define PERCEPTRON
#include <mpi.h>
#include "vectors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "myMacro.h"
#include <omp.h>
#include <math.h>
#define SET_A 1
#define SET_B -1
#define MAX_QC 1
#define NO_FAULT 0
#define FAULT 1
#define Q_NOT_CHECKED -1
#define POINT_CORRECT -1

#define ALL_POINTS_CORRECT 1
#define W_ADJUSTED 2

#define PRINT
/*
x - array of data for each dimension
set - point belonging to SET A or SET B
*/
typedef struct struct_point {
	int set;
	double* x;
} Point;

typedef struct struct_alpha {
	double value;
	double* W;
	double q;
} Alpha;


void Perceptron_readDataset(const char* path, int rank, MPI_Comm comm, int* N, int* K, double* alpha_zero, double* alpha_max, int* LIMIT, double* QC, Point** point_array);
/*
vector multiplication, W and x are one dimension higher than K
for multiplication Point.x[dim]=1
*/
void print_arr(double* W, int dim);
void free_alpha_array();
void init_alpha_array(double alpha_max, double alpha_zero, int dim);
int check_lowest_alpha(double* returned_alpha, double* returned_q, double QC, double* W, int dim);
double f(double* x, double* W, int dim);
double get_quality(Point* points, double* W, int N, int K);
int init_W(double** W, int K);
void zero_W(double* W, int K);
void adjustW(double* W, int dim, double* p_xi, double f_p_scalar, double alpha);
void initPointArray(Point** points, int N, int K);
void freePointArray(Point** points, int size);
void printPointArray(Point* points, int size, int dim,int rank);
void run_perceptron_sequential(const char* output_path, int N, int K, double alpha_zero, double alpha_max, int LIMIT, double QC, Point* points, double* W);
void run_perceptron_parallel(const char* output_path, int rank, int world_size,MPI_Comm comm,int N, int K, double alpha_zero, double alpha_max, int LIMIT, double QC, Point* points, double* W);

int sign(double a);
void printPerceptronOutput(const char* path, double* W, int K, double alpha, double q, double QC, double time);
//add mallocAdjustment for easier freeing later

//dummy???
double get_quality_with_alpha(Point* points, double alpha, double* W, int N, int K, int LIMIT);


#endif // !PERCEPTRON