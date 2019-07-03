#pragma once
#ifndef PERCEPTRON
#define PERCEPTRON
#include <mpi.h>
#include "vectors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include  <errno.h>
#include "myApp.h"
#define SET_A 1
#define SET_B -1
#define MAX_QC 1
#define NO_FAULT 0
#define FAULT 1


/*
x - array of data for each dimension
set - point belonging to SET A or SET B
*/
typedef struct struct_point {
	int set;
	double* x;
} Point;

void Perceptron_readDataset(const char* path, int rank, MPI_Comm comm, int* N, int* K, double* alpha_zero, double* alpha_max, int* LIMIT, double* QC, Point** point_array);
/*
vector multiplication, W and x are one dimension higher than K
for multiplication Point.x[dim]=1
*/
double f(double* x, double* W, int dim);
double get_quality(Point* points, double* W, int N, int K);
int init_W(double** W, int K);
void zero_W(double* W, int K);
void adjustW(double* W, int dim, double* p_xi, double f_p_scalar, double alpha);
void initPointArray(Point** points, int N, int K);
void freePointArray(Point** points, int size);
void printPointArray(Point* points, int size, int dim,int rank);
void run_perceptron_sequential(int N, int K, double alpha_zero, double alpha_max, int LIMIT, double QC, Point* points, double* W);
int sign(double a);
//add mallocAdjustment for easier freeing later

#endif // !PERCEPTRON