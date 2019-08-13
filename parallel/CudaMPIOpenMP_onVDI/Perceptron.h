#pragma once
#pragma warning( disable : 4996)
#ifndef PERCEPTRON
#define PERCEPTRON
#include <mpi.h>
#include "vectors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <math.h>

/* Point belongs to either set a or set b*/
#define SET_A 1
#define SET_B -1

/*highest possible value for q*/
#define MAX_QC 1
#define NO_FAULT 0
#define FAULT 1

/* initial value for alpha value*/
#define Q_NOT_CHECKED -1

/* flag for kernel*/
#define POINT_CORRECT -1

/*definitions for kernel flags - */
#define ALL_POINTS_CORRECT 1
#define W_ADJUSTED 2

/* Print result to screen or file*/
#define PRINT

/* Master host value*/
#define MASTER 0

/* Tags to send between processes*/
#define START_TASK_TAG 0
#define FINISH_PROCESS_TAG 1
#define FINISH_TASK_TAG 2

/* definitions for helper function - lowest alpha determine*/
#define ALPHA_NOT_FOUND 0
#define ALPHA_FOUND 1
#define ALPHA_POTENTIALLY_FOUND 2

/* buffer for MPI packages - max package size*/
#define BUFFER_SIZE 1000

/*
x - array of data for each dimension
set - point belonging to SET A or SET B
*/
typedef struct struct_point {
	int set;
	double* x;
} Point;

/*Helper structure - for each alpha, holds W and best q reached*/
typedef struct struct_alpha {
	double value;
	double* W;
	double q;
} Alpha;



//************UNIVERSAL******************//
/*zero the vector*/
void zero_W(double* W, int K);

/*malloc for W with an extra point in vector*/
int init_W(double** W, int K);

void printPointArray(Point* points, int size, int dim, int rank);

void printPerceptronOutput(const char* path, double* W, int K, double alpha, double q, double QC, double time);

void print_arr(double* W, int dim);

/*read dataset and send to host, also copies array to GPU and returns in pointer*/
void Perceptron_readDataset(const char* path, int rank, MPI_Comm comm, int* N, int* K, double* alpha_zero, double* alpha_max, int* LIMIT, double* QC, Point** point_array, Point** device_point_array);

//************SEQUENTIAL*****************//
/*multiplies x vector with W vector, taking into account the extra point in vector*/
double f(double* x, double* W, int dim);

/*returns ratio between incorrect points to total points - sequential*/
double get_quality(Point* points, double* W, int N, int K);

/*adjust W vector accoring to alpha and value - sequential */
void adjustW(double* W, int dim, double* p_xi, double f_p_scalar, double alpha);

/*if positive return SET A, else returns SET B*/
int sign(double a);

void run_perceptron_sequential(const char* output_path, int N, int K, double alpha_zero, double alpha_max, int LIMIT, double QC, Point* points);

double get_quality_with_alpha(Point* points, double alpha, double* W, int N, int K, int LIMIT);

/*init point array - sequential*/
void initPointArray(Point** points, int N, int K);

/*free point array sequential*/
void freePointArray(Point** points, Point** dev_points, int size);
//**********PARALLEL************//
/*free global alpha array*/
void free_alpha_array();

/*malloc the global array with the amount of alphas possible*/
void init_alpha_array(double alpha_max, double alpha_zero, int dim);
/*Helper function for Master host, decides whether alpha is found or not*/
int check_lowest_alpha(double* returned_alpha, double* returned_q, double QC, double* W, int dim);

/*main function to run perceptron in parallel, both master and slaves*/
void run_perceptron_parallel(const char* output_path, int rank, int world_size, MPI_Comm comm, int N, int K, double alpha_zero, double alpha_max, int LIMIT, double QC, Point* points);




#endif // !PERCEPTRON