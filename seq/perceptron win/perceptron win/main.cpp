//#include "pch.h"
#include "stdafx.h"

#include "Perceptron.h"


static const char PATH[] = "C:\\Dataset Maker Perceptron\\dataset.txt";

int test_mult_scalar_with_vector() {
	double vector1[10] = { 11,11,11,11,11,11,11,11,11,11 };
	double result[10];
	double scalar = 5;

	mult_scalar_with_vector(vector1, 10, scalar, result);
	for (int i = 0; i < 10; i++)
		if (result[i] != 55)
		{
			printf("test_mult_scalar_with_vector fail\n");
			return 1;
		}
	return 0;
}
int test_mult_vector_with_vector() {
	double vector1[10] = { 11,11,11,11,11,11,11,11,11,11 };
	double vector2[10] = { 11,11,11,11,11,11,11,11,11,11 };
	double result = mult_vector_with_vector(vector1, vector2, 10);

	if (result != 1210.0)
	{
		printf("test_mult_vector_with_vector fail\n");
		return 1;
	}
	return 0;
}

int test_add_vector_to_vector() {
	double vector1[10] = { 11,11,11,11,11,11,11,11,11,11 };
	double vector2[10] = { 11,11,11,11,11,11,11,11,11,11 };
	double result[10];

	add_vector_to_vector(vector1, vector2, 10, result);

	for (int i = 0; i < 10; i++)
		if (result[i] != 22)
		{
			printf("test_add_vector_to_vector fail\n");
			return 1;
		}
	return 0;
}
int test_adjust_W() {
	double W_init[10] = { 1,2,3,4 };
	double W[10] = { 1,2,3,4 };
	Point p;
	p.set = 1;
	p.x = (double*)malloc(sizeof(double) * 4);
	p.x[0] = 1;
	p.x[1] = 2;
	p.x[2] = 3;
	p.x[3] = 1;
	double alpha = 0.5;
	double scalar = mult_vector_with_vector(W, p.x, 4);
	adjustW(W, 3, p.x, scalar, alpha);
	for (int i = 0; i < 4; i++)
		if (W[i] != (W_init[i] - alpha * p.x[i]))
		{
			printf("test_adjust_W fail\n");
			return 1;
		}
	scalar *= -1;
	for (int i = 0; i < 4; i++)
		W[i] = i + 1;
	adjustW(W, 3, p.x, scalar, alpha);
	for (int i = 0; i < 4; i++)
		if (W[i] != (W_init[i] + alpha * p.x[i]))
		{
			printf("test_adjust_W fail\n");
			return 1;
		}
	free(p.x);
	return 0;
}

void runTests()
{
	int tests_fail = 0;
	tests_fail += test_mult_scalar_with_vector();
	tests_fail += test_mult_vector_with_vector();
	tests_fail += test_add_vector_to_vector();
	tests_fail += test_adjust_W();
	if (tests_fail)
		printf("failed %d tests.\n", tests_fail);
	else
		printf("tests passed.\n");
}

int main(int argc, char *argv[])
{
	runTests();

	int N, K, LIMIT;
	double alpha_zero, alpha_max, QC;
	double* W = 0;
	Point* points = 0;

	Perceptron_readDataset(PATH, &N, &K, &alpha_zero, &alpha_max, &LIMIT, &QC, &points);
	printf("N=%d K=%d alpha zero = %f alpha_max = %f LIMIT=%d QC = %f\n", N, K, alpha_zero, alpha_max, LIMIT, QC);
	//printPointArray(points, N, K);

	run_perceptron_sequential(N, K, alpha_zero, alpha_max, LIMIT, QC, points, W);

	freePointArray(&points, N);
	return 0;
}


