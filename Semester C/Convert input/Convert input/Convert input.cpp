// Convert input.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#pragma warning(disable:4996)

#include "pch.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#define INPUT "C://input.txt"
#define OUTPUT "C://output.txt"

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
void fileReadFailure(const char* error) {
	printf("%s\n", error);
	fileReadFailure();
}
void perceptron_read_dataset() {
	char delim[] = " ";
	const int line_size = 1000;
	int K, N, LIMIT;
	double QC, alpha_zero,alpha_max;
	FILE *file;
	FILE *file_out;
	char *token;
	char* line;
	printf("Reading..\n");
	line = (char*)malloc(line_size);
	fopen_s(&file, INPUT, "r");
	if (file == NULL)
		fileReadFailure("Could not find or open file for reading. \n");
	fopen_s(&file_out, OUTPUT, "w");
	if (file_out == NULL)
		fileReadFailure("Could not find or open file for writing. \n");
	//reading first line of the dataset file
	if (fgets(line, line_size, file) != NULL)
	{
		//reading N
		token = strtok(line, delim);
		if (token == NULL)
			fileReadFailure("could not token N");
		if (!isInteger(token, &N))
			fileReadFailure("wrong token - N");

		//reading K
		token = strtok(NULL, delim);
		if (token == NULL)
			fileReadFailure("could not token K");
		if (!isInteger(token, &K))
			fileReadFailure("wrong token - K");

		//reading alpha_zero
		token = strtok(NULL, delim);
		if (token == NULL)
			fileReadFailure();
		if (!isDouble(token, &alpha_zero))
			fileReadFailure();

		//reading alpha_max
		token = strtok(NULL, delim);
		if (token == NULL)
			fileReadFailure();
		if (!isDouble(token, &alpha_max))
			fileReadFailure();

		//reading LIMIT
		token = strtok(NULL, delim);
		if (token == NULL)
			fileReadFailure();
		if (!isInteger(token, &LIMIT))
			fileReadFailure();

		//reading QC
		token = strtok(NULL, delim);
		if (token == NULL)
			fileReadFailure();
		if (!isDouble(token, &QC))
			fileReadFailure();

	}
	double temp;
	int temp_int;
	fprintf(file_out, "%d %d %f %f %f %d %f\n", N, K, 0.0, 0.0, alpha_zero, LIMIT, QC);
	for (int i = 0; i < (N); i++)
	{
		if (fgets(line, line_size, file) != NULL)
			if (line == NULL)
				fileReadFailure("line is NULL");
		token = strtok(line, delim);

		for (int j = 0; j < (K) && token != NULL; j++)
		{
			isDouble(token, &temp);
			fprintf(file_out, "%f ", temp);
			token = strtok(NULL, delim);
		}
		isInteger(token, &temp_int);
		for (int j = 0; j < (K) && token != NULL; j++)
		{
			fprintf(file_out, "%f ", temp_int*0.3);
		}
		fprintf(file_out, "%d\n", temp_int);
		
	}
	fclose(file);
	fclose(file_out);
	free(line);
}
int main()
{
	perceptron_read_dataset();
}
