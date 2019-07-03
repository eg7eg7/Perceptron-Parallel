#pragma once
#ifndef MACRO_HEADER
#define MACRO_HEADER


#define HEAVY 10000
#define PATH "C:\\numbers.txt"

/*
The following defines are for debug purposes
*/

//generates Random numbers without using file
//#define DEBUG_GENERATE_RANDOM

//if using random numbers, uses fixed size of 1000 numbers
//#define DEBUG_RAND_FIXED_SIZE

//if reading from file, read a fixed size of numbers (file has to contain at least indicated amount)
//#define DEBUG_FIXED_SIZE_FILE
#define FIXED_SIZE 10000


//prints array after initializing array - not recommended for large arrays
//#define DEBUG_PRINT_ARRAY

//verbal prints
//#define DEBUG_VERB_PRINT 

#endif // !MACRO_HEADER