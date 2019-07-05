#pragma once

#ifndef MYAPP_H
#define myAPP_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include "myApp.h"

#include <omp.h>
#include <math.h>
#include "myMacro.h"
cudaError_t resultWithCuda(int *array, int arraysize, int *result);
#define MASTER 0

#endif

