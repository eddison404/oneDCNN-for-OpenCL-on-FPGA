#ifndef TOOLH
#define TOOLH

#include <CL/cl.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>
using namespace std;

void softmax(const float* input, const int size, float* output);

int classifier(const float* softmaxOutput, const int size);

float* readFloatsFromFile(const char* filename, int* size);

 
void print_platform_info(cl_platform_id platform);
/** convert the kernel file into a string */
int convertToString(const char* filename, std::string& s);

/**Getting platforms and choose an available one.*/
int getPlatform(cl_platform_id& platform);

/**Step 2:Query the platform and choose the first GPU device if has one.*/
cl_device_id* getCl_device_id(cl_platform_id& platform);

#endif