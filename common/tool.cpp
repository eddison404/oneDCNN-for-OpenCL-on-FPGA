#define _CRT_SECURE_NO_WARNINGS
#include <CL/cl.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>
#include "tool.h"
using namespace std;


//softmax function
void softmax(const float* input, const int size, float* output)
{
    float total = 0;
    for (int i = 0; i < size; ++i)
    {
        total += std::exp(input[i]);
    }

    for (int i = 0; i < size; ++i)
    {
        output[i] = std::exp(input[i]) / total;
    }
}

int classifier(const float* softmaxOutput, const int size)
{
    // 找到具有最大概率的类别索引
    int maxIndex = 0;
    float maxValue = softmaxOutput[0];
    for (int i = 1; i < size; ++i)
    {
        if (softmaxOutput[i] > maxValue)
        {
            maxValue = softmaxOutput[i];
            maxIndex = i;
        }
    }

    return maxIndex;
}


float* readFloatsFromFile(const char* filename, int* size) {
    FILE* file;
    float* array;
    int count = 0;

    file = fopen(filename, "r");
    if (file == NULL) {
        printf("无法打开文件 %s\n", filename);
        return NULL;
    }

    // 计算文件中的行数
    char ch;
    while ((ch = fgetc(file)) != EOF) {
        if (ch == '\n') {
            count++;
        }
    }
    rewind(file); // 将文件指针复位到文件开头

    // 动态分配内存用于保存浮点型数组
    array = (float*)malloc(count * sizeof(float));
    if (array == NULL) {
        printf("内存分配失败\n");
        fclose(file);
        return NULL;
    }

    // 从文件中读取每行的浮点数数据
    char line[100];
    int i = 0;
    while (fgets(line, sizeof(line), file)) {
        array[i] = atof(line);
        i++;
    }

    fclose(file);

    *size = count;
    return array;
}


 

void print_platform_info(cl_platform_id platform) {
    char buffer[1024];
    cl_uint buf_uint;
    cl_ulong buf_ulong;

    printf("\n--- Platform Info ---\n");



    clGetPlatformInfo(platform, CL_PLATFORM_VERSION, 1024, buffer, NULL);
    printf("VERSION = %s\n", buffer);

    clGetPlatformInfo(platform, CL_PLATFORM_NAME, 1024, buffer, NULL);
    printf("NAME = %s\n", buffer);


}



/** convert the kernel file into a string */
int convertToString(const char* filename, std::string& s)
{
    size_t size;
    char* str;
    std::fstream f(filename, (std::fstream::in | std::fstream::binary));

    if (f.is_open())
    {
        size_t fileSize;
        f.seekg(0, std::fstream::end);
        size = fileSize = (size_t)f.tellg();
        f.seekg(0, std::fstream::beg);
        str = new char[size + 1];
        if (!str)
        {
            f.close();
            return 0;
        }

        f.read(str, fileSize);
        f.close();
        str[size] = '\0';
        s = str;
        delete[] str;
        return 0;
    }
    cout << "Error: failed to open file\n:" << filename << endl;
    return -1;
}

/**Getting platforms and choose an available one.*/
int getPlatform(cl_platform_id& platform)
{
    platform = NULL;//the chosen platform

    cl_uint numPlatforms;//the NO. of platforms
    cl_int    status = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (status != CL_SUCCESS)
    {
        cout << "Error: Getting platforms in the function!" << endl;
        return -1;
    }

    /**For clarity, choose the first available platform. */
    if (numPlatforms > 0)
    {
        cl_platform_id* platforms =
            (cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id));
        status = clGetPlatformIDs(numPlatforms, platforms, NULL);
        platform = platforms[1];    //选择第一个平台，，本机为cpu


        //输出选择的平台信息
        print_platform_info(platform);
        char platformName[128];
        clGetPlatformInfo(platform ,CL_PLATFORM_NAME, sizeof(platformName), platformName, NULL);
        printf(" %s\n", platformName);

      //  free(platforms);
    }
    else
        return -1;

    return 0;
}

/**Step 2:Query the platform and choose the firstfpga device if has one.*/
cl_device_id* getCl_device_id(cl_platform_id& platform)
{
    cl_uint numDevices = 0;
    cl_device_id* devices = NULL;
    cl_int    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
    if (numDevices > 0) //fpga able.
    {
        devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);

        printf("    设备信息：\n");
        for (cl_uint j = 0; j < numDevices; ++j) {
            char deviceName[128];
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
            printf("  [%u] %s\n", j, deviceName);
        }
    }

    return devices;   //此处返回了所有设备
}

