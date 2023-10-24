#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tool.h"
//#include "CL/opencl.h"
#include "aocl_utils.h"

#include "oneDnet_params.h"
//#include "dog.h"

#pragma warning( disable : 4996 )
using namespace aocl_utils;


  // OpenCL runtime configuration
cl_platform_id platform = NULL;
unsigned num_devices = 0;
scoped_array<cl_device_id> device; // num_devices elements
cl_context context = NULL;
cl_command_queue queue;
cl_program program = NULL;


cl_kernel conv3x3, maxpool, avgGlobal, fc_matrix_mult;

cl_mem  d_sample, d_conv1_weight, d_conv1_bias, d_result_conv1, d_result_pool1;
cl_mem  d_conv2_weight, d_conv2_bias, d_result_conv2, d_result_pool2;
cl_mem  d_conv3_weight, d_conv3_bias, d_result_conv3, d_result_pool3;
cl_mem  d_conv4_weight, d_conv4_bias, d_result_conv4, d_result_pool4;
cl_mem  d_conv5_weight, d_conv5_bias, d_result_conv5, d_result_pool5;
cl_mem  d_conv6_weight, d_conv6_bias, d_result_conv6, d_result_pool6;
cl_mem  d_result_global_average;

cl_mem  d_result_fc1, d_fc1_weight, d_fc1_bias;

cl_mem  d_result_fc2, d_fc2_weight, d_fc2_bias;

cl_mem  d_result_fc3, d_fc3_weight, d_fc3_bias;


//创建时间数组用于存放执行时间

vector<float> time_conv1;
vector<float> time_conv2;
vector<float> time_conv3;
vector<float> time_conv4;
vector<float> time_conv5;
vector<float> time_conv6;
vector<float> time_globle_average;
vector<float> time_FC;


 


int oneDCNN( char* filenametxt) {


    // Initialize OpenCL.
    cl_int status;

    printf("Initializing OpenCL\n");

    if (!setCwdToExeDir()) {
        return 1;
    }


    //***************************************************
    //FPGA虚拟平台
    //**************************************************


    // Get the OpenCL platform.
    
    //platform = findPlatform("Intel(R) FPGA");
    //if (platform == NULL) {
    //    printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
    //    return 1;
    //}

    //// Query the available OpenCL device.
    //device.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));
    //printf("Platform: %s\n", getPlatformName(platform).c_str());
    //printf("Using %d device(s)\n", num_devices);
    //for (unsigned int i = 0; i < num_devices; ++i) {
    //    printf("  %s\n", getDeviceName(device[i]).c_str());
    //}

//***************************************************
//cpu平台
//**************************************************


   //the chosen platform

    status=getPlatform(platform);
    if (status != CL_SUCCESS)
    {
        cout << "Error: Getting platforms!" << endl;
        return -1;
    }

    cl_uint num_devices = 0;


    cl_device_id* device = NULL;
     status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    if (num_devices > 0) // 
    {
        device = (cl_device_id*)malloc(num_devices * sizeof(cl_device_id));
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, device, NULL);

       // printf("    设备信息：\n");
        /*for (cl_uint j = 0; j < num_devices; ++j) {
            char deviceName[128];
            clGetDeviceInfo(device[j], CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
            printf("  [%u] %s\n", j, deviceName);
        }*/
    }





   
   

    // Create the context.
    context = clCreateContext(NULL, num_devices, device, NULL, NULL, &status);   //创建上下文，包含全部平台（本机只有一个虚拟平台）
    checkError(status, "Failed to create context");




    const char* filename = "F:/VS2022/Prj/oneDCNN/oneDCNN/squeezenet.cl";
    std::string sourceStr;
    status = convertToString(filename, sourceStr);
    const char* source = sourceStr.c_str();
    size_t sourceSize[] = { strlen(source) };
    program = clCreateProgramWithSource(context, 1, &source, sourceSize, NULL);



    // Build the program that was just created.
    status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
    checkError(status, "Failed to build program");

    queue = clCreateCommandQueue(context, device[0], CL_QUEUE_PROFILING_ENABLE, &status);   //在上下文和设备直接创建 命令行队列
    checkError(status, "Failed to create command queue");


    // creat   Kernel.
    const char* kernel1 = "conv2d3x3";
    conv3x3 = clCreateKernel(program, kernel1, &status);
    checkError(status, "Failed to create kernel conv2d3x3");

    const char* kernel2 = "maxpool2d";
    maxpool = clCreateKernel(program, kernel2, &status);
    checkError(status, "Failed to create kernel maxpool2d");

    const char* kernel3 = "average2d";
    avgGlobal = clCreateKernel(program, kernel3, &status);
    checkError(status, "Failed to create kernel average2d");

    const char* kernel4 = "matrix_mult";
    fc_matrix_mult = clCreateKernel(program, kernel4, &status);
    checkError(status, "Failed to create kernel matrix_mult");



    /**************************************************************/
    /*                          conv1   CreateBuffer                         */
    /**************************************************************/
    //Creat device buffers





    int size;
    float* array = readFloatsFromFile(filenametxt, &size);

    //if (array != NULL) {
    //    printf("读取到的浮点数数组为：\n");
    //    for (int i = 0; i < size; i++) {
    //        printf("%f ", array[i]);
    //    }
    //    printf("\n");

    //    free(array);
    //}

    d_sample = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * (1 * 1 * 1 * 4097), array, &status);
    checkError(status, "Failed to create buffer d_sample");

    //conv1 params
    d_conv1_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(conv1_weight), conv1_weight, &status);      //4*1*1*6  有4个kernel。总共24个参数
    checkError(status, "Failed to create buffer d_conv1_weight");


    d_conv1_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(conv1_bias), conv1_bias, &status);    //4个卷积核对应四个偏置
    checkError(status, "Failed to create buffer d_conv1_bias");

    //conv1 result
    d_result_conv1 = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        sizeof(float) * (1 * 1 * 4 * 4092), NULL, &status);   //卷积一层输出结果
    checkError(status, "Failed to create buffer d_result_conv");

    d_result_pool1 = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        sizeof(float) * (1 * 1 * 4 * 2046), NULL, &status);   //池化一层输出结果
    checkError(status, "Failed to create buffer d_result_pool1");

    /**************************************************************/
  /*                          conv 2   CreateBuffer                        */
  /**************************************************************/
    //conv2 params
    d_conv2_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(conv2_weight), conv2_weight, &status);      // 
    checkError(status, "Failed to create buffer d_conv2_weight");


    d_conv2_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(conv2_bias), conv2_bias, &status);    // 
    checkError(status, "Failed to create buffer d_conv2_bias");

    //conv2 result
    d_result_conv2 = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        sizeof(float) * (1 * 1 * 4 * 2042), NULL, &status);   //卷积2层输出结果
    checkError(status, "Failed to create buffer d_result_conv");

    d_result_pool2 = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        sizeof(float) * (1 * 1 * 4 * 1021), NULL, &status);   //池化2层输出结果
    checkError(status, "Failed to create buffer d_result_pool2");

    //  /**************************************************************/
    ///*                          conv3  CreateBuffer                         */
    ///**************************************************************/

      //conv3 params
    d_conv3_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(conv3_weight), conv3_weight, &status);      // 
    checkError(status, "Failed to create buffer d_conv3_weight");


    d_conv3_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(conv3_bias), conv3_bias, &status);    // 
    checkError(status, "Failed to create buffer d_conv3_bias");

    //conv3 result
    d_result_conv3 = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        sizeof(float) * (1 * 1 * 10 * 1018), NULL, &status);   //卷积3层输出结果
    checkError(status, "Failed to create buffer d_result_conv");

    d_result_pool3 = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        sizeof(float) * (1 * 1 * 10 * 509), NULL, &status);   //池化3层输出结果
    checkError(status, "Failed to create buffer d_result_pool3");


    //  /**************************************************************/
    ///*                          conv 4   CreateBuffer                        */
    ///**************************************************************/
        //conv4 params
    d_conv4_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(conv4_weight), conv4_weight, &status);      // 
    checkError(status, "Failed to create buffer d_conv4_weight");


    d_conv4_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(conv4_bias), conv4_bias, &status);    // 
    checkError(status, "Failed to create buffer d_conv4_bias");

    //conv4 result
    d_result_conv4 = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        sizeof(float) * (1 * 1 * 10 * 506), NULL, &status);   //卷积4层输出结果
    checkError(status, "Failed to create buffer d_result_conv");

    d_result_pool4 = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        sizeof(float) * (1 * 1 * 10 * 253), NULL, &status);   //池化4层输出结果
    checkError(status, "Failed to create buffer d_result_pool4");

    //    /**************************************************************/
    //  /*                          conv5  CreateBuffer                          */
    //  /**************************************************************/

        //conv5 params
    d_conv5_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(conv5_weight), conv5_weight, &status);      // 
    checkError(status, "Failed to create buffer d_conv5_weight");


    d_conv5_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(conv5_bias), conv5_bias, &status);    // 
    checkError(status, "Failed to create buffer d_conv5_bias");

    //conv5 result
    d_result_conv5 = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        sizeof(float) * (1 * 1 * 15 * 250), NULL, &status);   //卷积5层输出结果
    checkError(status, "Failed to create buffer d_result_conv");

    d_result_pool5 = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        sizeof(float) * (1 * 1 * 15 * 125), NULL, &status);   //池化5层输出结果
    checkError(status, "Failed to create buffer d_result_pool5");


    //  /**************************************************************/
    //  /*                          conv 6   CreateBuffer                        */
    //  /**************************************************************/
        //conv6 params
    d_conv6_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(conv6_weight), conv6_weight, &status);      // 
    checkError(status, "Failed to create buffer d_conv6_weight");


    d_conv6_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(conv6_bias), conv6_bias, &status);    // 
    checkError(status, "Failed to create buffer d_conv6_bias");

    //conv6 result
    d_result_conv6 = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        sizeof(float) * (1 * 1 * 8 * 120), NULL, &status);   //卷积6层输出结果
    checkError(status, "Failed to create buffer d_result_conv");

    d_result_pool6 = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        sizeof(float) * (1 * 1 * 8 * 1), NULL, &status);   //              全局平均池化，池化6层输出结果，将120个项平均为一个项
    checkError(status, "Failed to create buffer d_result_pool6");

    //  /**************************************************************/
  //  /*                         global_average   CreateBuffer          */
  //  /**************************************************************/
    d_result_global_average = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        sizeof(float) * (8), NULL, &status);   //              全局平均池化，池化6层输出结果，将120个项平均为一个项
    checkError(status, "Failed to create buffer d_result_global_average");
    //  /**************************************************************/
  //  /*                        FC   CreateBuffer          */
  //  /**************************************************************/


    d_fc1_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(fc1_weight), fc1_weight, &status);
    checkError(status, "Failed to create buffer d_fc1_weight");

    d_fc2_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(fc2_weight), fc2_weight, &status);
    checkError(status, "Failed to create buffer d_fc2_weight");

    d_fc3_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(fc3_weight), fc3_weight, &status);
    checkError(status, "Failed to create buffer d_fc3_weight");

    d_fc1_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(fc1_bias), fc1_bias, &status);
    checkError(status, "Failed to create buffer d_fc1_bias");

    d_fc2_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(fc2_bias), fc2_bias, &status);
    checkError(status, "Failed to create buffer d_fc2_bias");

    d_fc3_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(fc3_bias), fc3_bias, &status);
    checkError(status, "Failed to create buffer d_fc3_bias");


    d_result_fc1 = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        sizeof(float) * (50), NULL, &status);
    checkError(status, "Failed to create buffer d_result_fc1");
    d_result_fc2 = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        sizeof(float) * (20), NULL, &status);
    checkError(status, "Failed to create buffer d_result_fc2");
    d_result_fc3 = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        sizeof(float) * (3), NULL, &status);
    checkError(status, "Failed to create buffer d_result_fc3");








    /**************************************************************/
    /*                          conv1                             */
    /**************************************************************/
    printf("\r\nOneDNet on FPGA start:\r\n");
    printf("kernel version 1.1\r\n");
    double total = 0.0;
    double start_time = getCurrentTimestamp();

    unsigned int input_channel, input_size, pad, stride, start_channel, output_size, filter_length, filter_height;

    input_channel = 1;
    input_size = 4097;
    pad = 0;
    stride = 1;
    start_channel = 0;
    output_size = 4092;
    filter_length = 6;
    filter_height = 1;


    status |= clSetKernelArg(conv3x3, 0, sizeof(int), &(input_channel));
    checkError(status, "Setting conv1: conv1x6 arguments");
    status |= clSetKernelArg(conv3x3, 1, sizeof(int), &(input_size));
    checkError(status, "Setting conv1: conv1x6 arguments");
    status |= clSetKernelArg(conv3x3, 2, sizeof(int), &(pad));
    checkError(status, "Setting conv1: conv1x6 arguments");
    status |= clSetKernelArg(conv3x3, 3, sizeof(int), &(stride));
    checkError(status, "Setting conv1: conv1x6 arguments");
    status |= clSetKernelArg(conv3x3, 4, sizeof(int), &(start_channel));
    checkError(status, "Setting conv1: conv1x6 arguments");
    status |= clSetKernelArg(conv3x3, 5, sizeof(int), &(output_size));
    checkError(status, "Setting conv1: conv1x6 arguments");
    status |= clSetKernelArg(conv3x3, 6, sizeof(int), &(filter_length));
    checkError(status, "Setting conv1: conv1x6 arguments");
    status |= clSetKernelArg(conv3x3, 7, sizeof(int), &(filter_height));
    checkError(status, "Setting conv1: conv1x6 arguments");
    status |= clSetKernelArg(conv3x3, 8, sizeof(cl_mem), &d_sample);
    checkError(status, "Setting conv1: conv1x6 arguments");
    status |= clSetKernelArg(conv3x3, 9, sizeof(cl_mem), &d_conv1_weight);
    checkError(status, "Setting conv1: conv1x6 arguments");
    status |= clSetKernelArg(conv3x3, 10, sizeof(cl_mem), &d_conv1_bias);
    checkError(status, "Setting conv1: conv1x6 arguments");
    status |= clSetKernelArg(conv3x3, 11, sizeof(cl_mem), &d_result_conv1);


    checkError(status, "Setting conv1: conv1x6 arguments");

    // 创建事件对象
    cl_event event;
    size_t global = 4;  //一维，64个工作项对应64个卷积核，也就是一个工作项执行3*224*224（Conv）3*3*3 => 1*111*111
    status = clEnqueueNDRangeKernel(queue, conv3x3, 1, NULL, &global, NULL, 0, NULL, &event);
    checkError(status, "Enqueueing conv1: conv3x3");
    //    command_queue：命令队列对象，用于指定执行内核函数的设备。
    //    kernel：要执行的内核函数对象。
    // 
    // 1         work_dim：工作维度，即工作项的维度数量。 
    // null      global_work_offset：全局工作偏移量，在各个维度上指定每个工作项的起始位置。
    // &global   global_work_size：全局工作大小，指定每个维度上的工作项数量。
    //NULL       local_work_size：局部工作大小，指定每个工作组中的工作项数量。如果为 NULL，则表示不使用工作组。
    // 0         num_events_in_wait_list：等待列表中的事件数量。
    //NULL       event_wait_list：要等待的事件列表，指定在执行内核函数之前需要等待的事件。
    //NULL       event：用于返回执行内核函数生成的事件对象



    if (status != CL_SUCCESS) {
        // 处理错误
        printf("出错");

    }
    // 等待事件对象完成
    status = clWaitForEvents(1, &event);
    if (status != CL_SUCCESS) {
        // 处理错误
    }
     //查询事件对象的执行状态
    cl_int event_status;
    status = clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &event_status, NULL);
    if (status != CL_SUCCESS) {
        // 处理错误
    }
    // 根据查询到的状态进行相应处理
    switch (event_status) {
    case CL_COMPLETE:
        printf("核函数执行完成");
        // 核函数执行完成
        break;
    case CL_RUNNING:
        printf("核函数正在运行中");
        // 核函数正在运行中
        break;
    case CL_SUBMITTED:
        printf("核函数已提交但尚未开始执行");
        // 核函数已提交但尚未开始执行
        break;
    case CL_QUEUED:
        printf("核函数被加入队列但尚未提交");
        // 核函数被加入队列但尚未提交
        break;
    default:
        printf("处理其他状态");
        // 处理其他状态
        break;
    }
    // 释放事件对象
    clReleaseEvent(event);


    input_size = 4092;
    output_size = 2046;


    status |= clSetKernelArg(maxpool, 0, sizeof(int), &(input_size));
    status |= clSetKernelArg(maxpool, 1, sizeof(int), &(output_size));
    status |= clSetKernelArg(maxpool, 2, sizeof(cl_mem), &d_result_conv1);
    status |= clSetKernelArg(maxpool, 3, sizeof(cl_mem), &d_result_pool1);
    checkError(status, "Setting maxpool1 arguments");

    global = 4;
    status = clEnqueueNDRangeKernel(queue, maxpool, 1, NULL, &global, NULL, 0, NULL, NULL);
    checkError(status, "Enqueueing maxpool1");

    status = clFinish(queue);
    checkError(status, "Wait for maxpool1 finish");

    double end_time = getCurrentTimestamp();

    time_conv1.push_back((end_time - start_time) * 1e3);

    printf("\r\nconv1 takes: %0.3f ms\r\n", (end_time - start_time) * 1e3);
    total += (end_time - start_time);
    start_time = end_time;

    /**************************************************************/
  /*                        conv2                            */
  /**************************************************************/

    input_channel = 4;
    input_size = 2046;
    pad = 0;
    stride = 1;
    start_channel = 0;
    output_size = 2042;
    filter_length = 5;
    filter_height = 1;


    status |= clSetKernelArg(conv3x3, 0, sizeof(int), &(input_channel));
    status |= clSetKernelArg(conv3x3, 1, sizeof(int), &(input_size));
    status |= clSetKernelArg(conv3x3, 2, sizeof(int), &(pad));
    status |= clSetKernelArg(conv3x3, 3, sizeof(int), &(stride));
    status |= clSetKernelArg(conv3x3, 4, sizeof(int), &(start_channel));
    status |= clSetKernelArg(conv3x3, 5, sizeof(int), &(output_size));
    status |= clSetKernelArg(conv3x3, 6, sizeof(int), &(filter_length));
    status |= clSetKernelArg(conv3x3, 7, sizeof(int), &(filter_height));
    status |= clSetKernelArg(conv3x3, 8, sizeof(cl_mem), &d_result_pool1);
    status |= clSetKernelArg(conv3x3, 9, sizeof(cl_mem), &d_conv2_weight);
    status |= clSetKernelArg(conv3x3, 10, sizeof(cl_mem), &d_conv2_bias);
    status |= clSetKernelArg(conv3x3, 11, sizeof(cl_mem), &d_result_conv2);


    checkError(status, "Setting conv2 arguments");



    global = 4;
    status = clEnqueueNDRangeKernel(queue, conv3x3, 1, NULL, &global, NULL, 0, NULL, NULL);
    checkError(status, "Enqueueing conv1: conv3x3");


    input_size = 2042;
    output_size = 1021;


    status |= clSetKernelArg(maxpool, 0, sizeof(int), &(input_size));
    status |= clSetKernelArg(maxpool, 1, sizeof(int), &(output_size));
    status |= clSetKernelArg(maxpool, 2, sizeof(cl_mem), &d_result_conv2);
    status |= clSetKernelArg(maxpool, 3, sizeof(cl_mem), &d_result_pool2);
    checkError(status, "Setting maxpool1 arguments");

    global = 4;
    status = clEnqueueNDRangeKernel(queue, maxpool, 1, NULL, &global, NULL, 0, NULL, NULL);
    checkError(status, "Enqueueing maxpool2");

    status = clFinish(queue);
    checkError(status, "Wait for maxpool2 finish");

    end_time = getCurrentTimestamp();

    time_conv2.push_back((end_time - start_time) * 1e3);
    printf("\r\nconv2 takes: %0.3f ms\r\n", (end_time - start_time) * 1e3);
    total += (end_time - start_time);
    start_time = end_time;


    /**************************************************************/
  /*                        conv3                           */
  /**************************************************************/


    input_channel = 4;
    input_size = 1021;
    pad = 0;
    stride = 1;
    start_channel = 0;
    output_size = 1018;
    filter_length = 4;
    filter_height = 1;


    status |= clSetKernelArg(conv3x3, 0, sizeof(int), &(input_channel));
    status |= clSetKernelArg(conv3x3, 1, sizeof(int), &(input_size));
    status |= clSetKernelArg(conv3x3, 2, sizeof(int), &(pad));
    status |= clSetKernelArg(conv3x3, 3, sizeof(int), &(stride));
    status |= clSetKernelArg(conv3x3, 4, sizeof(int), &(start_channel));
    status |= clSetKernelArg(conv3x3, 5, sizeof(int), &(output_size));
    status |= clSetKernelArg(conv3x3, 6, sizeof(int), &(filter_length));
    status |= clSetKernelArg(conv3x3, 7, sizeof(int), &(filter_height));
    status |= clSetKernelArg(conv3x3, 8, sizeof(cl_mem), &d_result_pool2);
    status |= clSetKernelArg(conv3x3, 9, sizeof(cl_mem), &d_conv3_weight);
    status |= clSetKernelArg(conv3x3, 10, sizeof(cl_mem), &d_conv3_bias);
    status |= clSetKernelArg(conv3x3, 11, sizeof(cl_mem), &d_result_conv3);


    checkError(status, "Setting conv3 arguments");



    global = 10;
    status = clEnqueueNDRangeKernel(queue, conv3x3, 1, NULL, &global, NULL, 0, NULL, NULL);
    checkError(status, "Enqueueing conv3: conv3x3");

    input_size = 1018;
    output_size = 509;


    status |= clSetKernelArg(maxpool, 0, sizeof(int), &(input_size));
    status |= clSetKernelArg(maxpool, 1, sizeof(int), &(output_size));
    status |= clSetKernelArg(maxpool, 2, sizeof(cl_mem), &d_result_conv3);
    status |= clSetKernelArg(maxpool, 3, sizeof(cl_mem), &d_result_pool3);
    checkError(status, "Setting maxpool3 arguments");

    global = 10;
    status = clEnqueueNDRangeKernel(queue, maxpool, 1, NULL, &global, NULL, 0, NULL, NULL);
    checkError(status, "Enqueueing maxpool3");

    status = clFinish(queue);
    checkError(status, "Wait for maxpool3 finish");

    end_time = getCurrentTimestamp();

    time_conv3.push_back((end_time - start_time) * 1e3);
    printf("\r\nconv3 takes: %0.3f ms\r\n", (end_time - start_time) * 1e3);
    total += (end_time - start_time);
    start_time = end_time;

    /**************************************************************/
  /*                        conv4                           */
  /**************************************************************/


    input_channel = 10;
    input_size = 509;
    pad = 0;
    stride = 1;
    start_channel = 0;
    output_size = 506;
    filter_length = 4;
    filter_height = 1;


    status |= clSetKernelArg(conv3x3, 0, sizeof(int), &(input_channel));
    status |= clSetKernelArg(conv3x3, 1, sizeof(int), &(input_size));
    status |= clSetKernelArg(conv3x3, 2, sizeof(int), &(pad));
    status |= clSetKernelArg(conv3x3, 3, sizeof(int), &(stride));
    status |= clSetKernelArg(conv3x3, 4, sizeof(int), &(start_channel));
    status |= clSetKernelArg(conv3x3, 5, sizeof(int), &(output_size));
    status |= clSetKernelArg(conv3x3, 6, sizeof(int), &(filter_length));
    status |= clSetKernelArg(conv3x3, 7, sizeof(int), &(filter_height));
    status |= clSetKernelArg(conv3x3, 8, sizeof(cl_mem), &d_result_pool3);
    status |= clSetKernelArg(conv3x3, 9, sizeof(cl_mem), &d_conv4_weight);
    status |= clSetKernelArg(conv3x3, 10, sizeof(cl_mem), &d_conv4_bias);
    status |= clSetKernelArg(conv3x3, 11, sizeof(cl_mem), &d_result_conv4);


    checkError(status, "Setting conv4 arguments");



    global = 10;
    status = clEnqueueNDRangeKernel(queue, conv3x3, 1, NULL, &global, NULL, 0, NULL, NULL);
    checkError(status, "Enqueueing conv4: conv3x3");





    input_size = 506;
    output_size = 253;


    status |= clSetKernelArg(maxpool, 0, sizeof(int), &(input_size));
    status |= clSetKernelArg(maxpool, 1, sizeof(int), &(output_size));
    status |= clSetKernelArg(maxpool, 2, sizeof(cl_mem), &d_result_conv4);
    status |= clSetKernelArg(maxpool, 3, sizeof(cl_mem), &d_result_pool4);
    checkError(status, "Setting maxpool4 arguments");

    global = 10;
    status = clEnqueueNDRangeKernel(queue, maxpool, 1, NULL, &global, NULL, 0, NULL, NULL);
    checkError(status, "Enqueueing maxpool4");

    status = clFinish(queue);
    checkError(status, "Wait for maxpool4 finish");

    end_time = getCurrentTimestamp();

    time_conv4.push_back((end_time - start_time) * 1e3);
    printf("\r\nconv4 takes: %0.3f ms\r\n", (end_time - start_time) * 1e3);
    total += (end_time - start_time);
    start_time = end_time;


    /**************************************************************/
  /*                        conv5                           */
  /**************************************************************/


    input_channel = 10;
    input_size = 253;
    pad = 0;
    stride = 1;
    start_channel = 0;
    output_size = 250;
    filter_length = 4;
    filter_height = 1;


    status |= clSetKernelArg(conv3x3, 0, sizeof(int), &(input_channel));
    status |= clSetKernelArg(conv3x3, 1, sizeof(int), &(input_size));
    status |= clSetKernelArg(conv3x3, 2, sizeof(int), &(pad));
    status |= clSetKernelArg(conv3x3, 3, sizeof(int), &(stride));
    status |= clSetKernelArg(conv3x3, 4, sizeof(int), &(start_channel));
    status |= clSetKernelArg(conv3x3, 5, sizeof(int), &(output_size));
    status |= clSetKernelArg(conv3x3, 6, sizeof(int), &(filter_length));
    status |= clSetKernelArg(conv3x3, 7, sizeof(int), &(filter_height));
    status |= clSetKernelArg(conv3x3, 8, sizeof(cl_mem), &d_result_pool4);
    status |= clSetKernelArg(conv3x3, 9, sizeof(cl_mem), &d_conv5_weight);
    status |= clSetKernelArg(conv3x3, 10, sizeof(cl_mem), &d_conv5_bias);
    status |= clSetKernelArg(conv3x3, 11, sizeof(cl_mem), &d_result_conv5);
    checkError(status, "Setting conv5 arguments");

    global = 15;
    status = clEnqueueNDRangeKernel(queue, conv3x3, 1, NULL, &global, NULL, 0, NULL, NULL);
    checkError(status, "Enqueueing conv5: conv3x3");


    input_size = 250;
    output_size = 125;


    status |= clSetKernelArg(maxpool, 0, sizeof(int), &(input_size));
    status |= clSetKernelArg(maxpool, 1, sizeof(int), &(output_size));
    status |= clSetKernelArg(maxpool, 2, sizeof(cl_mem), &d_result_conv5);
    status |= clSetKernelArg(maxpool, 3, sizeof(cl_mem), &d_result_pool5);
    checkError(status, "Setting maxpool5 arguments");

    global = 15;
    status = clEnqueueNDRangeKernel(queue, maxpool, 1, NULL, &global, NULL, 0, NULL, NULL);
    checkError(status, "Enqueueing maxpool5");

    status = clFinish(queue);
    checkError(status, "Wait for maxpool5 finish");

    end_time = getCurrentTimestamp();
    time_conv5.push_back((end_time - start_time) * 1e3);
    printf("\r\nconv5 takes: %0.3f ms\r\n", (end_time - start_time) * 1e3);
    total += (end_time - start_time);
    start_time = end_time;



    /**************************************************************/
  /*                        conv6                           */
  /**************************************************************/


    input_channel = 15;
    input_size = 125;
    pad = 0;
    stride = 1;
    start_channel = 0;
    output_size = 120;
    filter_length = 6;
    filter_height = 1;


    status |= clSetKernelArg(conv3x3, 0, sizeof(int), &(input_channel));
    status |= clSetKernelArg(conv3x3, 1, sizeof(int), &(input_size));
    status |= clSetKernelArg(conv3x3, 2, sizeof(int), &(pad));
    status |= clSetKernelArg(conv3x3, 3, sizeof(int), &(stride));
    status |= clSetKernelArg(conv3x3, 4, sizeof(int), &(start_channel));
    status |= clSetKernelArg(conv3x3, 5, sizeof(int), &(output_size));
    status |= clSetKernelArg(conv3x3, 6, sizeof(int), &(filter_length));
    status |= clSetKernelArg(conv3x3, 7, sizeof(int), &(filter_height));
    status |= clSetKernelArg(conv3x3, 8, sizeof(cl_mem), &d_result_pool5);
    status |= clSetKernelArg(conv3x3, 9, sizeof(cl_mem), &d_conv6_weight);
    status |= clSetKernelArg(conv3x3, 10, sizeof(cl_mem), &d_conv6_bias);
    status |= clSetKernelArg(conv3x3, 11, sizeof(cl_mem), &d_result_conv6);   //8*1*120

    checkError(status, "Setting conv6 arguments");

    global = 8;
    status = clEnqueueNDRangeKernel(queue, conv3x3, 1, NULL, &global, NULL, 0, NULL, &event);
    checkError(status, "Enqueueing conv6: conv3x3");

    end_time = getCurrentTimestamp();

    time_conv6.push_back((end_time - start_time) * 1e3);
    printf("\r\nconv5 takes: %0.3f ms\r\n", (end_time - start_time) * 1e3);
    total += (end_time - start_time);
    start_time = end_time;

    /**************************************************************/
  /*                        global Average                       */
  /**************************************************************/

    input_size = 120;

    status |= clSetKernelArg(avgGlobal, 0, sizeof(int), &(input_size));
    status |= clSetKernelArg(avgGlobal, 1, sizeof(cl_mem), &d_result_conv6);   //8*1*120
    status |= clSetKernelArg(avgGlobal, 2, sizeof(cl_mem), &d_result_global_average);  //8*1*1
    checkError(status, "Setting avgGlobal arguments");

    global = 8;
    status = clEnqueueNDRangeKernel(queue, avgGlobal, 1, NULL, &global, NULL, 0, NULL, NULL);
    checkError(status, "Enqueueing avgGlobal");

    status = clFinish(queue);
    checkError(status, "Wait for avgGlobal finish");




    //status = clEnqueueReadBuffer(queue, d_result_global_average, CL_TRUE, 0, sizeof(float) * 8, h_result_global_average, 0, NULL, NULL);

    end_time = getCurrentTimestamp();

    time_globle_average.push_back((end_time - start_time) * 1e3);
    printf("\r\n global Average takes: %0.3f ms\r\n", (end_time - start_time) * 1e3);
    total += (end_time - start_time);
    start_time = end_time;
    /**************************************************************/
   /*                       FC 1                                  */
  /**************************************************************/



    unsigned int matrix_1_height, matrix_P, matrix_2_length;

    //M×P，P×N
    matrix_1_height = 1;  //M
    matrix_P = 8;  //P
    matrix_2_length = 50; //N

    status |= clSetKernelArg(fc_matrix_mult, 0, sizeof(int), &(matrix_1_height));
    status |= clSetKernelArg(fc_matrix_mult, 1, sizeof(int), &(matrix_P));
    status |= clSetKernelArg(fc_matrix_mult, 2, sizeof(int), &(matrix_2_length));
    status |= clSetKernelArg(fc_matrix_mult, 3, sizeof(cl_mem), &d_result_global_average); //input matrix_1   1*8
    status |= clSetKernelArg(fc_matrix_mult, 4, sizeof(cl_mem), &d_fc1_weight);            //input matrix_2   8*50
    status |= clSetKernelArg(fc_matrix_mult, 5, sizeof(cl_mem), &d_fc1_bias);               //input bias   50个
    status |= clSetKernelArg(fc_matrix_mult, 6, sizeof(cl_mem), &d_result_fc1);             //output matrix_3   1*50
    checkError(status, "Setting fc1_matrix_mult arguments");

    size_t global_fc[2];
    global_fc[0] = 1;
    global_fc[1] = 50;


    status = clEnqueueNDRangeKernel(queue, fc_matrix_mult, 2, NULL, global_fc, NULL, 0, NULL, NULL);
    checkError(status, "Enqueueing fc1_matrix_mult");

    status = clFinish(queue);
    checkError(status, "Wait for fc1_matrix_mult finish");



    /**************************************************************/
  /*                       FC 2                                 */
  /**************************************************************/



    //M×P，P×N
    matrix_1_height = 1;  //M
    matrix_P = 50;  //P
    matrix_2_length = 20; //N
    status |= clSetKernelArg(fc_matrix_mult, 0, sizeof(int), &(matrix_1_height));
    status |= clSetKernelArg(fc_matrix_mult, 1, sizeof(int), &(matrix_P));
    status |= clSetKernelArg(fc_matrix_mult, 2, sizeof(int), &(matrix_2_length));
    status |= clSetKernelArg(fc_matrix_mult, 3, sizeof(cl_mem), &d_result_fc1); //input matrix_1   1*8
    status |= clSetKernelArg(fc_matrix_mult, 4, sizeof(cl_mem), &d_fc2_weight);            //input matrix_2   8*50
    status |= clSetKernelArg(fc_matrix_mult, 5, sizeof(cl_mem), &d_fc2_bias);               //input bias   50个
    status |= clSetKernelArg(fc_matrix_mult, 6, sizeof(cl_mem), &d_result_fc2);             //output matrix_3   1*50
    checkError(status, "Setting fc2_matrix_mult arguments");


    global_fc[0] = 1;
    global_fc[1] = 20;


    status = clEnqueueNDRangeKernel(queue, fc_matrix_mult, 2, NULL, global_fc, NULL, 0, NULL, NULL);
    checkError(status, "Enqueueing fc2_matrix_mult");

    status = clFinish(queue);
    checkError(status, "Wait for fc2_matrix_mult finish");

    /**************************************************************/
  /*                       FC 3                                  */
  /**************************************************************/



    //M×P，P×N
    matrix_1_height = 1;  //M
    matrix_P = 20;  //P
    matrix_2_length = 3; //N

    status |= clSetKernelArg(fc_matrix_mult, 0, sizeof(int), &(matrix_1_height));
    status |= clSetKernelArg(fc_matrix_mult, 1, sizeof(int), &(matrix_P));
    status |= clSetKernelArg(fc_matrix_mult, 2, sizeof(int), &(matrix_2_length));
    status |= clSetKernelArg(fc_matrix_mult, 3, sizeof(cl_mem), &d_result_fc2); //input matrix_1  
    status |= clSetKernelArg(fc_matrix_mult, 4, sizeof(cl_mem), &d_fc3_weight);            //input matrix_2  
    status |= clSetKernelArg(fc_matrix_mult, 5, sizeof(cl_mem), &d_fc3_bias);               //input bias  
    status |= clSetKernelArg(fc_matrix_mult, 6, sizeof(cl_mem), &d_result_fc3);             //output matrix_3 
    checkError(status, "Setting fc3_matrix_mult arguments");


    global_fc[0] = 1;
    global_fc[1] = 3;


    status = clEnqueueNDRangeKernel(queue, fc_matrix_mult, 2, NULL, global_fc, NULL, 0, NULL, NULL);
    checkError(status, "Enqueueing fc3_matrix_mult");





    status = clFinish(queue);
    checkError(status, "Wait for fc3_matrix_mult finish");

    end_time = getCurrentTimestamp();

    time_FC.push_back((end_time - start_time) * 1e3);
    printf("\r\n FC takes: %0.3f ms\r\n", (end_time - start_time) * 1e3);
    total += (end_time - start_time);
    start_time = end_time;




    ///**************************************************************/
    ///*                       softmax  and classifier             */
    ///**************************************************************/

    //读取d_result_fc3数据到主机
    float* h_result_fc3 = (float*)malloc((3) * sizeof(float));

    float* h_result_softmax = (float*)malloc((3) * sizeof(float));

    status = clEnqueueReadBuffer(queue, d_result_fc3, CL_TRUE, 0, sizeof(float) * 3, h_result_fc3, 0, NULL, NULL);

    softmax(h_result_fc3, 3, h_result_softmax);

    int maxindex;

    maxindex = classifier(h_result_softmax, 3);

    //printf("最大类别序号是%d", maxindex);



    return maxindex;


}

void cleanup() {
    clReleaseMemObject(d_sample);
    clReleaseMemObject(d_conv1_weight);
    clReleaseMemObject(d_conv1_bias);
    clReleaseMemObject(d_result_conv1);
    clReleaseMemObject(d_result_pool1);

    clReleaseMemObject(d_conv2_weight);
    clReleaseMemObject(d_conv2_bias);
    clReleaseMemObject(d_result_conv2);
    clReleaseMemObject(d_result_pool2);

    clReleaseMemObject(d_conv3_weight);
    clReleaseMemObject(d_conv3_bias);
    clReleaseMemObject(d_result_conv3);
    clReleaseMemObject(d_result_pool3);

    clReleaseMemObject(d_conv4_weight);
    clReleaseMemObject(d_conv4_bias);
    clReleaseMemObject(d_result_conv4);
    clReleaseMemObject(d_result_pool4);

    clReleaseMemObject(d_conv5_weight);
    clReleaseMemObject(d_conv5_bias);
    clReleaseMemObject(d_result_conv5);
    clReleaseMemObject(d_result_pool5);

    clReleaseMemObject(d_conv6_weight);
    clReleaseMemObject(d_conv6_bias);
    clReleaseMemObject(d_result_conv6);
    clReleaseMemObject(d_result_pool6);

    clReleaseMemObject(d_result_global_average);

    clReleaseMemObject(d_result_fc1);
    clReleaseMemObject(d_fc1_weight);
    clReleaseMemObject(d_fc1_bias);

    clReleaseMemObject(d_result_fc2);
    clReleaseMemObject(d_fc2_weight);
    clReleaseMemObject(d_fc2_bias);

    clReleaseMemObject(d_result_fc3);
    clReleaseMemObject(d_fc3_weight);
    clReleaseMemObject(d_fc3_bias);



    clReleaseKernel(conv3x3);
    clReleaseKernel(maxpool);
    clReleaseKernel(avgGlobal);
    clReleaseKernel(fc_matrix_mult);

    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);
}

double calculateAverage(std::vector<float> vec) {
    float sum = 0;
    for (int i = 0; i < vec.size(); i++) {
        sum += vec[i];
    }

    double average = static_cast<double>(sum) / vec.size();
    return average;
}

int main() {

     //const 变量可以使用指针改变其值
      //char filenametxt[] = "F:/VS2022/Prj/oneDCNN/oneDCNN/epileptic EEG/B_O/O001.txt";   //0
      //char filenametxt[] = "F:/VS2022/Prj/oneDCNN/oneDCNN/epileptic EEG/D_F/F001.txt";   //1
        char filenametxt[] = "F:/VS2022/Prj/oneDCNN/oneDCNN/epileptic EEG/E_S/S001.txt";   //2

      int classindex = 2;


     //filenametxt[51] = (char)( 8 + '0');
     //printf("filenametxt: %c\n", filenametxt[0]);   // char>string
     
     char* new_filenametxt;

     // 分配内存
     int* result = (int*)malloc(sizeof(int) * 100);
     new_filenametxt = (char*)malloc(strlen(filenametxt) + 1); // +1 是为了容纳字符串的结束字符 '\0'

     if (new_filenametxt == NULL) {
         printf("Memory allocation failed.\n");
         return 1; // 返回非零值表示程序出错
     }


     for (int i = 0; i < 100; i++) {

        if (i < 9) {  //0-8
             filenametxt[51] = (char)(i+1 + '0');
             
         }
         else if(i<99 and i>=9) {   // 9-98
               
             filenametxt[50] = (char)((i + 1 )/10 + '0');
             filenametxt[51] = (char)((i + 1 )%10 + '0');

         }
         else {          //99

            filenametxt[49] = (char)(1+ '0');
            filenametxt[50] = (char)(0 + '0');
            filenametxt[51] = (char)(0 + '0');
        
        }


        // 复制字符串
        strcpy(new_filenametxt, filenametxt);

        printf("New String: %s\n", new_filenametxt);
         result[i] = oneDCNN(new_filenametxt);
         printf("index %d result is %d\n",i+1, result[i]);

     }


//计算平均时间



      


     double average_conv1 = calculateAverage(time_conv1);
     double average_conv2 = calculateAverage(time_conv2);
     double average_conv3 = calculateAverage(time_conv3);
     double average_conv4 = calculateAverage(time_conv4);
     double average_conv5 = calculateAverage(time_conv5);
     double  average_conv6= calculateAverage(time_conv6);
     double  average_globle= calculateAverage(time_globle_average);
     double average_FC = calculateAverage(time_FC);
     

     printf("average_conv1=%0.3f\n", average_conv1);
     printf("average_conv2=%0.3f\n", average_conv2);
     printf("average_conv3=%0.3f\n", average_conv3);
     printf("average_conv4=%0.3f\n", average_conv4);
     printf("average_conv5=%0.3f\n", average_conv5);
     printf("average_conv6=%0.3f\n", average_conv6);
     printf("average_globle=%0.3f\n", average_globle);
     printf("average_FC=%0.3f\n", average_FC);




     //计算精度

     int count=0;
     double accurary;
     
     for (int i = 0; i < 100; i++) {
     
         if (result[i] == classindex) {
             count++;
         }
     
     }


    /* for (int i = 0; i < 100; i++) {

         printf("result index %d: %d \n ", i,result[i]);
        
     }*/

     accurary = count / 100.0;
     printf("accurary is %f\n", accurary);



     free(new_filenametxt);
    free(result);

    cleanup();

    return 0;
}



