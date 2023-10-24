
__kernel void matrix_mult(

     const int Mdim,   //M1's height
	 const int Pdim,
	 const int Ndim,   //M2's length
    
    
    __global const float* A, 
    __global const float* B, 
	__global const float* bias,
    __global float* C      )

{
    int i = get_global_id(0);  //1
    int j = get_global_id(1);  //50

    int k;
    float tmp;

    if ((i < Mdim) && (j < Ndim)) {
        tmp = 0.0;
        for (k = 0; k < Pdim; k++)
            tmp += A[i*Pdim + k] *  B[j*Pdim+k];  // B[k*Ndim + j]
        C[i*Mdim + j] = tmp + bias[j];
    }
}



__kernel void average2d(

    const int input_size,
	 
	__global float* input_im,    //8*1*120
	__global float *restrict output_im    //8*1*1
	
	)
{
	int channel_index = get_global_id(0);//get class score index

	input_im += input_size * channel_index;
	
	float tmp = 0.0f;

	for(int i = 0; i < input_size; i++)
	{
		tmp += input_im[i];
	}

	output_im[channel_index] = tmp / input_size;
}


//maxPool2d 
//kernel_size=3 stride=2
//output one feature map per kernel
__kernel void maxpool2d(
	const int input_size,
	const int output_size,

	__global float *input_im,
    __global float *restrict output_im)
{
	int channels = get_global_id(0);//get output channel index
	
	input_im += channels * 1 * input_size;
	output_im += channels * 1 * output_size;

	//loop over output feature map
	for(int i = 0; i < 1; i++)//row
	{
		for(int j = 0; j < output_size; j++)//col
		{
			//find the max value in 3x3 reigon 
			//to be one element in the output feature map
			float tmp = 0.0;

			#pragma unroll 1
			for(int k = 0; k < 1; k++)//row
			{
				#pragma unroll
				for(int l = 0; l < 2; l++)//col
				{
					float value = input_im[(i * 2 + k) * input_size  + j * 2 + l ]; //取值
					if(value > tmp)
						tmp = value;
				}
			}
			//store the result to output feature map
			*output_im = tmp; 
			output_im++;      //输出格式：[第一层][第二层]。。。                       
		}
	}
}

//3x3 convolution layer
//output one feature map per kernel
__kernel void conv2d3x3(
	const int input_channel, 
	const int input_size,
	const int pad, 
	const int stride,
	const int start_channel, //start_channel is for 1x1 feature map in fire layer
	const int output_size,
	const int filter_length,
	const int filter_height,
	__global float* input_im,  //1*1*4097
	__global const float* filter_weight,   //(1*1*6)*4
	__global const float* filter_bias,     //
	__global float *restrict output_im      //4*1*4092

	)
{
	int filter_index = get_global_id(0); //get output channel index  from 0 to 3

	filter_weight += filter_index * input_channel * filter_length ;   //每个id（工作项）对应输出的权重的起始位置,一个卷积核的参数量
	float bias = filter_bias[filter_index];                  //每个id（工作项）对应的偏置
	output_im += (start_channel + filter_index) * 1 * output_size;  //每个id（工作项）对应运行的输出起始位置
	
	//loop over output feature map
	for(int i = 0; i < 1; i++)   //高度偏移
	{
		for(int j = 0; j < output_size; j++)   //宽度偏移，4092
		{
			//compute one element in the output feature map
			float tmp = bias;
			
			//compute dot product of 2 input_channel x 3 x 3 matrix    卷积核内部偏移
			for(int k = 0; k < input_channel; k++)   // 卷积深度
			{
				for(int l = 0; l < filter_height; l++)    //卷积高
				{
					int h = i * stride + l - pad;     //在全局中的高度，一直为0，输出位置与输入位置的映射关系

					#pragma unroll
					for(int m = 0; m < filter_length; m++)   //卷积宽
					{
						int w = j * stride + m - pad;    //在全局中的宽度，输出位置与输入位置的映射关系

						if((h >= 0) && (h < 1) && (w >= 0) && (w < input_size))
						{
							tmp += input_im[k * 1 * input_size + (i * stride + l - pad) * input_size + j * stride + m - pad]  *  filter_weight[filter_length *filter_height* k  + m];
						}
					}
				}
			}

			//add leaky  relu activation after conv

			if(input_size==125){
				*output_im = (tmp > 0.0) ? tmp : 0;
			    output_im++;  //指针加一
			}
			else{
				*output_im = (tmp > 0.0) ? tmp : 0.01*tmp;
			    output_im++;  //指针加一
			}
		
		}
	}
}

//1x1 convolution layer
//output one feature map per kernel
__kernel void conv2d1x1(
	const int input_channels, const int input_size,
	__global float *input_im,
	__global const float* filter_weight,
	__global const float* filter_bias,
	__global float *restrict output_im)
{
	int filter_index = get_global_id(0); // 0 - (output_channels - 1)

	filter_weight += filter_index * input_channels;

	float bias = filter_bias[filter_index];
	
	output_im += filter_index * input_size * input_size;//start_channel is for 1x1 feature map in fire layer

	//loop over output feature map
	//out
	for(int i = 0; i < input_size; i++)
	{
		for(int j = 0; j < input_size; j++)
		{
			float tmp = bias;

			#pragma unroll 6
			for(int k = 0; k < input_channels; k++)
			{
				tmp += input_im[k * input_size * input_size + i * input_size + j] * filter_weight[k];
			}
			//add relu after conv
			*output_im = (tmp > 0.0) ? tmp : 0.0;
			output_im++;
		}
	}
}

//last layer use a 13 x 13 avgPool layer as classifier
//one class score per kernel
__kernel void avgpool2d(
	__global float* input_im,
	__global float *restrict output_im)
{
	int class_index = get_global_id(0);//get class score index

	input_im += 169 * class_index;
	
	float tmp = 0.0f;

	for(int i = 0; i < 169; i++)
	{
		tmp += input_im[i];
	}

	output_im[class_index] = tmp / 169.0;
}
