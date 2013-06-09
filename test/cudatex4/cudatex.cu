#include <iostream>
#include <stdio.h>

void CUDA_CALL(cudaError_t result)
{
	if (result != cudaSuccess)
	{
		std::cout << "Error " << result << "\n";
		exit(1);
	}
}

texture<float, 1, cudaReadModeElementType> texRef1;


// Simple transformation kernel
__global__ void transformKernel(float* output_x, float * output_y, int width, int height, int output_width, int output_height, float theta)
{
	// Calculate normalized texture coordinates
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
     
	float u = x / (float)width;
	float v = y / (float)height;

	float tu = ((float)width * x) / output_width;
	float tv = ((float)height * y) / output_height;

	output_x[y * output_width + x] = tu;
	output_y[y * output_width + x] = tex1D(texRef1, tu);
}

// Host code
int main()
{
	int width = 10;
	int height = 1;
	int size = width * height;

	float * h_data = (float*) malloc(width * height * sizeof(float));
	for (int i = 0 ; i < height; i++)
		for (int j = 0; j < width; j++)
				h_data[i * width + j] = i * width + j + 1;

	printf ("\n Original array \n");
	for (int i = 0; i < height; i++)
	{
		for (int j = 0 ; j < width; j++)
			printf ("%f ", h_data[i*width + j]);
		printf ("\n");
	}
          
	// create a CUDA array on the device
	cudaArray* cuArray;
	CUDA_CALL( cudaMallocArray(&cuArray, &texRef1.channelDesc, size, height) );
	CUDA_CALL( cudaMemcpyToArray(cuArray, 0, 0, h_data, sizeof(float) * size, cudaMemcpyHostToDevice) );
   
	// bind a texture to the CUDA array
	cudaBindTextureToArray (texRef1, cuArray);
 
	// Set texture access
	texRef1.normalized = false;
	texRef1.filterMode = cudaFilterModeLinear;
        
	int output_width = 100;
	int output_height = 1;

	// Set up block dims.
	dim3 dimBlock(10, 1);
	dim3 dimGrid((output_width - 1)/dimBlock.x + 1, (output_height - 1)/dimBlock.y + 1);
     
	printf("blockDim = %d, %d\n", dimBlock.x, dimBlock.y);
	printf("gridDim = %d, %d\n", dimGrid.x, dimGrid.y);

	// Allocate result of transformation in device memory
	int output_size =  dimGrid.x * dimBlock.x * dimGrid.y * dimBlock.y;
	printf("output_size %d\n", output_size);
	float* d_output_x;
	float* d_output_y;
	CUDA_CALL( cudaMalloc(&d_output_x, output_size * sizeof(float)) );
	CUDA_CALL( cudaMalloc(&d_output_y, output_size * sizeof(float)) );
	float * h_output_x = (float*)malloc(output_size * sizeof(float));
	float * h_output_y = (float*)malloc(output_size * sizeof(float));

	transformKernel<<<dimGrid, dimBlock>>>(d_output_x, d_output_y, width, height, output_width, output_height, 90);
    cudaDeviceSynchronize();

	CUDA_CALL( cudaMemcpy(h_output_x, d_output_x, output_size * sizeof(float), cudaMemcpyDeviceToHost) );
	CUDA_CALL( cudaMemcpy(h_output_y, d_output_y, output_size * sizeof(float), cudaMemcpyDeviceToHost) );

	printf ("\nAfter operation\n");
	for (int i = 0; i < output_height; i++)
	{
		for (int j = 0; j < output_width; j++)
			printf ("%f %f\n", h_output_x[i*output_width + j], h_output_y[i*output_width + j]);
		printf ("\n");
	}

	system ("pause");
	// Free device memory
	cudaFree(d_output_x);
	cudaFree(d_output_y);
	return 0;
}

