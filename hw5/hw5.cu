#include <iostream>

__global__ void fun(float * d, int size)
{
	extern __shared__ int buf[];
    int idx = threadIdx.x +
            blockDim.x * blockIdx.x
            + blockDim.x * gridDim.x * blockDim.y * blockIdx.y
            + blockDim.x * gridDim.x * threadIdx.y;
    if (idx < 0)
        return;
    if (idx >= size)
        return;
	buf[idx] = idx;
	__syncthreads();
    d[idx] = buf[(idx+50) % size] * 10.0 / 0.1;
}

int main()
{
    float * h;
    float * d;
    int size = 100;
	int shared = 400;
    h = (float*)malloc(size*sizeof(float));
    cudaMalloc(&d, size*sizeof(float));
    cudaMemcpy(d, h, size*sizeof(float), cudaMemcpyHostToDevice);
    fun<<<1, size, shared>>>(d, size);
    cudaThreadSynchronize();
    int rv = cudaGetLastError();
    cudaMemcpy(h, d, size*sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; ++i)
        std::cout << "Result = " << h[i] << "\n";
    return 0;
}
