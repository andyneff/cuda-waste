#include <iostream>

__global__ void fun(int * d, int size)
{
    int idx = threadIdx.x +
            blockDim.x * blockIdx.x
            + blockDim.x * gridDim.x * blockDim.y * blockIdx.y
            + blockDim.x * gridDim.x * threadIdx.y;
    if (idx < 0)
        return;
    if (idx >= size)
        return;
    d[idx] = idx;
}

int main()
{
    int * h;
    int * d;
    int size = 100;
    h = (int*)malloc(size*sizeof(int));
    cudaMalloc(&d, size*sizeof(int));
    cudaMemcpy(d, h, size*sizeof(int), cudaMemcpyHostToDevice);
    fun<<<1,size>>>(d, size);
    cudaThreadSynchronize();
    int rv = cudaGetLastError();
    cudaMemcpy(h, d, size*sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; ++i)
        std::cout << "Result = " << h[i] << "\n";
    return 0;
}
