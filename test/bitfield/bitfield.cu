#include <iostream>

struct fields
{
   unsigned a : 4;
   unsigned b : 4;
   unsigned c : 4;
   unsigned d : 4;
   unsigned e : 4;
   unsigned f : 4;
   unsigned g : 4;
   unsigned h : 4;
};

union u
{
	unsigned int i;
	fields f;
};

__device__ __forceinline__
unsigned int bfe(unsigned int x, unsigned int bit, unsigned int numBits) {
    unsigned int ret;
    asm volatile ("{ \n\t"
                  "    bfe.u32 %0, %1, %2, %3;\n\t"
	"} \n\t" : "=r"(ret) : "r"(x), "r"(bit), "r"(numBits));
    return ret;
}

__device__ __forceinline__
unsigned int c_extract(unsigned int in, int which)
{
    u x;
	x.i = in;
	if (which == 0)
		return x.f.a;
	else if (which == 1)
		return x.f.b;
	else if (which == 2)
		return x.f.c;
	else if (which == 3)
		return x.f.d;
	else return 0;
}

__device__ __forceinline__
unsigned int d_extract(unsigned int in, int which)
{
	if (which == 0)
		return bfe(in, 0, 4);
	else if (which == 1)
		return bfe(in, 4, 4);
	else if (which == 2)
		return bfe(in, 8, 4);
	else if (which == 3)
		return bfe(in, 12, 4);
	else return 0;
}


__global__ void fun_c(int * mem, int size)
{
	for (int i = 0; i < size; ++i)
	{
		mem[i] = c_extract(mem[i], 0);
	}
}

__global__ void fun_d(int * mem, int size)
{
	for (int i = 0; i < size; ++i)
	{
		mem[i] = d_extract(mem[i], 0);
	}
}

int main()
{
	u y;

	y.f.a = 0xf; std::cout << std::hex << y.i << "\n";
	y.f.b = 0xf; std::cout << std::hex << y.i << "\n";
	y.f.c = 0xf; std::cout << std::hex << y.i << "\n";
	y.f.d = 0xf; std::cout << std::hex << y.i << "\n";
	y.f.e = 0xf; std::cout << std::hex << y.i << "\n";
	y.f.f = 0xf; std::cout << std::hex << y.i << "\n";
	y.f.g = 0xf; std::cout << std::hex << y.i << "\n";
	y.f.h = 0xf; std::cout << std::hex << y.i << "\n";

	int * h;
	int * d;
	int n = 50;
	h = (int*)malloc(n * sizeof(int));
	cudaMalloc(&d, n * sizeof(int));
	{
		for (int i = 0; i < n; ++i)
			h[i] = i;
		cudaMemcpy(d, h, n * sizeof(int), cudaMemcpyHostToDevice);
		fun_c<<<1,1>>>(d, n);
		cudaThreadSynchronize();
		int rv = cudaGetLastError();
		cudaMemcpy(h, d, n * sizeof(int), cudaMemcpyDeviceToHost);
		std::cout << "Result:\n";
		for (int i = 0; i < n; ++i)
			std::cout << std::hex << h[i] << "\n";
	}
	{
		for (int i = 0; i < n; ++i)
			h[i] = i;
		cudaMemcpy(d, h, n * sizeof(int), cudaMemcpyHostToDevice);
		fun_d<<<1,1>>>(d, n);
		cudaThreadSynchronize();
		int rv = cudaGetLastError();
		cudaMemcpy(h, d, n * sizeof(int), cudaMemcpyDeviceToHost);
		std::cout << "Result:\n";
		for (int i = 0; i < n; ++i)
			std::cout << std::hex << h[i] << "\n";
	}
	return 0;
}
