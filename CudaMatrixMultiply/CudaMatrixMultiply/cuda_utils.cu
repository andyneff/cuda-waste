#include "stdafx.h"

cudaError_t Check_CUDA_Error(cudaError_t err, const char * msg = "")
{
    if( cudaSuccess != err)
    {
        std::cout << "Cuda error: " << msg << ": " << cudaGetErrorString(err) << std::endl;
    }
    return err;
}


static void**list = (void**)malloc(1000);
static int list_ptr = 0;

bool Is_In_Device_Space(void * ptr)
{
	for (int i = 0; i < list_ptr; ++i)
		if (list[i] == ptr)
			return true;
	return false;
}

void Add_Ptr_In_Device_Space(void * ptr)
{
	for (int i = 0; i < list_ptr; ++i)
		if (list[i] == ptr)
			return;
	list[list_ptr++] = ptr;
}

void Remove_Ptr_In_Device_Space(void * ptr)
{
	for (int i = 0; i < list_ptr; ++i)
		if (list[i] == ptr)
		{
			// remove i'th pointer.
			for (int j = i; j < list_ptr; ++i)
			{
				list[j] = list[j+1];
			}
			list_ptr--;
			return;
		}
}
