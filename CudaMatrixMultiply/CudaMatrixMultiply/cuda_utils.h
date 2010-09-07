#ifndef _CUDA_UTILS_CUH
#define _CUDA_UTILS_CUH


cudaError_t Check_CUDA_Error(cudaError_t err, const char * msg);
bool Is_In_Device_Space(void * ptr);
void Add_Ptr_In_Device_Space(void * ptr);
void Remove_Ptr_In_Device_Space(void * ptr);

#endif
