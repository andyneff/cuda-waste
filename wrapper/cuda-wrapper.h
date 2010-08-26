/*
   Copyright 2010 Ken Domino

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#pragma once

#include <vector>
#include <cuda_runtime.h> // cudaError_t, CUDARTAPI, etc.

class HookManager;

#ifdef DLL_EXPORTS
#define DLL_API __declspec(dllexport)
#else
#define DLL_API __declspec(dllimport)
#endif

/// Wrapper for CUDA's dynamic link library.
#ifndef __dv
#define __dv(x)
#endif


class DLL_API CudaMemoryDebug
{
private:
	struct data
    {
        void * ptr;
        bool is_host;
        int size;
        char * context;
    };
	friend std::vector<data>;
private:
	CudaMemoryDebug();
	static CudaMemoryDebug * singleton;
	HookManager * hook_manager;
    size_t padding_size;
    unsigned char padding_byte;
    bool device_pointer_to_first_byte_in_block;
    std::ostream * output_stream;
    bool trace_all_calls;
    bool quit_on_error;
    bool do_not_call_cuda_after_sanity_check_fail;
	bool do_emulation;
	bool do_crash;
    bool init;
    static void ExitHandler();
    std::vector<data> alloc_list;
    static char * Context(int lines = 1);
    char * global_context;

    typedef cudaError_t (CUDARTAPI *typePtrCudaMalloc3D)(struct cudaPitchedPtr* pitchedDevPtr, struct cudaExtent extent);
    typedef cudaError_t (CUDARTAPI *typePtrCudaMalloc3DArray)(struct cudaArray** arrayPtr, const struct cudaChannelFormatDesc* desc, struct cudaExtent extent, unsigned int flags __dv(0));
    typedef cudaError_t (CUDARTAPI *typePtrCudaMemset3D)(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent);
    typedef cudaError_t (CUDARTAPI *typePtrCudaMemcpy3D)(const struct cudaMemcpy3DParms *p);
    typedef cudaError_t (CUDARTAPI *typePtrCudaMemcpy3DAsync)(const struct cudaMemcpy3DParms *p, cudaStream_t stream __dv(0));
    typedef cudaError_t (CUDARTAPI *typePtrCudaMalloc)(void **devPtr, size_t size);
    typedef cudaError_t (CUDARTAPI *typePtrCudaMallocHost)(void **ptr, size_t size);
    typedef cudaError_t (CUDARTAPI *typePtrCudaMallocPitch)(void **devPtr, size_t *pitch, size_t width, size_t height);
    typedef cudaError_t (CUDARTAPI *typePtrCudaMallocArray)(struct cudaArray **array, const struct cudaChannelFormatDesc *desc, size_t width, size_t height __dv(0), unsigned int flags __dv(0));
    typedef cudaError_t (CUDARTAPI *typePtrCudaFree)(void *devPtr);
    typedef cudaError_t (CUDARTAPI *typePtrCudaFreeHost)(void *ptr);
    typedef cudaError_t (CUDARTAPI *typePtrCudaFreeArray)(struct cudaArray *array);
    typedef cudaError_t (CUDARTAPI *typePtrCudaHostAlloc)(void **pHost, size_t bytes, unsigned int flags);
    typedef cudaError_t (CUDARTAPI *typePtrCudaHostGetDevicePointer)(void **pDevice, void *pHost, unsigned int flags);
    typedef cudaError_t (CUDARTAPI *typePtrCudaHostGetFlags)(unsigned int *pFlags, void *pHost);
    typedef cudaError_t (CUDARTAPI *typePtrCudaMemGetInfo)(size_t *free, size_t *total);
    typedef cudaError_t (CUDARTAPI *typePtrCudaMemcpy)(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
    typedef cudaError_t (CUDARTAPI *typePtrCudaMemcpyToArray)(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind);
    typedef cudaError_t (CUDARTAPI *typePtrCudaMemcpyFromArray)(void *dst, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind);
    typedef cudaError_t (CUDARTAPI *typePtrCudaMemcpyArrayToArray)(struct cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToDevice));
    typedef cudaError_t (CUDARTAPI *typePtrCudaMemcpy2D)(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
    typedef cudaError_t (CUDARTAPI *typePtrCudaMemcpy2DToArray)(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
    typedef cudaError_t (CUDARTAPI *typePtrCudaMemcpy2DFromArray)(void *dst, size_t dpitch, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind);
    typedef cudaError_t (CUDARTAPI *typePtrCudaMemcpy2DArrayToArray)(struct cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToDevice));
    typedef cudaError_t (CUDARTAPI *typePtrCudaMemcpyToSymbol)(const char *symbol, const void *src, size_t count, size_t offset __dv(0), enum cudaMemcpyKind kind __dv(cudaMemcpyHostToDevice));
    typedef cudaError_t (CUDARTAPI *typePtrCudaMemcpyFromSymbol)(void *dst, const char *symbol, size_t count, size_t offset __dv(0), enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToHost));
    typedef cudaError_t (CUDARTAPI *typePtrCudaMemcpyAsync)(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
    typedef cudaError_t (CUDARTAPI *typePtrCudaMemcpyToArrayAsync)(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
    typedef cudaError_t (CUDARTAPI *typePtrCudaMemcpyFromArrayAsync)(void *dst, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
    typedef cudaError_t (CUDARTAPI *typePtrCudaMemcpy2DAsync)(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
    typedef cudaError_t (CUDARTAPI *typePtrCudaMemcpy2DToArrayAsync)(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
    typedef cudaError_t (CUDARTAPI *typePtrCudaMemcpy2DFromArrayAsync)(void *dst, size_t dpitch, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
    typedef cudaError_t (CUDARTAPI *typePtrCudaMemcpyToSymbolAsync)(const char *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
    typedef cudaError_t (CUDARTAPI *typePtrCudaMemcpyFromSymbolAsync)(void *dst, const char *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
    typedef cudaError_t (CUDARTAPI *typePtrCudaMemset)(void *devPtr, int value, size_t count);
    typedef cudaError_t (CUDARTAPI *typePtrCudaMemset2D)(void *devPtr, size_t pitch, int value, size_t width, size_t height);
    typedef cudaError_t (CUDARTAPI *typePtrCudaGetSymbolAddress)(void **devPtr, const char *symbol);
    typedef cudaError_t (CUDARTAPI *typePtrCudaGetSymbolSize)(size_t *size, const char *symbol);
    typedef cudaError_t (CUDARTAPI *typePtrCudaGetDeviceCount)(int *count);
    typedef cudaError_t (CUDARTAPI *typePtrCudaGetDeviceProperties)(struct cudaDeviceProp *prop, int device);
    typedef cudaError_t (CUDARTAPI *typePtrCudaChooseDevice)(int *device, const struct cudaDeviceProp *prop);
    typedef cudaError_t (CUDARTAPI *typePtrCudaSetDevice)(int device);
    typedef cudaError_t (CUDARTAPI *typePtrCudaGetDevice)(int *device);
    typedef cudaError_t (CUDARTAPI *typePtrCudaSetValidDevices)(int *device_arr, int len);
    typedef cudaError_t (CUDARTAPI *typePtrCudaSetDeviceFlags)( int flags );
    typedef cudaError_t (CUDARTAPI *typePtrCudaBindTexture)(size_t *offset, const struct textureReference *texref, const void *devPtr, const struct cudaChannelFormatDesc *desc, size_t size __dv(UINT_MAX));
    typedef cudaError_t (CUDARTAPI *typePtrCudaBindTexture2D)(size_t *offset,const struct textureReference *texref,const void *devPtr, const struct cudaChannelFormatDesc *desc,size_t width, size_t height, size_t pitch);
    typedef cudaError_t (CUDARTAPI *typePtrCudaBindTextureToArray)(const struct textureReference *texref, const struct cudaArray *array, const struct cudaChannelFormatDesc *desc);
    typedef cudaError_t (CUDARTAPI *typePtrCudaUnbindTexture)(const struct textureReference *texref);
    typedef cudaError_t (CUDARTAPI *typePtrCudaGetTextureAlignmentOffset)(size_t *offset, const struct textureReference *texref);
    typedef cudaError_t (CUDARTAPI *typePtrCudaGetTextureReference)(const struct textureReference **texref, const char *symbol);
    typedef cudaError_t (CUDARTAPI *typePtrCudaBindSurfaceToArray)(const struct surfaceReference *surfref, const struct cudaArray *array, const struct cudaChannelFormatDesc *desc);
    typedef cudaError_t (CUDARTAPI *typePtrCudaGetSurfaceAlignmentOffset)(size_t *offset, const struct surfaceReference *surfref);
    typedef cudaError_t (CUDARTAPI *typePtrCudaGetSurfaceReference)(const struct surfaceReference **surfref, const char *symbol);
    typedef cudaError_t (CUDARTAPI *typePtrCudaGetChannelDesc)(struct cudaChannelFormatDesc *desc, const struct cudaArray *array);
    typedef struct cudaChannelFormatDesc (CUDARTAPI *typePtrCudaCreateChannelDesc)(int x, int y, int z, int w, enum cudaChannelFormatKind f);
    typedef cudaError_t (CUDARTAPI *typePtrCudaGetLastError)(void);
    typedef cudaError_t (CUDARTAPI *typePtrCudaPeekAtLastError)(void);
    typedef const char* (CUDARTAPI *typePtrCudaGetErrorString)(cudaError_t error);
    typedef cudaError_t (CUDARTAPI *typePtrCudaConfigureCall)(dim3 gridDim, dim3 blockDim, size_t sharedMem __dv(0), cudaStream_t stream __dv(0));
    typedef cudaError_t (CUDARTAPI *typePtrCudaSetupArgument)(const void *arg, size_t size, size_t offset);
    typedef cudaError_t (CUDARTAPI *typePtrCudaFuncSetCacheConfig)(const char *func, enum cudaFuncCache cacheConfig);
    typedef cudaError_t (CUDARTAPI *typePtrCudaLaunch)(const char *entry);
    typedef cudaError_t (CUDARTAPI *typePtrCudaFuncGetAttributes)(struct cudaFuncAttributes *attr, const char *func);
    typedef cudaError_t (CUDARTAPI *typePtrCudaStreamCreate)(cudaStream_t *pStream);
    typedef cudaError_t (CUDARTAPI *typePtrCudaStreamDestroy)(cudaStream_t stream);
    typedef cudaError_t (CUDARTAPI *typePtrCudaStreamSynchronize)(cudaStream_t stream);
    typedef cudaError_t (CUDARTAPI *typePtrCudaStreamQuery)(cudaStream_t stream);
    typedef cudaError_t (CUDARTAPI *typePtrCudaEventCreate)(cudaEvent_t *event);
    typedef cudaError_t (CUDARTAPI *typePtrCudaEventCreateWithFlags)(cudaEvent_t *event, int flags);
    typedef cudaError_t (CUDARTAPI *typePtrCudaEventRecord)(cudaEvent_t event, cudaStream_t stream __dv(0));
    typedef cudaError_t (CUDARTAPI *typePtrCudaEventQuery)(cudaEvent_t event);
    typedef cudaError_t (CUDARTAPI *typePtrCudaEventSynchronize)(cudaEvent_t event);
    typedef cudaError_t (CUDARTAPI *typePtrCudaEventDestroy)(cudaEvent_t event);
    typedef cudaError_t (CUDARTAPI *typePtrCudaEventElapsedTime)(float *ms, cudaEvent_t start, cudaEvent_t end);
    typedef cudaError_t (CUDARTAPI *typePtrCudaSetDoubleForDevice)(double *d);
    typedef cudaError_t (CUDARTAPI *typePtrCudaSetDoubleForHost)(double *d);
    typedef cudaError_t (CUDARTAPI *typePtrCudaThreadExit)(void);
    typedef cudaError_t (CUDARTAPI *typePtrCudaThreadSynchronize)(void);
    typedef cudaError_t (CUDARTAPI *typePtrCudaThreadSetLimit)(enum cudaLimit limit, size_t value);
    typedef cudaError_t (CUDARTAPI *typePtrCudaThreadGetLimit)(size_t *pValue, enum cudaLimit limit);
    typedef cudaError_t (CUDARTAPI *typePtrCudaDriverGetVersion)(int *driverVersion);
    typedef cudaError_t (CUDARTAPI *typePtrCudaRuntimeGetVersion)(int *runtimeVersion);
    typedef cudaError_t (CUDARTAPI *typePtrCudaGetExportTable)(const void **ppExportTable, const cudaUUID_t *pExportTableId);
    typedef cudaError_t (CUDARTAPI *typePtrCudaGraphicsUnregisterResource)(struct cudaGraphicsResource *resource);
    typedef cudaError_t (CUDARTAPI *typePtrCudaGraphicsResourceSetMapFlags)(struct cudaGraphicsResource *resource, unsigned int flags); 
    typedef cudaError_t (CUDARTAPI *typePtrCudaGraphicsMapResources)(int count, struct cudaGraphicsResource **resources, cudaStream_t stream __dv(0));
    typedef cudaError_t (CUDARTAPI *typePtrCudaGraphicsUnmapResources)(int count, struct cudaGraphicsResource **resources, cudaStream_t stream __dv(0));
    typedef cudaError_t (CUDARTAPI *typePtrCudaGraphicsResourceGetMappedPointer)(void **devPtr, size_t *size, struct cudaGraphicsResource *resource);
    typedef cudaError_t (CUDARTAPI *typePtrCudaGraphicsSubResourceGetMappedArray)(struct cudaArray **arrayPtr, struct cudaGraphicsResource *resource, unsigned int arrayIndex, unsigned int mipLevel);
    typedef cudaError_t (CUDARTAPI *typePtrCudaGraphicsSubResourceGetMappedArray)(struct cudaArray **arrayPtr, struct cudaGraphicsResource *resource, unsigned int arrayIndex, unsigned int mipLevel);
    typedef void** (CUDARTAPI *typePtrCudaRegisterFatBinary)(void *fatCubin);
    typedef void (CUDARTAPI *typePtrCudaUnregisterFatBinary)(void **fatCubinHandle);
    typedef void (CUDARTAPI *typePtrCudaRegisterVar)(void **fatCubinHandle, char *hostVar, char *deviceAddress, const char *deviceName, int ext, int size, int constant, int global);
    typedef void (CUDARTAPI *typePtrCudaRegisterTexture)(void **fatCubinHandle, const struct textureReference *hostVar, const void **deviceAddress, const char *deviceName, int dim, int norm, int ext);
    typedef void (CUDARTAPI *typePtrCudaRegisterSurface)(void **fatCubinHandle, const struct surfaceReference *hostVar, const void **deviceAddress, const char *deviceName, int dim, int ext);
    typedef void (CUDARTAPI *typePtrCudaRegisterFunction)(void **fatCubinHandle, const char *hostFun, char *deviceFun, const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize);

    static typePtrCudaMalloc3D ptrCudaMalloc3D;
    static typePtrCudaMalloc3DArray ptrCudaMalloc3DArray;
    static typePtrCudaMemset3D ptrCudaMemset3D;
    static typePtrCudaMemcpy3D ptrCudaMemcpy3D;
    static typePtrCudaMemcpy3DAsync ptrCudaMemcpy3DAsync;
    static typePtrCudaMalloc ptrCudaMalloc;
    static typePtrCudaMallocHost ptrCudaMallocHost;
    static typePtrCudaMallocPitch ptrCudaMallocPitch;
    static typePtrCudaMallocArray ptrCudaMallocArray;
    static typePtrCudaFree ptrCudaFree;
    static typePtrCudaFreeHost ptrCudaFreeHost;
    static typePtrCudaFreeArray ptrCudaFreeArray;
    static typePtrCudaHostAlloc ptrCudaHostAlloc;
    static typePtrCudaHostGetDevicePointer ptrCudaHostGetDevicePointer;
    static typePtrCudaHostGetFlags ptrCudaHostGetFlags;
    static typePtrCudaMemGetInfo ptrCudaMemGetInfo;
    static typePtrCudaMemcpy ptrCudaMemcpy;
    static typePtrCudaMemcpyToArray ptrCudaMemcpyToArray;
    static typePtrCudaMemcpyFromArray ptrCudaMemcpyFromArray;
    static typePtrCudaMemcpyArrayToArray ptrCudaMemcpyArrayToArray;
    static typePtrCudaMemcpy2D ptrCudaMemcpy2D;
    static typePtrCudaMemcpy2DToArray ptrCudaMemcpy2DToArray;
    static typePtrCudaMemcpy2DFromArray ptrCudaMemcpy2DFromArray;
    static typePtrCudaMemcpy2DArrayToArray ptrCudaMemcpy2DArrayToArray;
    static typePtrCudaMemcpyToSymbol ptrCudaMemcpyToSymbol;
    static typePtrCudaMemcpyFromSymbol ptrCudaMemcpyFromSymbol;
    static typePtrCudaMemcpyAsync ptrCudaMemcpyAsync;
    static typePtrCudaMemcpyToArrayAsync ptrCudaMemcpyToArrayAsync;
    static typePtrCudaMemcpyFromArrayAsync ptrCudaMemcpyFromArrayAsync;
    static typePtrCudaMemcpy2DAsync ptrCudaMemcpy2DAsync;
    static typePtrCudaMemcpy2DToArrayAsync ptrCudaMemcpy2DToArrayAsync;
    static typePtrCudaMemcpy2DFromArrayAsync ptrCudaMemcpy2DFromArrayAsync;
    static typePtrCudaMemcpyToSymbolAsync ptrCudaMemcpyToSymbolAsync;
    static typePtrCudaMemcpyFromSymbolAsync ptrCudaMemcpyFromSymbolAsync;
    static typePtrCudaMemset ptrCudaMemset;
    static typePtrCudaMemset2D ptrCudaMemset2D;
    static typePtrCudaGetSymbolAddress ptrCudaGetSymbolAddress;
    static typePtrCudaGetSymbolSize ptrCudaGetSymbolSize;
    static typePtrCudaGetDeviceCount ptrCudaGetDeviceCount;
    static typePtrCudaGetDeviceProperties ptrCudaGetDeviceProperties;
    static typePtrCudaChooseDevice ptrCudaChooseDevice;
    static typePtrCudaSetDevice ptrCudaSetDevice;
    static typePtrCudaGetDevice ptrCudaGetDevice;
    static typePtrCudaSetValidDevices ptrCudaSetValidDevices;
    static typePtrCudaSetDeviceFlags ptrCudaSetDeviceFlags;
    static typePtrCudaBindTexture ptrCudaBindTexture;
    static typePtrCudaBindTexture2D ptrCudaBindTexture2D;
    static typePtrCudaBindTextureToArray ptrCudaBindTextureToArray;
    static typePtrCudaUnbindTexture ptrCudaUnbindTexture;
    static typePtrCudaGetTextureAlignmentOffset ptrCudaGetTextureAlignmentOffset;
    static typePtrCudaGetTextureReference ptrCudaGetTextureReference;
    static typePtrCudaBindSurfaceToArray ptrCudaBindSurfaceToArray;
    static typePtrCudaGetSurfaceAlignmentOffset ptrCudaGetSurfaceAlignmentOffset;
    static typePtrCudaGetSurfaceReference ptrCudaGetSurfaceReference;
    static typePtrCudaGetChannelDesc ptrCudaGetChannelDesc;
    static typePtrCudaCreateChannelDesc ptrCudaCreateChannelDesc;
    static typePtrCudaGetLastError ptrCudaGetLastError;
    static typePtrCudaPeekAtLastError ptrCudaPeekAtLastError;
    static typePtrCudaGetErrorString ptrCudaGetErrorString;
    static typePtrCudaConfigureCall ptrCudaConfigureCall;
    static typePtrCudaSetupArgument ptrCudaSetupArgument;
    static typePtrCudaFuncSetCacheConfig ptrCudaFuncSetCacheConfig;
    static typePtrCudaLaunch ptrCudaLaunch;
    static typePtrCudaFuncGetAttributes ptrCudaFuncGetAttributes;
    static typePtrCudaStreamCreate ptrCudaStreamCreate;
    static typePtrCudaStreamDestroy ptrCudaStreamDestroy;
    static typePtrCudaStreamSynchronize ptrCudaStreamSynchronize;
    static typePtrCudaStreamQuery ptrCudaStreamQuery;
    static typePtrCudaEventCreate ptrCudaEventCreate;
    static typePtrCudaEventCreateWithFlags ptrCudaEventCreateWithFlags;
    static typePtrCudaEventRecord ptrCudaEventRecord;
    static typePtrCudaEventQuery ptrCudaEventQuery;
    static typePtrCudaEventSynchronize ptrCudaEventSynchronize;
    static typePtrCudaEventDestroy ptrCudaEventDestroy;
    static typePtrCudaEventElapsedTime ptrCudaEventElapsedTime;
    static typePtrCudaSetDoubleForDevice ptrCudaSetDoubleForDevice;
    static typePtrCudaSetDoubleForHost ptrCudaSetDoubleForHost;
    static typePtrCudaThreadExit ptrCudaThreadExit;
    static typePtrCudaThreadSynchronize ptrCudaThreadSynchronize;
    static typePtrCudaThreadSetLimit ptrCudaThreadSetLimit;
    static typePtrCudaThreadGetLimit ptrCudaThreadGetLimit;
    static typePtrCudaDriverGetVersion ptrCudaDriverGetVersion;
    static typePtrCudaRuntimeGetVersion ptrCudaRuntimeGetVersion;
    static typePtrCudaGetExportTable ptrCudaGetExportTable;
    static typePtrCudaGraphicsUnregisterResource ptrCudaGraphicsUnregisterResource;
    static typePtrCudaGraphicsResourceSetMapFlags ptrCudaGraphicsResourceSetMapFlags;
    static typePtrCudaGraphicsMapResources ptrCudaGraphicsMapResources;
    static typePtrCudaGraphicsUnmapResources ptrCudaGraphicsUnmapResources;
    static typePtrCudaGraphicsResourceGetMappedPointer ptrCudaGraphicsResourceGetMappedPointer;
    static typePtrCudaGraphicsSubResourceGetMappedArray ptrCudaGraphicsSubResourceGetMappedArray;
    static typePtrCudaRegisterFatBinary ptrCudaRegisterFatBinary;
    static typePtrCudaUnregisterFatBinary ptrCudaUnregisterFatBinary;
    static typePtrCudaRegisterVar ptrCudaRegisterVar;
    static typePtrCudaRegisterTexture ptrCudaRegisterTexture;
    static typePtrCudaRegisterSurface ptrCudaRegisterSurface;
    static typePtrCudaRegisterFunction ptrCudaRegisterFunction;


public:
	static CudaMemoryDebug * Singleton();
	static bool DoInit(char * cuda_module_name, HookManager * hm);
    static void MakeContext(char * file_name, int line);
    static cudaError_t CUDARTAPI Malloc(void ** ptr, size_t size);
    static cudaError_t CUDARTAPI Free(void *);
    static cudaError_t CUDARTAPI HostAlloc(void ** ptr, size_t size, unsigned int flags);
    static cudaError_t CUDARTAPI FreeHost(void *);
    static cudaError_t CUDARTAPI HostGetDevicePointer(void ** pDevice, void * pHost, unsigned int flags);
    static cudaError_t CUDARTAPI Memcpy(void * dst, const void * src, size_t count, enum cudaMemcpyKind kind);    
    static cudaError_t CUDARTAPI Memset(void * devPtr, int value, size_t count);          
    static cudaError_t CUDARTAPI ThreadExit();
	static cudaError_t CUDARTAPI GetLastError();
	static void** CUDARTAPI RegisterFatBinary(void *fatCubin);
	static cudaError_t CUDARTAPI Launch(const char *entry);
	static void CUDARTAPI RegisterFunction(void **fatCubinHandle, const char *hostFun, char *deviceFun, const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize);
    static cudaError_t CUDARTAPI SetupArgument(const void *arg, size_t size, size_t offset);
    static cudaError_t CUDARTAPI ConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem __dv(0), cudaStream_t stream __dv(0));

    enum return_type {
        NOT_OK = 0,
        OK = 1
    };
    static return_type CheckOverwrite();
    static return_type CUDARTAPI SetPaddingSize(size_t s);
    static return_type CUDARTAPI SetPaddingByte(unsigned char b);
    static return_type CUDARTAPI SetDevicePointerToFirstByteInBlock(bool b);
    static return_type CUDARTAPI SetOutputStream(std::ostream * fp);
    static return_type CUDARTAPI SetTraceAllCalls(bool b);
	static return_type CUDARTAPI SetQuitOnError(bool b);
	static return_type CUDARTAPI SetDoNotCallCudaAfterSanityCheckFail(bool b);
	static return_type CopyOptions(CudaMemoryDebug * ptr);
private:
    static return_type CheckSinglePtrOverwrite(const data * d);
    static bool IsBadPointer(const void * ptr);
    static int FindAllocatedBlock(const void * pointer);
	static void Unimplemented();
};

