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
#include <Windows.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h> // cudaError_t, CUDARTAPI, etc.

class HOOK_MANAGER;

#ifdef DLL_EXPORTS
#define DLL_API __declspec(dllexport)
#else
#define DLL_API __declspec(dllimport)
#endif

/// Wrapper for CUDA's dynamic link library.
#ifndef __dv
#define __dv(x)
#endif


class DLL_API _CUDA_RUNTIME
{
private:
    bool did_wrap;
public:
    HMODULE hModule;
    _CUDA_RUNTIME() { did_wrap = false; }
public:
    void WrapModule(char * cuda_module_name);
    // CUDA C Runtime API.
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
    typedef cudaError_t (CUDARTAPI *ptrCudaMemcpyAsync)(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
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
    typedef cudaError_t (CUDARTAPI *ptrCudaChooseDevice)(int *device, const struct cudaDeviceProp *prop);
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
    typedef cudaError_t (CUDARTAPI *ptrCudaStreamCreate)(cudaStream_t *pStream);
    typedef cudaError_t (CUDARTAPI *ptrCudaStreamDestroy)(cudaStream_t stream);
    typedef cudaError_t (CUDARTAPI *ptrCudaStreamSynchronize)(cudaStream_t stream);
    typedef cudaError_t (CUDARTAPI *ptrCudaStreamQuery)(cudaStream_t stream);

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

	typedef void (CUDARTAPI *typePtrCudaRegisterTexture)(void                    **fatCubinHandle,
		  const struct textureReference  *hostVar,
		  const void                    **deviceAddress,
		  const char                     *deviceName,
				int                       dim,       
				int                       norm,      
				int                       ext     );

    static cudaError_t CUDARTAPI Malloc(void ** ptr, size_t size);
    static cudaError_t CUDARTAPI Free(void *);
    static cudaError_t CUDARTAPI HostAlloc(void ** ptr, size_t size, unsigned int flags);
    static cudaError_t CUDARTAPI FreeHost(void *);
    static cudaError_t CUDARTAPI HostGetDevicePointer(void ** pDevice, void * pHost, unsigned int flags);
    static cudaError_t CUDARTAPI Memcpy(void * dst, const void * src, size_t count, enum cudaMemcpyKind kind);    
    static cudaError_t CUDARTAPI Memset(void * devPtr, int value, size_t count);          
    static cudaError_t CUDARTAPI ThreadExit();
    static struct cudaChannelFormatDesc CUDARTAPI _cudaCreateChannelDesc(int x, int y, int z, int w, enum cudaChannelFormatKind e);
    static cudaError_t CUDARTAPI _cudaGetLastError();
    static void** CUDARTAPI _cudaRegisterFatBinary(void *fatCubin);
    static cudaError_t CUDARTAPI _cudaLaunch(const char *entry);
    static void CUDARTAPI _cudaRegisterFunction(void **fatCubinHandle, const char *hostFun, char *deviceFun, const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize);
    static cudaError_t CUDARTAPI _cudaSetupArgument(const void *arg, size_t size, size_t offset);
    static cudaError_t CUDARTAPI _cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem __dv(0), cudaStream_t stream __dv(0));
    static cudaError_t CUDARTAPI _cudaThreadSynchronize(void);
    static void CUDARTAPI _cudaUnregisterFatBinary(void **fatCubinHandle);
    static void CUDARTAPI _cudaRegisterVar(void **fatCubinHandle, char *hostVar, char *deviceAddress, const char *deviceName, int ext, int size, int constant, int global);
    static cudaError_t CUDARTAPI _cudaGetDevice(int *device);
    static cudaError_t CUDARTAPI _cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device);
    static cudaError_t CUDARTAPI _cudaGetDeviceCount(int *count);
    static cudaError_t CUDARTAPI _cudaSetDevice(int device);
    static cudaError_t CUDARTAPI _cudaSetDeviceFlags(unsigned int flags);
    static cudaError_t CUDARTAPI _cudaBindTexture(size_t *offset, const struct textureReference *texref, const void *devPtr,
		const struct cudaChannelFormatDesc *desc, size_t size __dv(UINT_MAX));
    static cudaError_t CUDARTAPI _cudaStreamCreate(cudaStream_t *pStream);
    static cudaError_t CUDARTAPI _cudaStreamDestroy(cudaStream_t stream);
    static cudaError_t CUDARTAPI _cudaStreamSynchronize(cudaStream_t stream);
    static cudaError_t CUDARTAPI _cudaStreamQuery(cudaStream_t stream);
    static cudaError_t CUDARTAPI _cudaChooseDevice(int *device, const struct cudaDeviceProp *prop);
	static void CUDARTAPI __cudaRegisterTexture(
				void                    **fatCubinHandle,
		  const struct textureReference  *hostVar,
		  const void                    **deviceAddress,
		  const char                     *deviceName,
				int                       dim,       
				int                       norm,      
				int                       ext        
		);
	static cudaError_t CUDARTAPI cudaUnbindTexture(const struct textureReference *texref);
    static void Unimplemented();
	static void Unimplemented_cudaMalloc3DArray();
	static void Unimplemented_cudaMemset3D();
	static void Unimplemented_cudaMemcpy3D();
	static void Unimplemented_cudaMemcpy3DAsync();
	static void Unimplemented_cudaMallocHost();
	static void Unimplemented_cudaMallocPitch();
	static void Unimplemented_cudaMallocArray();
	static void Unimplemented_cudaFreeArray();
	static void Unimplemented_cudaHostGetFlags();
	static void Unimplemented_cudaMemGetInfo();
	static void Unimplemented_cudaMemcpyToArray();
	static void Unimplemented_cudaMemcpyFromArray();
	static void Unimplemented_cudaMemcpyArrayToArray();
	static void Unimplemented_cudaMemcpy2D();
	static void Unimplemented_cudaMemcpy2DToArray();
	static void Unimplemented_cudaMemcpy2DFromArray();
	static void Unimplemented_cudaMemcpy2DArrayToArray();
	static void Unimplemented_cudaMemcpyToSymbol();
	static void Unimplemented_cudaMemcpyFromSymbol();
	static void Unimplemented_cudaMemcpyAsync();
	static void Unimplemented_cudaMemcpyToArrayAsync();
	static void Unimplemented_cudaMemcpyFromArrayAsync();
	static void Unimplemented_cudaMemcpy2DAsync();
	static void Unimplemented_cudaMemcpy2DToArrayAsync();
	static void Unimplemented_cudaMemcpy2DFromArrayAsync();
	static void Unimplemented_cudaMemcpyToSymbolAsync();
	static void Unimplemented_cudaMemcpyFromSymbolAsync();
	static void Unimplemented_cudaMemset2D();
	static void Unimplemented_cudaGetSymbolAddress();
	static void Unimplemented_cudaGetSymbolSize();
	static void Unimplemented_cudaSetValidDevices();
	static void Unimplemented_cudaBindTexture2D();
	static void Unimplemented_cudaBindTextureToArray();
	static void Unimplemented_cudaGetTextureAlignmentOffset();
	static void Unimplemented_cudaGetTextureReference();
	static void Unimplemented_cudaBindSurfaceToArray();
	static void Unimplemented_cudaGetSurfaceReference();
	static void Unimplemented_cudaGetChannelDesc();
	static void Unimplemented_cudaPeekAtLastError();
	static void Unimplemented_cudaGetErrorString();
	static void Unimplemented_cudaFuncSetCacheConfig();
	static void Unimplemented_cudaFuncGetAttributes();
	static void Unimplemented_cudaEventCreate();
	static void Unimplemented_cudaEventCreateWithFlags();
	static void Unimplemented_cudaEventRecord();
	static void Unimplemented_cudaEventQuery();
	static void Unimplemented_cudaEventSynchronize();
	static void Unimplemented_cudaEventDestroy();
	static void Unimplemented_cudaEventElapsedTime();
	static void Unimplemented_cudaSetDoubleForDevice();
	static void Unimplemented_cudaSetDoubleForHost();
	static void Unimplemented_cudaThreadSetLimit();
	static void Unimplemented_cudaThreadGetLimit();
	static void Unimplemented_cudaDriverGetVersion();
	static void Unimplemented_cudaRuntimeGetVersion();
	static void Unimplemented_cudaGetExportTable();
	static void Unimplemented_cudaGraphicsUnregisterResource();
	static void Unimplemented_cudaGraphicsResourceSetMapFlags();
	static void Unimplemented_cudaGraphicsMapResources();
	static void Unimplemented_cudaGraphicsUnmapResources();
	static void Unimplemented_cudaGraphicsResourceGetMappedPointer();
	static void Unimplemented_cudaGraphicsSubResourceGetMappedArray();
	static void Unimplemented___cudaRegisterSurface();
};
