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
#include <list>

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

class DEVICE;

class DLL_API _CUDA_RUNTIME
{
private:
    bool did_wrap;
public:
    HMODULE hModule;
    _CUDA_RUNTIME();
    void WrapModule(char * cuda_module_name);

    // CUDA C Runtime API.
    typedef cudaError_t (CUDARTAPI *typePtrCudaArrayGetInfo)();
    typedef cudaError_t (CUDARTAPI *typePtrCudaBindSurfaceToArray)(const struct surfaceReference *surfref, const struct cudaArray *array, const struct cudaChannelFormatDesc *desc);
    typedef cudaError_t (CUDARTAPI *typePtrCudaBindTexture)(size_t *offset, const struct textureReference *texref, const void *devPtr, const struct cudaChannelFormatDesc *desc, size_t size __dv(UINT_MAX));
    typedef cudaError_t (CUDARTAPI *typePtrCudaBindTexture2D)(size_t *offset,const struct textureReference *texref,const void *devPtr, const struct cudaChannelFormatDesc *desc,size_t width, size_t height, size_t pitch);
    typedef cudaError_t (CUDARTAPI *typePtrCudaBindTextureToArray)(const struct textureReference *texref, const struct cudaArray *array, const struct cudaChannelFormatDesc *desc);
    typedef cudaError_t (CUDARTAPI *typePtrCudaBindTextureToMipmappedArray)();
    typedef cudaError_t (CUDARTAPI *typePtrCudaChooseDevice)(int *device, const struct cudaDeviceProp *prop);
    typedef cudaError_t (CUDARTAPI *typePtrCudaConfigureCall)(dim3 gridDim, dim3 blockDim, size_t sharedMem __dv(0), cudaStream_t stream __dv(0));
    typedef struct cudaChannelFormatDesc (CUDARTAPI *typePtrCudaCreateChannelDesc)(int x, int y, int z, int w, enum cudaChannelFormatKind f);
    typedef cudaError_t (CUDARTAPI *typePtrCudaCreateSurfaceObject)();
    typedef cudaError_t (CUDARTAPI *typePtrCudaCreateTextureObject)();
    typedef cudaError_t (CUDARTAPI *typePtrCudaDriverGetVersion)(int *driverVersion);
    typedef cudaError_t (CUDARTAPI *typePtrCudaEventCreate)(cudaEvent_t *event);
    typedef cudaError_t (CUDARTAPI *typePtrCudaEventCreateWithFlags)(cudaEvent_t *event, int flags);
    typedef cudaError_t (CUDARTAPI *typePtrCudaEventDestroy)(cudaEvent_t event);
    typedef cudaError_t (CUDARTAPI *typePtrCudaEventElapsedTime)(float *ms, cudaEvent_t start, cudaEvent_t end);
    typedef cudaError_t (CUDARTAPI *typePtrCudaEventQuery)(cudaEvent_t event);
    typedef cudaError_t (CUDARTAPI *typePtrCudaEventRecord)(cudaEvent_t event, cudaStream_t stream __dv(0));
    typedef cudaError_t (CUDARTAPI *typePtrCudaEventSynchronize)(cudaEvent_t event);
    typedef cudaError_t (CUDARTAPI *typePtrCudaFree)(void *devPtr);
    typedef cudaError_t (CUDARTAPI *typePtrCudaFreeArray)(struct cudaArray *array);
    typedef cudaError_t (CUDARTAPI *typePtrCudaFreeHost)(void *ptr);
    typedef cudaError_t (CUDARTAPI *typePtrCudaFuncGetAttributes)(struct cudaFuncAttributes *attr, const char *func);
    typedef cudaError_t (CUDARTAPI *typePtrCudaFuncSetCacheConfig)(const char *func, enum cudaFuncCache cacheConfig);
    typedef cudaError_t (CUDARTAPI *typePtrCudaGetChannelDesc)(struct cudaChannelFormatDesc *desc, const struct cudaArray *array);
    typedef cudaError_t (CUDARTAPI *typePtrCudaGetDevice)(int *device);
    typedef cudaError_t (CUDARTAPI *typePtrCudaGetDeviceCount)(int *count);
    typedef cudaError_t (CUDARTAPI *typePtrCudaGetDeviceProperties)(struct cudaDeviceProp *prop, int device);
    typedef const char* (CUDARTAPI *typePtrCudaGetErrorString)(cudaError_t error);
    typedef cudaError_t (CUDARTAPI *typePtrCudaGetExportTable)(const void **ppExportTable, const cudaUUID_t *pExportTableId);
    typedef cudaError_t (CUDARTAPI *typePtrCudaGetLastError)(void);
    typedef cudaError_t (CUDARTAPI *typePtrCudaGetSurfaceAlignmentOffset)(size_t *offset, const struct surfaceReference *surfref);
    typedef cudaError_t (CUDARTAPI *typePtrCudaGetSurfaceReference)(const struct surfaceReference **surfref, const char *symbol);
    typedef cudaError_t (CUDARTAPI *typePtrCudaGetSymbolAddress)(void **devPtr, const char *symbol);
    typedef cudaError_t (CUDARTAPI *typePtrCudaGetSymbolSize)(size_t *size, const char *symbol);
    typedef cudaError_t (CUDARTAPI *typePtrCudaGetTextureAlignmentOffset)(size_t *offset, const struct textureReference *texref);
    typedef cudaError_t (CUDARTAPI *typePtrCudaGetTextureReference)(const struct textureReference **texref, const char *symbol);
    typedef cudaError_t (CUDARTAPI *typePtrCudaGraphicsMapResources)(int count, struct cudaGraphicsResource **resources, cudaStream_t stream __dv(0));
    typedef cudaError_t (CUDARTAPI *typePtrCudaGraphicsResourceGetMappedPointer)(void **devPtr, size_t *size, struct cudaGraphicsResource *resource);
    typedef cudaError_t (CUDARTAPI *typePtrCudaGraphicsResourceSetMapFlags)(struct cudaGraphicsResource *resource, unsigned int flags); 
    typedef cudaError_t (CUDARTAPI *typePtrCudaGraphicsSubResourceGetMappedArray)(struct cudaArray **arrayPtr, struct cudaGraphicsResource *resource, unsigned int arrayIndex, unsigned int mipLevel);
    typedef cudaError_t (CUDARTAPI *typePtrCudaGraphicsUnmapResources)(int count, struct cudaGraphicsResource **resources, cudaStream_t stream __dv(0));
    typedef cudaError_t (CUDARTAPI *typePtrCudaGraphicsUnregisterResource)(struct cudaGraphicsResource *resource);
    typedef cudaError_t (CUDARTAPI *typePtrCudaHostAlloc)(void **pHost, size_t bytes, unsigned int flags);
    typedef cudaError_t (CUDARTAPI *typePtrCudaHostGetDevicePointer)(void **pDevice, void *pHost, unsigned int flags);
    typedef cudaError_t (CUDARTAPI *typePtrCudaHostGetFlags)(unsigned int *pFlags, void *pHost);
    typedef cudaError_t (CUDARTAPI *typePtrCudaLaunch)(const char *entry);
    typedef cudaError_t (CUDARTAPI *typePtrCudaMalloc)(void **devPtr, size_t size);
    typedef cudaError_t (CUDARTAPI *typePtrCudaMalloc3D)(struct cudaPitchedPtr* pitchedDevPtr, struct cudaExtent extent);
    typedef cudaError_t (CUDARTAPI *typePtrCudaMalloc3DArray)(struct cudaArray** arrayPtr, const struct cudaChannelFormatDesc* desc, struct cudaExtent extent, unsigned int flags __dv(0));
    typedef cudaError_t (CUDARTAPI *typePtrCudaMallocArray)(struct cudaArray **array, const struct cudaChannelFormatDesc *desc, size_t width, size_t height __dv(0), unsigned int flags __dv(0));
    typedef cudaError_t (CUDARTAPI *typePtrCudaMallocHost)(void **ptr, size_t size);
    typedef cudaError_t (CUDARTAPI *typePtrCudaMallocPitch)(void **devPtr, size_t *pitch, size_t width, size_t height);
    typedef cudaError_t (CUDARTAPI *typePtrCudaMemcpy)(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
    typedef cudaError_t (CUDARTAPI *typePtrCudaMemcpy2D)(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
    typedef cudaError_t (CUDARTAPI *typePtrCudaMemcpy2DArrayToArray)(struct cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToDevice));
    typedef cudaError_t (CUDARTAPI *typePtrCudaMemcpy2DAsync)(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
    typedef cudaError_t (CUDARTAPI *typePtrCudaMemcpy2DFromArray)(void *dst, size_t dpitch, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind);
    typedef cudaError_t (CUDARTAPI *typePtrCudaMemcpy2DFromArrayAsync)(void *dst, size_t dpitch, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
    typedef cudaError_t (CUDARTAPI *typePtrCudaMemcpy2DToArray)(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
    typedef cudaError_t (CUDARTAPI *typePtrCudaMemcpy2DToArrayAsync)(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
    typedef cudaError_t (CUDARTAPI *typePtrCudaMemcpy3D)(const struct cudaMemcpy3DParms *p);
    typedef cudaError_t (CUDARTAPI *typePtrCudaMemcpy3DAsync)(const struct cudaMemcpy3DParms *p, cudaStream_t stream __dv(0));
    typedef cudaError_t (CUDARTAPI *typePtrCudaMemcpyArrayToArray)(struct cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToDevice));
    typedef cudaError_t (CUDARTAPI *typePtrCudaMemcpyAsync)(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
    typedef cudaError_t (CUDARTAPI *typePtrCudaMemcpyFromArray)(void *dst, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind);
    typedef cudaError_t (CUDARTAPI *typePtrCudaMemcpyFromArrayAsync)(void *dst, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
    typedef cudaError_t (CUDARTAPI *typePtrCudaMemcpyFromSymbol)(void *dst, const char *symbol, size_t count, size_t offset __dv(0), enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToHost));
    typedef cudaError_t (CUDARTAPI *typePtrCudaMemcpyFromSymbolAsync)(void *dst, const char *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
    typedef cudaError_t (CUDARTAPI *typePtrCudaMemcpyToArray)(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind);
    typedef cudaError_t (CUDARTAPI *typePtrCudaMemcpyToArrayAsync)(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
    typedef cudaError_t (CUDARTAPI *typePtrCudaMemcpyToSymbol)(const char *symbol, const void *src, size_t count, size_t offset __dv(0), enum cudaMemcpyKind kind __dv(cudaMemcpyHostToDevice));
    typedef cudaError_t (CUDARTAPI *typePtrCudaMemcpyToSymbolAsync)(const char *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
    typedef cudaError_t (CUDARTAPI *typePtrCudaMemGetInfo)(size_t *free, size_t *total);
    typedef cudaError_t (CUDARTAPI *typePtrCudaMemset)(void *devPtr, int value, size_t count);
    typedef cudaError_t (CUDARTAPI *typePtrCudaMemset2D)(void *devPtr, size_t pitch, int value, size_t width, size_t height);
    typedef cudaError_t (CUDARTAPI *typePtrCudaMemset3D)(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent);
    typedef cudaError_t (CUDARTAPI *typePtrCudaPeekAtLastError)(void);
    typedef void** (CUDARTAPI *typePtrCudaRegisterDeviceFunction)();
    typedef void** (CUDARTAPI *typePtrCudaRegisterFatBinary)(void *fatCubin);
    typedef void (CUDARTAPI *typePtrCudaRegisterFunction)(void **fatCubinHandle, const char *hostFun, char *deviceFun, const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize);

    typedef cudaError_t (CUDARTAPI *typePtrCudaRegisterPrelinkedFatBinary)();
    typedef cudaError_t (CUDARTAPI *typePtrCudaRegisterShared)();
    typedef cudaError_t (CUDARTAPI *typePtrCudaRegisterSharedVar)();

    typedef void (CUDARTAPI *typePtrCudaRegisterSurface)(void **fatCubinHandle, const struct surfaceReference *hostVar, const void **deviceAddress, const char *deviceName, int dim, int ext);
    typedef void (CUDARTAPI *typePtrCudaRegisterTexture)(void **fatCubinHandle, const struct textureReference *hostVar, const void **deviceAddress, const char *deviceName, int dim, int norm, int ext);
    typedef void (CUDARTAPI *typePtrCudaRegisterVar)(void **fatCubinHandle, char *hostVar, char *deviceAddress, const char *deviceName, int ext, int size, int constant, int global);
    typedef cudaError_t (CUDARTAPI *typePtrCudaRuntimeGetVersion)(int *runtimeVersion);
    typedef cudaError_t (CUDARTAPI *typePtrCudaSetDevice)(int device);
    typedef cudaError_t (CUDARTAPI *typePtrCudaSetDeviceFlags)( int flags );
    typedef cudaError_t (CUDARTAPI *typePtrCudaSetDoubleForDevice)(double *d);
    typedef cudaError_t (CUDARTAPI *typePtrCudaSetDoubleForHost)(double *d);
    typedef cudaError_t (CUDARTAPI *typePtrCudaSetupArgument)(const void *arg, size_t size, size_t offset);
    typedef cudaError_t (CUDARTAPI *typePtrCudaSetValidDevices)(int *device_arr, int len);
    typedef cudaError_t (CUDARTAPI *typePtrCudaStreamCreate)(cudaStream_t *pStream);
    typedef cudaError_t (CUDARTAPI *typePtrCudaStreamDestroy)(cudaStream_t stream);
    typedef cudaError_t (CUDARTAPI *typePtrCudaStreamQuery)(cudaStream_t stream);
    typedef cudaError_t (CUDARTAPI *typePtrCudaStreamSynchronize)(cudaStream_t stream);
    typedef cudaError_t (CUDARTAPI *typePtrCudaThreadExit)(void);
    typedef cudaError_t (CUDARTAPI *typePtrCudaThreadGetLimit)(size_t *pValue, enum cudaLimit limit);
    typedef cudaError_t (CUDARTAPI *typePtrCudaThreadSetLimit)(enum cudaLimit limit, size_t value);
    typedef cudaError_t (CUDARTAPI *typePtrCudaThreadSynchronize)(void);
    typedef cudaError_t (CUDARTAPI *typePtrCudaUnbindTexture)(const struct textureReference *texref);
    typedef void (CUDARTAPI *typePtrCudaUnregisterFatBinary)(void **fatCubinHandle);

    

	static cudaError_t CUDARTAPI _cudaArrayGetInfo();
    static cudaError_t CUDARTAPI _cudaBindSurfaceToArray(const struct surfaceReference *surfref, const struct cudaArray *array, const struct cudaChannelFormatDesc *desc);
    static cudaError_t CUDARTAPI _cudaBindTexture(size_t *offset, const struct textureReference *texref, const void *devPtr, const struct cudaChannelFormatDesc *desc, size_t size __dv(UINT_MAX));
    static cudaError_t CUDARTAPI _cudaBindTexture2D(size_t *offset,const struct textureReference *texref,const void *devPtr, const struct cudaChannelFormatDesc *desc,size_t width, size_t height, size_t pitch);
    static cudaError_t CUDARTAPI _cudaBindTextureToArray(const struct textureReference *texref, const struct cudaArray *array, const struct cudaChannelFormatDesc *desc);
    static cudaError_t CUDARTAPI _cudaBindTextureToMipmappedArray();
    static cudaError_t CUDARTAPI _cudaChooseDevice(int *device, const struct cudaDeviceProp *prop);
    static cudaError_t CUDARTAPI _cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem __dv(0), cudaStream_t stream __dv(0));
    static struct cudaChannelFormatDesc CUDARTAPI _cudaCreateChannelDesc(int x, int y, int z, int w, enum cudaChannelFormatKind e);
    static cudaError_t CUDARTAPI _cudaCreateSurfaceObject();
    static cudaError_t CUDARTAPI _cudaCreateTextureObject();
    static cudaError_t CUDARTAPI _cudaD3D10GetDevice();
    static cudaError_t CUDARTAPI _cudaD3D10GetDevices();
    static cudaError_t CUDARTAPI _cudaD3D10GetDirect3DDevice();
    static cudaError_t CUDARTAPI _cudaD3D10MapResources();
    static cudaError_t CUDARTAPI _cudaD3D10RegisterResource();
    static cudaError_t CUDARTAPI _cudaD3D10ResourceGetMappedArray();
    static cudaError_t CUDARTAPI _cudaD3D10ResourceGetMappedPitch();
    static cudaError_t CUDARTAPI _cudaD3D10ResourceGetMappedPointer();
    static cudaError_t CUDARTAPI _cudaD3D10ResourceGetMappedSize();
    static cudaError_t CUDARTAPI _cudaD3D10ResourceGetSurfaceDimensions();
    static cudaError_t CUDARTAPI _cudaD3D10ResourceSetMapFlags();
    static cudaError_t CUDARTAPI _cudaD3D10SetDirect3DDevice();
    static cudaError_t CUDARTAPI _cudaD3D10UnmapResources();
    static cudaError_t CUDARTAPI _cudaD3D10UnregisterResource();
    static cudaError_t CUDARTAPI _cudaD3D11GetDevice();
    static cudaError_t CUDARTAPI _cudaD3D11GetDevices();
    static cudaError_t CUDARTAPI _cudaD3D11GetDirect3DDevice();
    static cudaError_t CUDARTAPI _cudaD3D11SetDirect3DDevice();
    static cudaError_t CUDARTAPI _cudaD3D9Begin();
    static cudaError_t CUDARTAPI _cudaD3D9End();
    static cudaError_t CUDARTAPI _cudaD3D9GetDevice();
    static cudaError_t CUDARTAPI _cudaD3D9GetDevices();
    static cudaError_t CUDARTAPI _cudaD3D9GetDirect3DDevice();
    static cudaError_t CUDARTAPI _cudaD3D9MapResources();
    static cudaError_t CUDARTAPI _cudaD3D9MapVertexBuffer();
    static cudaError_t CUDARTAPI _cudaD3D9RegisterResource();
    static cudaError_t CUDARTAPI _cudaD3D9RegisterVertexBuffer();
    static cudaError_t CUDARTAPI _cudaD3D9ResourceGetMappedArray();
    static cudaError_t CUDARTAPI _cudaD3D9ResourceGetMappedPitch();
    static cudaError_t CUDARTAPI _cudaD3D9ResourceGetMappedPointer();
    static cudaError_t CUDARTAPI _cudaD3D9ResourceGetMappedSize();
    static cudaError_t CUDARTAPI _cudaD3D9ResourceGetSurfaceDimensions();
    static cudaError_t CUDARTAPI _cudaD3D9ResourceSetMapFlags();
    static cudaError_t CUDARTAPI _cudaD3D9SetDirect3DDevice();
    static cudaError_t CUDARTAPI _cudaD3D9UnmapResources();
    static cudaError_t CUDARTAPI _cudaD3D9UnmapVertexBuffer();
    static cudaError_t CUDARTAPI _cudaD3D9UnregisterResource();
    static cudaError_t CUDARTAPI _cudaD3D9UnregisterVertexBuffer();
    static cudaError_t CUDARTAPI _cudaDestroySurfaceObject();
    static cudaError_t CUDARTAPI _cudaDestroyTextureObject();
    static cudaError_t CUDARTAPI _cudaDeviceCanAccessPeer();
    static cudaError_t CUDARTAPI _cudaDeviceDisablePeerAccess();
    static cudaError_t CUDARTAPI _cudaDeviceEnablePeerAccess();
    static cudaError_t CUDARTAPI _cudaDeviceGetAttribute();
    static cudaError_t CUDARTAPI _cudaDeviceGetByPCIBusId();
    static cudaError_t CUDARTAPI _cudaDeviceGetCacheConfig();
    static cudaError_t CUDARTAPI _cudaDeviceGetLimit();
    static cudaError_t CUDARTAPI _cudaDeviceGetPCIBusId();
    static cudaError_t CUDARTAPI _cudaDeviceGetSharedMemConfig();
    static cudaError_t CUDARTAPI _cudaDeviceGetStreamPriorityRange();
    static cudaError_t CUDARTAPI _cudaDeviceReset();
    static cudaError_t CUDARTAPI _cudaDeviceSetCacheConfig();
    static cudaError_t CUDARTAPI _cudaDeviceSetLimit();
    static cudaError_t CUDARTAPI _cudaDeviceSetSharedMemConfig();
    static cudaError_t CUDARTAPI _cudaDeviceSynchronize();
    static cudaError_t CUDARTAPI _cudaDriverGetVersion(int *driverVersion);
    static cudaError_t CUDARTAPI _cudaEventCreate(cudaEvent_t *event);
    static cudaError_t CUDARTAPI _cudaEventCreateWithFlags(cudaEvent_t *event, int flags);
    static cudaError_t CUDARTAPI _cudaEventDestroy(cudaEvent_t event);
    static cudaError_t CUDARTAPI _cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end);
    static cudaError_t CUDARTAPI _cudaEventQuery(cudaEvent_t event);
    static cudaError_t CUDARTAPI _cudaEventRecord(cudaEvent_t event, cudaStream_t stream __dv(0));
    static cudaError_t CUDARTAPI _cudaEventSynchronize(cudaEvent_t event);
    static cudaError_t CUDARTAPI _cudaFree(void * ptr);
    static cudaError_t CUDARTAPI _cudaFreeArray(struct cudaArray *array);
    static cudaError_t CUDARTAPI _cudaFreeHost(void * ptr);
    static cudaError_t CUDARTAPI _cudaFreeMipmappedArray();
    static cudaError_t CUDARTAPI _cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const char *func);
    static cudaError_t CUDARTAPI _cudaFuncSetCacheConfig(const char *func, enum cudaFuncCache cacheConfig);
    static cudaError_t CUDARTAPI _cudaFuncSetSharedMemConfig();
    static cudaError_t CUDARTAPI _cudaGetChannelDesc(struct cudaChannelFormatDesc *desc, const struct cudaArray *array);
    static cudaError_t CUDARTAPI _cudaGetDevice(int *device);
    static cudaError_t CUDARTAPI _cudaGetDeviceCount(int *count);
    static cudaError_t CUDARTAPI _cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device);
    static const char* CUDARTAPI _cudaGetErrorString(cudaError_t error);
    static cudaError_t CUDARTAPI _cudaGetExportTable(const void **ppExportTable, const cudaUUID_t *pExportTableId);
    static cudaError_t CUDARTAPI _cudaGetLastError(void);
    static cudaError_t CUDARTAPI _cudaGetMipmappedArrayLevel();
    static cudaError_t CUDARTAPI _cudaGetSurfaceAlignmentOffset(size_t *offset, const struct surfaceReference *surfref);
    static cudaError_t CUDARTAPI _cudaGetSurfaceObjectResourceDesc();
    static cudaError_t CUDARTAPI _cudaGetSurfaceReference(const struct surfaceReference **surfref, const char *symbol);
    static cudaError_t CUDARTAPI _cudaGetSymbolAddress(void **devPtr, const char *symbol);
    static cudaError_t CUDARTAPI _cudaGetSymbolSize(size_t *size, const char *symbol);
    static cudaError_t CUDARTAPI _cudaGetTextureAlignmentOffset(size_t *offset, const struct textureReference *texref);
    static cudaError_t CUDARTAPI _cudaGetTextureObjectResourceDesc();
    static cudaError_t CUDARTAPI _cudaGetTextureObjectResourceViewDesc();
    static cudaError_t CUDARTAPI _cudaGetTextureObjectTextureDesc();
    static cudaError_t CUDARTAPI _cudaGetTextureReference(const struct textureReference **texref, const char *symbol);
    static cudaError_t CUDARTAPI _cudaGLGetDevices();
    static cudaError_t CUDARTAPI _cudaGLMapBufferObject();
    static cudaError_t CUDARTAPI _cudaGLMapBufferObjectAsync();
    static cudaError_t CUDARTAPI _cudaGLRegisterBufferObject();
    static cudaError_t CUDARTAPI _cudaGLSetBufferObjectMapFlags();
    static cudaError_t CUDARTAPI _cudaGLSetGLDevice();
    static cudaError_t CUDARTAPI _cudaGLUnmapBufferObject();
    static cudaError_t CUDARTAPI _cudaGLUnmapBufferObjectAsync();
    static cudaError_t CUDARTAPI _cudaGLUnregisterBufferObject();
    static cudaError_t CUDARTAPI _cudaGraphicsD3D10RegisterResource();
    static cudaError_t CUDARTAPI _cudaGraphicsD3D11RegisterResource();
    static cudaError_t CUDARTAPI _cudaGraphicsD3D9RegisterResource();
    static cudaError_t CUDARTAPI _cudaGraphicsGLRegisterBuffer();
    static cudaError_t CUDARTAPI _cudaGraphicsGLRegisterImage();
    static cudaError_t CUDARTAPI _cudaGraphicsMapResources(int count, struct cudaGraphicsResource **resources, cudaStream_t stream __dv(0));
    static cudaError_t CUDARTAPI _cudaGraphicsResourceGetMappedPointer(void **devPtr, size_t *size, struct cudaGraphicsResource *resource);
    static cudaError_t CUDARTAPI _cudaGraphicsResourceSetMapFlags(struct cudaGraphicsResource *resource, unsigned int flags);
    static cudaError_t CUDARTAPI _cudaGraphicsSubResourceGetMappedArray(struct cudaArray **arrayPtr, struct cudaGraphicsResource *resource, unsigned int arrayIndex, unsigned int mipLevel);
    static cudaError_t CUDARTAPI _cudaGraphicsUnmapResources(int count, struct cudaGraphicsResource **resources, cudaStream_t stream __dv(0));
    static cudaError_t CUDARTAPI _cudaGraphicsUnregisterResource(struct cudaGraphicsResource *resource);
    static cudaError_t CUDARTAPI _cudaHostAlloc(void ** ptr, size_t size, unsigned int flags);
    static cudaError_t CUDARTAPI _cudaHostGetDevicePointer(void ** pDevice, void * pHost, unsigned int flags);
    static cudaError_t CUDARTAPI _cudaHostGetFlags(unsigned int *pFlags, void *pHost);
    static cudaError_t CUDARTAPI _cudaHostRegister();
    static cudaError_t CUDARTAPI _cudaHostUnregister();
    static cudaError_t CUDARTAPI _cudaIpcCloseMemHandle();
    static cudaError_t CUDARTAPI _cudaIpcGetEventHandle();
    static cudaError_t CUDARTAPI _cudaIpcGetMemHandle();
    static cudaError_t CUDARTAPI _cudaIpcOpenEventHandle();
    static cudaError_t CUDARTAPI _cudaIpcOpenMemHandle();
    static cudaError_t CUDARTAPI _cudaLaunch(const char *entry);
    static cudaError_t CUDARTAPI _cudaMalloc(void ** ptr, size_t size);
    static cudaError_t CUDARTAPI _cudaMalloc3D();
    static cudaError_t CUDARTAPI _cudaMalloc3DArray(struct cudaPitchedPtr* pitchedDevPtr, struct cudaExtent extent);
    static cudaError_t CUDARTAPI _cudaMallocArray(struct cudaArray **array, const struct cudaChannelFormatDesc *desc, size_t width, size_t height __dv(0), unsigned int flags __dv(0));
    static cudaError_t CUDARTAPI _cudaMallocHost(void **ptr, size_t size);
    static cudaError_t CUDARTAPI _cudaMallocMipmappedArray();
    static cudaError_t CUDARTAPI _cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height);
    static cudaError_t CUDARTAPI _cudaMemcpy(void * dst, const void * src, size_t count, enum cudaMemcpyKind kind);
    static cudaError_t CUDARTAPI _cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
    static cudaError_t CUDARTAPI _cudaMemcpy2DArrayToArray(struct cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToDevice));
    static cudaError_t CUDARTAPI _cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
    static cudaError_t CUDARTAPI _cudaMemcpy2DFromArray(void *dst, size_t dpitch, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind);
    static cudaError_t CUDARTAPI _cudaMemcpy2DFromArrayAsync(void *dst, size_t dpitch, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
    static cudaError_t CUDARTAPI _cudaMemcpy2DToArray(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
    static cudaError_t CUDARTAPI _cudaMemcpy2DToArrayAsync(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
    static cudaError_t CUDARTAPI _cudaMemcpy3D(const struct cudaMemcpy3DParms *p);
    static cudaError_t CUDARTAPI _cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *p, cudaStream_t stream __dv(0));
    static cudaError_t CUDARTAPI _cudaMemcpy3DPeer();
    static cudaError_t CUDARTAPI _cudaMemcpy3DPeerAsync();
    static cudaError_t CUDARTAPI _cudaMemcpyArrayToArray(struct cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToDevice));
    static cudaError_t CUDARTAPI _cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
    static cudaError_t CUDARTAPI _cudaMemcpyFromArray(void *dst, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind);
    static cudaError_t CUDARTAPI _cudaMemcpyFromArrayAsync(void *dst, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
    static cudaError_t CUDARTAPI _cudaMemcpyFromSymbol(void *dst, const char *symbol, size_t count, size_t offset __dv(0), enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToHost));
    static cudaError_t CUDARTAPI _cudaMemcpyFromSymbolAsync(void *dst, const char *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
    static cudaError_t CUDARTAPI _cudaMemcpyPeer();
    static cudaError_t CUDARTAPI _cudaMemcpyPeerAsync();
    static cudaError_t CUDARTAPI _cudaMemcpyToArray(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind);
    static cudaError_t CUDARTAPI _cudaMemcpyToArrayAsync(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
    static cudaError_t CUDARTAPI _cudaMemcpyToSymbol(const char *symbol, const void *src, size_t count, size_t offset __dv(0), enum cudaMemcpyKind kind __dv(cudaMemcpyHostToDevice));
    static cudaError_t CUDARTAPI _cudaMemcpyToSymbolAsync(const char *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
    static cudaError_t CUDARTAPI _cudaMemGetInfo(size_t *free, size_t *total);
    static cudaError_t CUDARTAPI _cudaMemset(void * devPtr, int value, size_t count);          
    static cudaError_t CUDARTAPI _cudaMemsetAsync();
    static cudaError_t CUDARTAPI _cudaMemset2D(void *devPtr, size_t pitch, int value, size_t width, size_t height);
    static cudaError_t CUDARTAPI _cudaMemset2DAsync();
    static cudaError_t CUDARTAPI _cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent);
    static cudaError_t CUDARTAPI _cudaMemset3DAsync();
	static cudaError_t CUDARTAPI _cudaPeekAtLastError(void);
	static cudaError_t CUDARTAPI _cudaPointerGetAttributes();
	static cudaError_t CUDARTAPI _cudaProfilerStart();
	static cudaError_t CUDARTAPI _cudaProfilerStop();
    static cudaError_t CUDARTAPI _cudaRegisterDeviceFunction();
    static void** CUDARTAPI _cudaRegisterFatBinary(void *fatCubin);
    static void CUDARTAPI _cudaRegisterFunction(void **fatCubinHandle, const char *hostFun, char *deviceFun, const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize);
    static cudaError_t CUDARTAPI _cudaRegisterPrelinkedFatBinary();
    static cudaError_t CUDARTAPI _cudaRegisterShared();
    static cudaError_t CUDARTAPI _cudaRegisterSharedVar();
    static void CUDARTAPI _cudaRegisterSurface(void **fatCubinHandle, const struct surfaceReference *hostVar, const void **deviceAddress, const char *deviceName, int dim, int ext);
    static void CUDARTAPI _cudaRegisterTexture(void **fatCubinHandle, const struct textureReference *hostVar, const void **deviceAddress, const char *deviceName, int dim, int norm, int ext);
    static void CUDARTAPI _cudaRegisterVar(void **fatCubinHandle, char *hostVar, char *deviceAddress, const char *deviceName, int ext, int size, int constant, int global);
    static cudaError_t CUDARTAPI _cudaRuntimeGetVersion(int *runtimeVersion);
    static cudaError_t CUDARTAPI _cudaSetDevice(int device);
    static cudaError_t CUDARTAPI _cudaSetDeviceFlags(unsigned int flags);
    static cudaError_t CUDARTAPI _cudaSetDoubleForDevice(double *d);
    static cudaError_t CUDARTAPI _cudaSetDoubleForHost(double *d);
    static cudaError_t CUDARTAPI _cudaSetupArgument(const void *arg, size_t size, size_t offset);
    static cudaError_t CUDARTAPI _cudaSetValidDevices(int *device_arr, int len);
	static cudaError_t CUDARTAPI _cudaStreamAddCallback();
	static cudaError_t CUDARTAPI _cudaStreamCreate(cudaStream_t *pStream);
	static cudaError_t CUDARTAPI _cudaStreamCreateWithFlags();
	static cudaError_t CUDARTAPI _cudaStreamCreateWithPriority();
	static cudaError_t CUDARTAPI _cudaStreamDestroy(cudaStream_t stream);
	static cudaError_t CUDARTAPI _cudaStreamGetFlags();
	static cudaError_t CUDARTAPI _cudaStreamGetPriority();
    static cudaError_t CUDARTAPI _cudaStreamQuery(cudaStream_t stream);
	static cudaError_t CUDARTAPI _cudaStreamSynchronize(cudaStream_t stream);
	static cudaError_t CUDARTAPI _cudaStreamWaitEvent();
    static cudaError_t CUDARTAPI _cudaThreadExit();
	static cudaError_t CUDARTAPI _cudaThreadGetCacheConfig();
	static cudaError_t CUDARTAPI _cudaThreadGetLimit(enum cudaLimit limit, size_t value);
	static cudaError_t CUDARTAPI _cudaThreadSetCacheConfig();
    static cudaError_t CUDARTAPI _cudaThreadSetLimit(enum cudaLimit limit, size_t value);
    static cudaError_t CUDARTAPI _cudaThreadSynchronize(void);
	static cudaError_t CUDARTAPI _cudaUnbindTexture(const struct textureReference *texref);
	static void CUDARTAPI _cudaUnregisterFatBinary(void **fatCubinHandle);
	static cudaError_t CUDARTAPI _cudaWGLGetDevice();
	static cudaError_t CUDARTAPI _NvOptimusEnablementCuda();

    static void Unimplemented();
};
