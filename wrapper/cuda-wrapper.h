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
#include <cuda.h>
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


class DLL_API CUDA_WRAPPER
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
    CUDA_WRAPPER();
    static CUDA_WRAPPER * singleton;
    HookManager * hook_manager;
    size_t padding_size;
    unsigned char padding_byte;
    bool device_pointer_to_first_byte_in_block;
    std::ostream * output_stream;
    bool trace_all_calls;
    bool quit_on_error;
    bool do_not_call_cuda_after_sanity_check_fail;
    bool do_emulation;
    bool do_debug_halt;
    bool init;
    static void ExitHandler();
    std::vector<data> alloc_list;
    static char * Context(int lines = 1);
    char * global_context;
    char * device; // device to run.

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

public:
    static CUDA_WRAPPER * Singleton();
    void DoInit();
    bool WrapModule(char * modname);
    static BOOL CUDARTAPI WrapCuda();
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
    static cudaError_t CUDARTAPI _cudaLaunch(const char *entry);
    static void CUDARTAPI _cudaRegisterFunction(void **fatCubinHandle, const char *hostFun, char *deviceFun, const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize);
    static cudaError_t CUDARTAPI _cudaSetupArgument(const void *arg, size_t size, size_t offset);
    static cudaError_t CUDARTAPI _cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem __dv(0), cudaStream_t stream __dv(0));
    static cudaError_t CUDARTAPI _cudaThreadSynchronize(void);
    static void CUDARTAPI UnregisterFatBinary(void **fatCubinHandle);
    static cudaError_t CUDARTAPI _cudaGetDevice(int *device);
    static cudaError_t CUDARTAPI _cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device);
    static cudaError_t CUDARTAPI GetDeviceCount(int *count);
    static cudaError_t CUDARTAPI _cudaSetDevice(int device);

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
    static return_type CopyOptions(CUDA_WRAPPER * ptr);
    static return_type CUDARTAPI RunDevice(char * device);
    static void CUDARTAPI SetTrace(int level);

private:
    static return_type CheckSinglePtrOverwrite(const data * d);
    static bool IsBadPointer(const void * ptr);
    static int FindAllocatedBlock(const void * pointer);
    static void Unimplemented();


public:
    // Driver API.
    typedef CUresult (CUDAAPI * ptrCuInit)(unsigned int Flags);
    typedef CUresult (CUDAAPI * ptrCuDriverGetVersion)(int *driverVersion);
    typedef CUresult (CUDAAPI * ptrCuDeviceGet)(CUdevice *device, int ordinal);
    typedef CUresult (CUDAAPI * ptrCuDeviceGetCount)(int *count);
    typedef CUresult (CUDAAPI * ptrCuDeviceGetName)(char *name, int len, CUdevice dev);
    typedef CUresult (CUDAAPI * ptrCuDeviceComputeCapability)(int *major, int *minor, CUdevice dev);
    typedef CUresult (CUDAAPI * ptrCuDeviceTotalMem)(unsigned int *bytes, CUdevice dev);
    typedef CUresult (CUDAAPI * ptrCuDeviceGetProperties)(CUdevprop *prop, CUdevice dev);
    typedef CUresult (CUDAAPI * ptrCuDeviceGetAttribute)(int *pi, CUdevice_attribute attrib, CUdevice dev);
    typedef CUresult (CUDAAPI * ptrCuCtxCreate)(CUcontext *pctx, unsigned int flags, CUdevice dev );
    typedef CUresult (CUDAAPI * ptrCuCtxDestroy)( CUcontext ctx );
    typedef CUresult (CUDAAPI * ptrCuCtxAttach)(CUcontext *pctx, unsigned int flags);
    typedef CUresult (CUDAAPI * ptrCuCtxDetach)(CUcontext ctx);
    typedef CUresult (CUDAAPI * ptrCuCtxPushCurrent)( CUcontext ctx );
    typedef CUresult (CUDAAPI * ptrCuCtxPopCurrent)( CUcontext *pctx );
    typedef CUresult (CUDAAPI * ptrCuCtxGetDevice)(CUdevice *device);
    typedef CUresult (CUDAAPI * ptrCuCtxSynchronize)(void);
    typedef CUresult (CUDAAPI * ptrCuModuleLoad)(CUmodule *module, const char *fname);
    typedef CUresult (CUDAAPI * ptrCuModuleLoadData)(CUmodule *module, const void *image);
    typedef CUresult (CUDAAPI * ptrCuModuleLoadDataEx)(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues);
    typedef CUresult (CUDAAPI * ptrCuModuleLoadFatBinary)(CUmodule *module, const void *fatCubin);
    typedef CUresult (CUDAAPI * ptrCuModuleUnload)(CUmodule hmod);
    typedef CUresult (CUDAAPI * ptrCuModuleGetFunction)(CUfunction *hfunc, CUmodule hmod, const char *name);
    typedef CUresult (CUDAAPI * ptrCuModuleGetGlobal)(CUdeviceptr *dptr, unsigned int *bytes, CUmodule hmod, const char *name);
    typedef CUresult (CUDAAPI * ptrCuModuleGetTexRef)(CUtexref *pTexRef, CUmodule hmod, const char *name);
    typedef CUresult (CUDAAPI * ptrCuModuleGetSurfRef)(CUsurfref *pSurfRef, CUmodule hmod, const char *name);    
    typedef CUresult (CUDAAPI * ptrCuMemGetInfo)(unsigned int *free, unsigned int *total);
    typedef CUresult (CUDAAPI * ptrCuMemAlloc)( CUdeviceptr *dptr, unsigned int bytesize);
    typedef CUresult (CUDAAPI * ptrCuMemAllocPitch)( CUdeviceptr *dptr, unsigned int *pPitch, unsigned int WidthInBytes, unsigned int Height, unsigned int ElementSizeBytes);
    typedef CUresult (CUDAAPI * ptrCuMemFree)(CUdeviceptr dptr);
    typedef CUresult (CUDAAPI * ptrCuMemGetAddressRange)( CUdeviceptr *pbase, unsigned int *psize, CUdeviceptr dptr );
    typedef CUresult (CUDAAPI * ptrCuMemAllocHost)(void **pp, unsigned int bytesize);
    typedef CUresult (CUDAAPI * ptrCuMemFreeHost)(void *p);
    typedef CUresult (CUDAAPI * ptrCuMemHostAlloc)(void **pp, size_t bytesize, unsigned int Flags );
    typedef CUresult (CUDAAPI * ptrCuMemHostGetDevicePointer)( CUdeviceptr *pdptr, void *p, unsigned int Flags );
    typedef CUresult (CUDAAPI * ptrCuMemHostGetFlags)( unsigned int *pFlags, void *p );
    typedef CUresult (CUDAAPI * ptrCuMemcpyHtoD) (CUdeviceptr dstDevice, const void *srcHost, unsigned int ByteCount );
    typedef CUresult (CUDAAPI * ptrCuMemcpyDtoH) (void *dstHost, CUdeviceptr srcDevice, unsigned int ByteCount );
    typedef CUresult (CUDAAPI * ptrCuMemcpyDtoD) (CUdeviceptr dstDevice, CUdeviceptr srcDevice, unsigned int ByteCount );
    typedef CUresult (CUDAAPI * ptrCuMemcpyDtoA) ( CUarray dstArray, unsigned int dstOffset, CUdeviceptr srcDevice, unsigned int ByteCount );
    typedef CUresult (CUDAAPI * ptrCuMemcpyAtoD) ( CUdeviceptr dstDevice, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount );
    typedef CUresult (CUDAAPI * ptrCuMemcpyHtoA)( CUarray dstArray, unsigned int dstOffset, const void *srcHost, unsigned int ByteCount );
    typedef CUresult (CUDAAPI * ptrCuMemcpyAtoH)( void *dstHost, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount );
    typedef CUresult (CUDAAPI * ptrCuMemcpyAtoA)( CUarray dstArray, unsigned int dstOffset, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount );
    typedef CUresult (CUDAAPI * ptrCuMemcpy2D)( const CUDA_MEMCPY2D *pCopy );
    typedef CUresult (CUDAAPI * ptrCuMemcpy2DUnaligned)( const CUDA_MEMCPY2D *pCopy );
    typedef CUresult (CUDAAPI * ptrCuMemcpy3D)( const CUDA_MEMCPY3D *pCopy );
    typedef CUresult (CUDAAPI * ptrCuMemcpyHtoDAsync) (CUdeviceptr dstDevice, const void *srcHost, unsigned int ByteCount, CUstream hStream );
    typedef CUresult (CUDAAPI * ptrCuMemcpyDtoHAsync) (void *dstHost, CUdeviceptr srcDevice, unsigned int ByteCount, CUstream hStream );
    typedef CUresult (CUDAAPI * ptrCuMemcpyDtoDAsync) (CUdeviceptr dstDevice, CUdeviceptr srcDevice, unsigned int ByteCount, CUstream hStream );
    typedef CUresult (CUDAAPI * ptrCuMemcpyHtoAAsync)( CUarray dstArray, unsigned int dstOffset, const void *srcHost, unsigned int ByteCount, CUstream hStream );
    typedef CUresult (CUDAAPI * ptrCuMemcpyAtoHAsync)( void *dstHost, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount, CUstream hStream );
    typedef CUresult (CUDAAPI * ptrCuMemcpy2DAsync)( const CUDA_MEMCPY2D *pCopy, CUstream hStream );
    typedef CUresult (CUDAAPI * ptrCuMemcpy3DAsync)( const CUDA_MEMCPY3D *pCopy, CUstream hStream );
    typedef CUresult (CUDAAPI * ptrCuMemsetD8)( CUdeviceptr dstDevice, unsigned char uc, unsigned int N );
    typedef CUresult (CUDAAPI * ptrCuMemsetD16)( CUdeviceptr dstDevice, unsigned short us, unsigned int N );
    typedef CUresult (CUDAAPI * ptrCuMemsetD32)( CUdeviceptr dstDevice, unsigned int ui, unsigned int N );
    typedef CUresult (CUDAAPI * ptrCuMemsetD2D8)( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned char uc, unsigned int Width, unsigned int Height );
    typedef CUresult (CUDAAPI * ptrCuMemsetD2D16)( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned short us, unsigned int Width, unsigned int Height );
    typedef CUresult (CUDAAPI * ptrCuMemsetD2D32)( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned int ui, unsigned int Width, unsigned int Height );
    typedef CUresult (CUDAAPI * ptrCuFuncSetBlockShape) (CUfunction hfunc, int x, int y, int z);
    typedef CUresult (CUDAAPI * ptrCuFuncSetSharedSize) (CUfunction hfunc, unsigned int bytes);
    typedef CUresult (CUDAAPI * ptrCuFuncGetAttribute) (int *pi, CUfunction_attribute attrib, CUfunction hfunc);
    typedef CUresult (CUDAAPI * ptrCuFuncSetCacheConfig)(CUfunction hfunc, CUfunc_cache config);
    typedef CUresult (CUDAAPI * ptrCuArrayCreate)( CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray );
    typedef CUresult (CUDAAPI * ptrCuArrayGetDescriptor)( CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor, CUarray hArray );
    typedef CUresult (CUDAAPI * ptrCuArrayDestroy)( CUarray hArray );
    typedef CUresult (CUDAAPI * ptrCuArray3DCreate)( CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray );
    typedef CUresult (CUDAAPI * ptrCuArray3DGetDescriptor)( CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, CUarray hArray );
    typedef CUresult (CUDAAPI * ptrCuTexRefCreate)( CUtexref *pTexRef );
    typedef CUresult (CUDAAPI * ptrCuTexRefDestroy)( CUtexref hTexRef );
    typedef CUresult (CUDAAPI * ptrCuTexRefSetArray)( CUtexref hTexRef, CUarray hArray, unsigned int Flags );
    typedef CUresult (CUDAAPI * ptrCuTexRefSetAddress)( unsigned int *ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, unsigned int bytes );
    typedef CUresult (CUDAAPI * ptrCuTexRefSetAddress2D)( CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR *desc, CUdeviceptr dptr, unsigned int Pitch);
    typedef CUresult (CUDAAPI * ptrCuTexRefSetFormat)( CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents );
    typedef CUresult (CUDAAPI * ptrCuTexRefSetAddressMode)( CUtexref hTexRef, int dim, CUaddress_mode am );
    typedef CUresult (CUDAAPI * ptrCuTexRefSetFilterMode)( CUtexref hTexRef, CUfilter_mode fm );
    typedef CUresult (CUDAAPI * ptrCuTexRefSetFlags)( CUtexref hTexRef, unsigned int Flags );
    typedef CUresult (CUDAAPI * ptrCuTexRefGetAddress)( CUdeviceptr *pdptr, CUtexref hTexRef );
    typedef CUresult (CUDAAPI * ptrCuTexRefGetArray)( CUarray *phArray, CUtexref hTexRef );
    typedef CUresult (CUDAAPI * ptrCuTexRefGetAddressMode)( CUaddress_mode *pam, CUtexref hTexRef, int dim );
    typedef CUresult (CUDAAPI * ptrCuTexRefGetFilterMode)( CUfilter_mode *pfm, CUtexref hTexRef );
    typedef CUresult (CUDAAPI * ptrCuTexRefGetFormat)( CUarray_format *pFormat, int *pNumChannels, CUtexref hTexRef );
    typedef CUresult (CUDAAPI * ptrCuTexRefGetFlags)( unsigned int *pFlags, CUtexref hTexRef );
    typedef CUresult (CUDAAPI * ptrCuSurfRefSetArray)( CUsurfref hSurfRef, CUarray hArray, unsigned int Flags );
    typedef CUresult (CUDAAPI * ptrCuSurfRefGetArray)( CUarray *phArray, CUsurfref hSurfRef );
    typedef CUresult (CUDAAPI * ptrCuParamSetSize) (CUfunction hfunc, unsigned int numbytes);
    typedef CUresult (CUDAAPI * ptrCuParamSeti)    (CUfunction hfunc, int offset, unsigned int value);
    typedef CUresult (CUDAAPI * ptrCuParamSetf)    (CUfunction hfunc, int offset, float value);
    typedef CUresult (CUDAAPI * ptrCuParamSetv)    (CUfunction hfunc, int offset, void *ptr, unsigned int numbytes);
    typedef CUresult (CUDAAPI * ptrCuParamSetTexRef)(CUfunction hfunc, int texunit, CUtexref hTexRef);
    typedef CUresult (CUDAAPI * ptrCuLaunch) ( CUfunction f );
    typedef CUresult (CUDAAPI * ptrCuLaunchGrid) (CUfunction f, int grid_width, int grid_height);
    typedef CUresult (CUDAAPI * ptrCuLaunchGridAsync)( CUfunction f, int grid_width, int grid_height, CUstream hStream );
    typedef CUresult (CUDAAPI * ptrCuEventCreate)( CUevent *phEvent, unsigned int Flags );
    typedef CUresult (CUDAAPI * ptrCuEventRecord)( CUevent hEvent, CUstream hStream );
    typedef CUresult (CUDAAPI * ptrCuEventQuery)( CUevent hEvent );
    typedef CUresult (CUDAAPI * ptrCuEventSynchronize)( CUevent hEvent );
    typedef CUresult (CUDAAPI * ptrCuEventDestroy)( CUevent hEvent );
    typedef CUresult (CUDAAPI * ptrCuEventElapsedTime)( float *pMilliseconds, CUevent hStart, CUevent hEnd );
    typedef CUresult (CUDAAPI * ptrCuStreamCreate)( CUstream *phStream, unsigned int Flags );
    typedef CUresult (CUDAAPI * ptrCuStreamQuery)( CUstream hStream );
    typedef CUresult (CUDAAPI * ptrCuStreamSynchronize)( CUstream hStream );
    typedef CUresult (CUDAAPI * ptrCuStreamDestroy)( CUstream hStream );
    typedef CUresult (CUDAAPI * ptrCuGraphicsUnregisterResource)(CUgraphicsResource resource);
    typedef CUresult (CUDAAPI * ptrCuGraphicsSubResourceGetMappedArray)( CUarray *pArray, CUgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel );
    typedef CUresult (CUDAAPI * ptrCuGraphicsResourceGetMappedPointer)( CUdeviceptr *pDevPtr, unsigned int *pSize, CUgraphicsResource resource );
    typedef CUresult (CUDAAPI * ptrCuGraphicsResourceSetMapFlags)( CUgraphicsResource resource, unsigned int flags ); 
    typedef CUresult (CUDAAPI * ptrCuGraphicsMapResources)( unsigned int count, CUgraphicsResource *resources, CUstream hStream );
    typedef CUresult (CUDAAPI * ptrCuGraphicsUnmapResources)( unsigned int count, CUgraphicsResource *resources, CUstream hStream );
    typedef CUresult (CUDAAPI * ptrCuGetExportTable)( const void **ppExportTable, const CUuuid *pExportTableId );
    typedef CUresult (CUDAAPI * ptrCuCtxSetLimit)(CUlimit limit, size_t value);
    typedef CUresult (CUDAAPI * ptrCuCtxGetLimit)(size_t *pvalue, CUlimit limit);
    

    static CUresult CUDAAPI _cuInit(unsigned int Flags);
    static CUresult CUDAAPI _cuDeviceGet(CUdevice *device, int ordinal);
    static CUresult CUDAAPI _cuDeviceGetCount(int *count);
    static CUresult CUDAAPI _cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev );
    static CUresult CUDAAPI _cuModuleLoad(CUmodule *module, const char *fname);
    static CUresult CUDAAPI _cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name);
    static CUresult CUDAAPI _cuMemAlloc( CUdeviceptr *dptr, unsigned int bytesize);
    static CUresult CUDAAPI _cuMemFree(CUdeviceptr dptr);
    static CUresult CUDAAPI _cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, unsigned int ByteCount );
    static CUresult CUDAAPI _cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, unsigned int ByteCount );
    static CUresult CUDAAPI _cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z);
    static CUresult CUDAAPI _cuParamSetSize(CUfunction hfunc, unsigned int numbytes);
    static CUresult CUDAAPI _cuParamSetv(CUfunction hfunc, int offset, void *ptr, unsigned int numbytes);
    static CUresult CUDAAPI _cuLaunchGrid(CUfunction f, int grid_width, int grid_height);

};

