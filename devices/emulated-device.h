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
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#include <list>
#include <map>
#include "../ptxp/PtxLexer.h"
#include "tree.h"
#include <cuda.h>
#include <cuda_runtime.h> // cudaError_t, CUDARTAPI, etc.
#include "constant.h"
#include "../wrapper/lock-mgr.h"
#include "../devices/device.h"
#include "../wrapper/cuda-wrapper.h"

class SYMBOL_TABLE;
class STRING_TABLE;
class SYMBOL;
class TYPES;

class EMULATED_DEVICE : DEVICE
{
public:
    static CRIT_SECTION sm_CritSec;
    EMULATED_DEVICE();
    ~EMULATED_DEVICE();
private:
    struct ltstr
    {
        bool operator()(const char* s1, const char* s2) const
        {
            return strcmp(s1, s2) < 0;
        }
    };

    STRING_TABLE * string_table;

    char * device;
	int num_threads;

private:
    struct data
    {
        void * ptr;
        bool is_host;
        int size;
        char * context;
    };
    friend std::vector<data>;
    std::vector<data> * alloc_list;

private:
    int trace_level;

public:
	int TraceLevel()
	{
		return trace_level;
	}

public:
    struct arg
    {
        void * argument;
        size_t size;
        size_t offset;
    };

    struct config
    {
        dim3 gridDim;
        dim3 blockDim;
        size_t sharedMem;
        cudaStream_t stream;
    };

    std::map<void*, char*> fun_to_name;

    typedef struct MOD
    {
        char * module_name;
        TREE * tree;
        std::map<char*, TREE *, ltstr> entry;
        std::map<char*, TREE *, ltstr> func;
    } MOD;
    std::list<MOD*> modules;

    std::list<arg*> arguments;
    config conf;
    void * extern_memory_buffer;
	int max_instruction_thread;
    
public:
    class EMU_ERROR {
    private:
        char * the_reason;
    public:
        EMU_ERROR(char * reason)
        {
            the_reason = strdup(reason);
        };
        EMU_ERROR(const EMU_ERROR & r)
        {
            this->the_reason = r.the_reason;
        }
        ~EMU_ERROR()
        {
           free(the_reason);
        };
        const char *ShowReason() const {
            return the_reason;
        }
    };
    void unimplemented(bool condition, char * text);
    void unimplemented(char * text);
	void Unimplemented();
public:
    TREE * FindBlock(TREE * node);
    TREE * GetInst(TREE * block, int pc);
    void SetupParams(SYMBOL_TABLE * symbol_table, TREE * entry);
    void SetupVariables(SYMBOL_TABLE * symbol_table, TREE * block, int * desired_storage_classes);
    size_t Sizeof(int type);
    int GetSize(TREE * tree_par_register);
    void SetupDimensionLocals();
    void SetupPredefined(SYMBOL_TABLE * symbol_table, dim3 tid, dim3 bid);
    void CreateSymbol(SYMBOL_TABLE * symbol_table, char * name, char * typestring, int type, void * value, size_t size, int storage_class);
    void SetupGotos(SYMBOL_TABLE * symbol_table, TREE * block);
    void Print(TREE * node, int level);
    void PrintName(TREE * node);
    void SetupThreadQueue();
    void ProcessThreadQueue();
    int FindFirstInst(TREE * block, int first);
    CONSTANT Eval(int expected_type, TREE * const_expr);
    SYMBOL_TABLE * PushSymbolTable(SYMBOL_TABLE * parent);
    void ExecuteBlocks(bool do_thread_synch, TREE * code);
    void ExecuteSingleBlock(SYMBOL_TABLE * symbol_table, bool do_thread_synch, TREE * code, int bidx, int bidy, int bidz);
    bool CodeRequiresThreadSynchronization(TREE * code);
    void SetupExternShared(SYMBOL_TABLE * symbol_table, TREE * code);
    void Extract_From_Tree(MOD * module, TREE * node);
    void SetupSingleVar(SYMBOL_TABLE * symbol_table, TREE * var, int * desired_storage_classes, bool externed, size_t total_size);
    MOD * Parse(char * module_name, char * source);
    void ResetArgs();

    // cuda_runtime.h equivalents.
    cudaError_t _cudaBindSurfaceToArray(const struct surfaceReference *surfref, const struct cudaArray *array, const struct cudaChannelFormatDesc *desc);
    cudaError_t _cudaBindTexture(size_t *offset, const struct textureReference *texref, const void *devPtr, const struct cudaChannelFormatDesc *desc, size_t size __dv(UINT_MAX));
    cudaError_t _cudaBindTexture2D(size_t *offset,const struct textureReference *texref,const void *devPtr, const struct cudaChannelFormatDesc *desc,size_t width, size_t height, size_t pitch);
    cudaError_t _cudaBindTextureToArray(const struct textureReference *texref, const struct cudaArray *array, const struct cudaChannelFormatDesc *desc);
    cudaError_t _cudaChooseDevice(int *device, const struct cudaDeviceProp *prop);
    cudaError_t _cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem __dv(0), cudaStream_t stream __dv(0));
    struct cudaChannelFormatDesc _cudaCreateChannelDesc(int x, int y, int z, int w, enum cudaChannelFormatKind e);
    cudaError_t _cudaDriverGetVersion(int *driverVersion);
    cudaError_t _cudaEventCreate(cudaEvent_t *event);
    cudaError_t _cudaEventCreateWithFlags(cudaEvent_t *event, int flags);
    cudaError_t _cudaEventDestroy(cudaEvent_t event);
    cudaError_t _cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end);
    cudaError_t _cudaEventQuery(cudaEvent_t event);
    cudaError_t _cudaEventRecord(cudaEvent_t event, cudaStream_t stream __dv(0));
    cudaError_t _cudaEventSynchronize(cudaEvent_t event);
    cudaError_t _cudaFree(void *);
    cudaError_t _cudaFreeArray(struct cudaArray *array);
    cudaError_t _cudaFreeHost(void *);
    cudaError_t _cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const char *func);
    cudaError_t _cudaFuncSetCacheConfig(const char *func, enum cudaFuncCache cacheConfig);
    cudaError_t _cudaGetChannelDesc(struct cudaChannelFormatDesc *desc, const struct cudaArray *array);
    cudaError_t _cudaGetDevice(int *device);
    cudaError_t _cudaGetDeviceCount(int *count);
    cudaError_t _cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device);
    const char* _cudaGetErrorString(cudaError_t error);
    cudaError_t _cudaGetExportTable(const void **ppExportTable, const cudaUUID_t *pExportTableId);
    cudaError_t _cudaGetLastError(void);
    cudaError_t _cudaGetSurfaceAlignmentOffset(size_t *offset, const struct surfaceReference *surfref);
    cudaError_t _cudaGetSurfaceReference(const struct surfaceReference **surfref, const char *symbol);
    cudaError_t _cudaGetSymbolAddress(void **devPtr, const char *symbol);
    cudaError_t _cudaGetSymbolSize(size_t *size, const char *symbol);
    cudaError_t _cudaGetTextureAlignmentOffset(size_t *offset, const struct textureReference *texref);
    cudaError_t _cudaGetTextureReference(const struct textureReference **texref, const char *symbol);
    cudaError_t _cudaGraphicsMapResources(int count, struct cudaGraphicsResource **resources, cudaStream_t stream __dv(0));
    cudaError_t _cudaGraphicsResourceGetMappedPointer(void **devPtr, size_t *size, struct cudaGraphicsResource *resource);
    cudaError_t _cudaGraphicsResourceSetMapFlags(struct cudaGraphicsResource *resource, unsigned int flags);
    cudaError_t _cudaGraphicsSubResourceGetMappedArray(struct cudaArray **arrayPtr, struct cudaGraphicsResource *resource, unsigned int arrayIndex, unsigned int mipLevel);
    cudaError_t _cudaGraphicsUnmapResources(int count, struct cudaGraphicsResource **resources, cudaStream_t stream __dv(0));
    cudaError_t _cudaGraphicsUnregisterResource(struct cudaGraphicsResource *resource);
    cudaError_t _cudaHostAlloc(void ** ptr, size_t size, unsigned int flags);
    cudaError_t _cudaHostGetDevicePointer(void ** pDevice, void * pHost, unsigned int flags);
    cudaError_t _cudaHostGetFlags(unsigned int *pFlags, void *pHost);
    cudaError_t _cudaLaunch(const char *entry);
    cudaError_t _cudaMalloc(void ** ptr, size_t size);
    cudaError_t _cudaMalloc3DArray(struct cudaPitchedPtr* pitchedDevPtr, struct cudaExtent extent);
    cudaError_t _cudaMallocArray(struct cudaArray **array, const struct cudaChannelFormatDesc *desc, size_t width, size_t height __dv(0), unsigned int flags __dv(0));
    cudaError_t _cudaMallocHost(void **ptr, size_t size);
    cudaError_t _cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height);
    cudaError_t _cudaMemcpy(void * dst, const void * src, size_t count, enum cudaMemcpyKind kind);    
    cudaError_t _cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
    cudaError_t _cudaMemcpy2DArrayToArray(struct cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToDevice));
    cudaError_t _cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
    cudaError_t _cudaMemcpy2DFromArray(void *dst, size_t dpitch, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind);
    cudaError_t _cudaMemcpy2DFromArrayAsync(void *dst, size_t dpitch, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
    cudaError_t _cudaMemcpy2DToArray(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
    cudaError_t _cudaMemcpy2DToArrayAsync(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
    cudaError_t _cudaMemcpy3D(const struct cudaMemcpy3DParms *p);
    cudaError_t _cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *p, cudaStream_t stream __dv(0));
    cudaError_t _cudaMemcpyArrayToArray(struct cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToDevice));
    cudaError_t _cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
    cudaError_t _cudaMemcpyFromArray(void *dst, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind);
    cudaError_t _cudaMemcpyFromArrayAsync(void *dst, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
    cudaError_t _cudaMemcpyFromSymbol(void *dst, const char *symbol, size_t count, size_t offset __dv(0), enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToHost));
    cudaError_t _cudaMemcpyFromSymbolAsync(void *dst, const char *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
    cudaError_t _cudaMemcpyToArray(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind);
    cudaError_t _cudaMemcpyToArrayAsync(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
    cudaError_t _cudaMemcpyToSymbol(const char *symbol, const void *src, size_t count, size_t offset __dv(0), enum cudaMemcpyKind kind __dv(cudaMemcpyHostToDevice));
    cudaError_t _cudaMemcpyToSymbolAsync(const char *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
    cudaError_t _cudaMemGetInfo(size_t *free, size_t *total);
    cudaError_t _cudaMemset(void * devPtr, int value, size_t count);          
    cudaError_t _cudaMemset2D(void *devPtr, size_t pitch, int value, size_t width, size_t height);
    cudaError_t _cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent);
    cudaError_t _cudaPeekAtLastError(void);
    void** _cudaRegisterFatBinary(void *fatCubin);
    void _cudaRegisterFunction(void **fatCubinHandle, const char *hostFun, char *deviceFun, const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize);
    void _cudaRegisterSurface(void **fatCubinHandle, const struct surfaceReference *hostVar, const void **deviceAddress, const char *deviceName, int dim, int ext);
    void _cudaRegisterTexture(void **fatCubinHandle, const struct textureReference *hostVar, const void **deviceAddress, const char *deviceName, int dim, int norm, int ext);
    void _cudaRegisterVar(void **fatCubinHandle, char *hostVar, char *deviceAddress, const char *deviceName, int ext, int size, int constant, int global);
    cudaError_t _cudaRuntimeGetVersion(int *runtimeVersion);
    cudaError_t _cudaSetDevice(int device);
    cudaError_t _cudaSetDeviceFlags(unsigned int flags);
    cudaError_t _cudaSetDoubleForDevice(double *d);
    cudaError_t _cudaSetDoubleForHost(double *d);
    cudaError_t _cudaSetupArgument(const void *arg, size_t size, size_t offset);
    cudaError_t _cudaSetValidDevices(int *device_arr, int len);
    cudaError_t _cudaStreamCreate(cudaStream_t *pStream);
    cudaError_t _cudaStreamDestroy(cudaStream_t stream);
    cudaError_t _cudaStreamQuery(cudaStream_t stream);
    cudaError_t _cudaStreamSynchronize(cudaStream_t stream);
    cudaError_t _cudaThreadExit();
    cudaError_t _cudaThreadGetLimit(enum cudaLimit limit, size_t value);
    cudaError_t _cudaThreadSetLimit(enum cudaLimit limit, size_t value);
    cudaError_t _cudaThreadSynchronize(void);
    cudaError_t _cudaUnbindTexture(const struct textureReference *texref);
    void _cudaUnregisterFatBinary(void **fatCubinHandle);

    // cuda.h equivalents.
    virtual CUresult _cuArray3DCreate(CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray);
    virtual CUresult _cuArray3DCreate_v2(CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray);
    virtual CUresult _cuArray3DGetDescriptor_v2( CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, CUarray hArray );
    virtual CUresult _cuArray3DGetDescriptor( CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, CUarray hArray );
    virtual CUresult _cuArrayCreate( CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray );
    virtual CUresult _cuArrayCreate_v2( CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray );
    virtual CUresult _cuArrayDestroy( CUarray hArray );
    virtual CUresult _cuArrayGetDescriptor( CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor, CUarray hArray );
    virtual CUresult _cuArrayGetDescriptor_v2( CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor, CUarray hArray );
    virtual CUresult _cuCtxAttach(CUcontext *pctx, unsigned int flags);
    virtual CUresult _cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev );
    virtual CUresult _cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev );
    virtual CUresult _cuCtxDestroy( CUcontext ctx );
    virtual CUresult _cuCtxDetach(CUcontext ctx);
    virtual CUresult _cuCtxGetDevice(CUdevice *device);
    virtual CUresult _cuCtxGetLimit(size_t *pvalue, CUlimit limit);
    virtual CUresult _cuCtxPopCurrent( CUcontext *pctx );
    virtual CUresult _cuCtxPushCurrent( CUcontext ctx );
    virtual CUresult _cuCtxSetLimit(CUlimit limit, size_t value);
    virtual CUresult _cuCtxSynchronize(void);
    virtual CUresult _cuDeviceComputeCapability(int *major, int *minor, CUdevice dev);
    virtual CUresult _cuDeviceGet(CUdevice *device, int ordinal);
    virtual CUresult _cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev);
    virtual CUresult _cuDeviceGetCount(int *count);
    virtual CUresult _cuDeviceGetName(char *name, int len, CUdevice dev);
    virtual CUresult _cuDeviceGetProperties(CUdevprop *prop, CUdevice dev);
    virtual CUresult _cuDeviceTotalMem(unsigned int *bytes, CUdevice dev);
    virtual CUresult _cuDeviceTotalMem_v2(unsigned int *bytes, CUdevice dev);
    virtual CUresult _cuDriverGetVersion(int *driverVersion);
    virtual CUresult _cuEventCreate( CUevent *phEvent, unsigned int Flags );
    virtual CUresult _cuEventDestroy( CUevent hEvent );
    virtual CUresult _cuEventElapsedTime( float *pMilliseconds, CUevent hStart, CUevent hEnd );
    virtual CUresult _cuEventQuery( CUevent hEvent );
    virtual CUresult _cuEventRecord( CUevent hEvent, CUstream hStream );
    virtual CUresult _cuEventSynchronize( CUevent hEvent );
    virtual CUresult _cuFuncGetAttribute (int *pi, CUfunction_attribute attrib, CUfunction hfunc);
    virtual CUresult _cuFuncSetBlockShape (CUfunction hfunc, int x, int y, int z);
    virtual CUresult _cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config);
    virtual CUresult _cuFuncSetSharedSize (CUfunction hfunc, unsigned int bytes);
    virtual CUresult _cuGetExportTable( const void **ppExportTable, const CUuuid *pExportTableId );
    virtual CUresult _cuGraphicsMapResources( unsigned int count, CUgraphicsResource *resources, CUstream hStream );
    virtual CUresult _cuGraphicsResourceGetMappedPointer( CUdeviceptr *pDevPtr, unsigned int *pSize, CUgraphicsResource resource );
    virtual CUresult _cuGraphicsResourceGetMappedPointer_v2( CUdeviceptr *pDevPtr, unsigned int *pSize, CUgraphicsResource resource );
    virtual CUresult _cuGraphicsResourceSetMapFlags( CUgraphicsResource resource, unsigned int flags );
    virtual CUresult _cuGraphicsSubResourceGetMappedArray( CUarray *pArray, CUgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel );
    virtual CUresult _cuGraphicsUnmapResources( unsigned int count, CUgraphicsResource *resources, CUstream hStream );
    virtual CUresult _cuGraphicsUnregisterResource(CUgraphicsResource resource);
    virtual CUresult _cuInit(unsigned int Flags);
    virtual CUresult _cuLaunch ( CUfunction f );
    virtual CUresult _cuLaunchGrid (CUfunction f, int grid_width, int grid_height);
    virtual CUresult _cuLaunchGridAsync( CUfunction f, int grid_width, int grid_height, CUstream hStream );
    virtual CUresult _cuMemAlloc( CUdeviceptr *dptr, unsigned int bytesize);
    virtual CUresult _cuMemAlloc_v2( CUdeviceptr *dptr, unsigned int bytesize);
    virtual CUresult _cuMemAllocHost(void **pp, unsigned int bytesize);
    virtual CUresult _cuMemAllocHost_v2(void **pp, unsigned int bytesize);
    virtual CUresult _cuMemAllocPitch( CUdeviceptr *dptr, unsigned int *pPitch, unsigned int WidthInBytes, unsigned int Height, unsigned int ElementSizeBytes);
    virtual CUresult _cuMemAllocPitch_v2( CUdeviceptr *dptr, unsigned int *pPitch, unsigned int WidthInBytes, unsigned int Height, unsigned int ElementSizeBytes);
    virtual CUresult _cuMemcpy2D( const CUDA_MEMCPY2D *pCopy );
    virtual CUresult _cuMemcpy2D_v2( const CUDA_MEMCPY2D *pCopy );
    virtual CUresult _cuMemcpy2DAsync( const CUDA_MEMCPY2D *pCopy, CUstream hStream );
    virtual CUresult _cuMemcpy2DAsync_v2( const CUDA_MEMCPY2D *pCopy, CUstream hStream );
    virtual CUresult _cuMemcpy2DUnaligned( const CUDA_MEMCPY2D *pCopy );
    virtual CUresult _cuMemcpy2DUnaligned_v2( const CUDA_MEMCPY2D *pCopy );
    virtual CUresult _cuMemcpy3D( const CUDA_MEMCPY3D *pCopy );
    virtual CUresult _cuMemcpy3D_v2( const CUDA_MEMCPY3D *pCopy );
    virtual CUresult _cuMemcpy3DAsync( const CUDA_MEMCPY3D *pCopy, CUstream hStream );
    virtual CUresult _cuMemcpy3DAsync_v2( const CUDA_MEMCPY3D *pCopy, CUstream hStream );
    virtual CUresult _cuMemcpyAtoA( CUarray dstArray, unsigned int dstOffset, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount );
    virtual CUresult _cuMemcpyAtoA_v2( CUarray dstArray, unsigned int dstOffset, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount );
    virtual CUresult _cuMemcpyAtoD ( CUdeviceptr dstDevice, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount );
    virtual CUresult _cuMemcpyAtoD_v2 ( CUdeviceptr dstDevice, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount );
    virtual CUresult _cuMemcpyAtoH( void *dstHost, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount );
    virtual CUresult _cuMemcpyAtoH_v2( void *dstHost, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount );
    virtual CUresult _cuMemcpyAtoHAsync( void *dstHost, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount, CUstream hStream );
    virtual CUresult _cuMemcpyAtoHAsync_v2( void *dstHost, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount, CUstream hStream );
    virtual CUresult _cuMemcpyDtoA ( CUarray dstArray, unsigned int dstOffset, CUdeviceptr srcDevice, unsigned int ByteCount );
    virtual CUresult _cuMemcpyDtoA_v2 ( CUarray dstArray, unsigned int dstOffset, CUdeviceptr srcDevice, unsigned int ByteCount );
    virtual CUresult _cuMemcpyDtoD (CUdeviceptr dstDevice, CUdeviceptr srcDevice, unsigned int ByteCount );
    virtual CUresult _cuMemcpyDtoD_v2 (CUdeviceptr dstDevice, CUdeviceptr srcDevice, unsigned int ByteCount );
    virtual CUresult _cuMemcpyDtoDAsync (CUdeviceptr dstDevice, CUdeviceptr srcDevice, unsigned int ByteCount, CUstream hStream );
    virtual CUresult _cuMemcpyDtoDAsync_v2 (CUdeviceptr dstDevice, CUdeviceptr srcDevice, unsigned int ByteCount, CUstream hStream );
    virtual CUresult _cuMemcpyDtoH (void *dstHost, CUdeviceptr srcDevice, unsigned int ByteCount );
    virtual CUresult _cuMemcpyDtoH_v2 (void *dstHost, CUdeviceptr srcDevice, unsigned int ByteCount );
    virtual CUresult _cuMemcpyDtoHAsync (void *dstHost, CUdeviceptr srcDevice, unsigned int ByteCount, CUstream hStream );
    virtual CUresult _cuMemcpyDtoHAsync_v2 (void *dstHost, CUdeviceptr srcDevice, unsigned int ByteCount, CUstream hStream );
    virtual CUresult _cuMemcpyHtoA( CUarray dstArray, unsigned int dstOffset, const void *srcHost, unsigned int ByteCount );
    virtual CUresult _cuMemcpyHtoA_v2( CUarray dstArray, unsigned int dstOffset, const void *srcHost, unsigned int ByteCount );
    virtual CUresult _cuMemcpyHtoAAsync( CUarray dstArray, unsigned int dstOffset, const void *srcHost, unsigned int ByteCount, CUstream hStream );
    virtual CUresult _cuMemcpyHtoAAsync_v2( CUarray dstArray, unsigned int dstOffset, const void *srcHost, unsigned int ByteCount, CUstream hStream );
    virtual CUresult _cuMemcpyHtoD (CUdeviceptr dstDevice, const void *srcHost, unsigned int ByteCount );
    virtual CUresult _cuMemcpyHtoD_v2 (CUdeviceptr dstDevice, const void *srcHost, unsigned int ByteCount );
    virtual CUresult _cuMemcpyHtoDAsync (CUdeviceptr dstDevice, const void *srcHost, unsigned int ByteCount, CUstream hStream );
    virtual CUresult _cuMemcpyHtoDAsync_v2 (CUdeviceptr dstDevice, const void *srcHost, unsigned int ByteCount, CUstream hStream );
    virtual CUresult _cuMemFree(CUdeviceptr dptr);
    virtual CUresult _cuMemFree_v2(CUdeviceptr dptr);
    virtual CUresult _cuMemFreeHost(void *p);
    virtual CUresult _cuMemGetAddressRange( CUdeviceptr *pbase, unsigned int *psize, CUdeviceptr dptr );
    virtual CUresult _cuMemGetAddressRange_v2( CUdeviceptr *pbase, unsigned int *psize, CUdeviceptr dptr );
    virtual CUresult _cuMemGetInfo(unsigned int *free, unsigned int *total);
    virtual CUresult _cuMemGetInfo_v2(unsigned int *free, unsigned int *total);
    virtual CUresult _cuMemHostAlloc(void **pp, size_t bytesize, unsigned int Flags );
    virtual CUresult _cuMemHostGetDevicePointer( CUdeviceptr *pdptr, void *p, unsigned int Flags );
    virtual CUresult _cuMemHostGetDevicePointer_v2( CUdeviceptr *pdptr, void *p, unsigned int Flags );
    virtual CUresult _cuMemHostGetFlags( unsigned int *pFlags, void *p );
    virtual CUresult _cuMemsetD16( CUdeviceptr dstDevice, unsigned short us, unsigned int N );
    virtual CUresult _cuMemsetD16_v2( CUdeviceptr dstDevice, unsigned short us, unsigned int N );
    virtual CUresult _cuMemsetD2D16( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned short us, unsigned int Width, unsigned int Height );
    virtual CUresult _cuMemsetD2D16_v2( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned short us, unsigned int Width, unsigned int Height );
    virtual CUresult _cuMemsetD2D32( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned int ui, unsigned int Width, unsigned int Height );
    virtual CUresult _cuMemsetD2D32_v2( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned int ui, unsigned int Width, unsigned int Height );
    virtual CUresult _cuMemsetD2D8( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned char uc, unsigned int Width, unsigned int Height );
    virtual CUresult _cuMemsetD2D8_v2( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned char uc, unsigned int Width, unsigned int Height );
    virtual CUresult _cuMemsetD32( CUdeviceptr dstDevice, unsigned int ui, unsigned int N );
    virtual CUresult _cuMemsetD32_v2( CUdeviceptr dstDevice, unsigned int ui, unsigned int N );
    virtual CUresult _cuMemsetD8( CUdeviceptr dstDevice, unsigned char uc, unsigned int N );
    virtual CUresult _cuMemsetD8_v2( CUdeviceptr dstDevice, unsigned char uc, unsigned int N );
    virtual CUresult _cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name);
    virtual CUresult _cuModuleGetGlobal(CUdeviceptr *dptr, unsigned int *bytes, CUmodule hmod, const char *name);
    virtual CUresult _cuModuleGetGlobal_v2(CUdeviceptr *dptr, unsigned int *bytes, CUmodule hmod, const char *name);
    virtual CUresult _cuModuleGetSurfRef(CUsurfref *pSurfRef, CUmodule hmod, const char *name);
    virtual CUresult _cuModuleGetTexRef(CUtexref *pTexRef, CUmodule hmod, const char *name);
    virtual CUresult _cuModuleLoad(CUmodule *module, const char *fname);
    virtual CUresult _cuModuleLoadData(CUmodule *module, const void *image);
    virtual CUresult _cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues);
    virtual CUresult _cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin);
    virtual CUresult _cuModuleUnload(CUmodule hmod);
    virtual CUresult _cuParamSetf    (CUfunction hfunc, int offset, float value);
    virtual CUresult _cuParamSeti    (CUfunction hfunc, int offset, unsigned int value);
    virtual CUresult _cuParamSetSize (CUfunction hfunc, unsigned int numbytes);
    virtual CUresult _cuParamSetTexRef(CUfunction hfunc, int texunit, CUtexref hTexRef);
    virtual CUresult _cuParamSetv    (CUfunction hfunc, int offset, void *ptr, unsigned int numbytes);
    virtual CUresult _cuStreamCreate( CUstream *phStream, unsigned int Flags );
    virtual CUresult _cuStreamDestroy( CUstream hStream );
    virtual CUresult _cuStreamQuery( CUstream hStream );
    virtual CUresult _cuStreamSynchronize( CUstream hStream );
    virtual CUresult _cuSurfRefGetArray( CUarray *phArray, CUsurfref hSurfRef );
    virtual CUresult _cuSurfRefSetArray( CUsurfref hSurfRef, CUarray hArray, unsigned int Flags );
    virtual CUresult _cuTexRefCreate( CUtexref *pTexRef );
    virtual CUresult _cuTexRefDestroy( CUtexref hTexRef );
    virtual CUresult _cuTexRefGetAddress( CUdeviceptr *pdptr, CUtexref hTexRef );
    virtual CUresult _cuTexRefGetAddress_v2( CUdeviceptr *pdptr, CUtexref hTexRef );
    virtual CUresult _cuTexRefGetAddressMode( CUaddress_mode *pam, CUtexref hTexRef, int dim );
    virtual CUresult _cuTexRefGetArray( CUarray *phArray, CUtexref hTexRef );
    virtual CUresult _cuTexRefGetFilterMode( CUfilter_mode *pfm, CUtexref hTexRef );
    virtual CUresult _cuTexRefGetFlags( unsigned int *pFlags, CUtexref hTexRef );
    virtual CUresult _cuTexRefGetFormat( CUarray_format *pFormat, int *pNumChannels, CUtexref hTexRef );
    virtual CUresult _cuTexRefSetAddress( unsigned int *ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, unsigned int bytes );
    virtual CUresult _cuTexRefSetAddress_v2( unsigned int *ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, unsigned int bytes );
    virtual CUresult _cuTexRefSetAddress2D( CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR *desc, CUdeviceptr dptr, unsigned int Pitch);
    virtual CUresult _cuTexRefSetAddress2D_v2( CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR *desc, CUdeviceptr dptr, unsigned int Pitch);
    virtual CUresult _cuTexRefSetAddressMode( CUtexref hTexRef, int dim, CUaddress_mode am );
    virtual CUresult _cuTexRefSetArray( CUtexref hTexRef, CUarray hArray, unsigned int Flags );
    virtual CUresult _cuTexRefSetFilterMode( CUtexref hTexRef, CUfilter_mode fm );
    virtual CUresult _cuTexRefSetFlags( CUtexref hTexRef, unsigned int Flags );
    virtual CUresult _cuTexRefSetFormat( CUtexref hTexRef, CUarray_format fmt, int );


    // Generic setup, execution.
    void ConfigureBlock(dim3 dim);
    void ConfigureGrid(dim3 dim);
    void ConfigureSharedMemory(size_t sharedMem);
    void ConfigureStream(cudaStream_t stream);
    void Execute(TREE * entry);

    // Options for emulator.
    void SetTrace(int level);
    char * StringTableEntry(char * text);
    void RunDevice(char * device);
	void SetEmulationThreads(int i);
	    
	enum return_type {
        NOT_OK = 0,
        OK = 1
    };

    return_type CheckOverwrite();
    return_type CheckSinglePtrOverwrite(const data * d);
    void ExitHandler();
private:
	static std::list<EMULATED_DEVICE*> all_emulated_devices;
public:
    bool IsBadPointer(const void * ptr);
    int FindAllocatedBlock(const void * pointer);
};

