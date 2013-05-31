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
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <queue>
#include "tree.h"
#include <process.h>    /* _beginthread, _endthread */
#include "thread.h"
#include "symbol-table.h"
#include "string-table.h"
#include "symbol.h"
#include "constant.h"
#include "types.h"
#include "../wrapper/memdbg.h"
#include "../waste/version.h"
#include "../wrapper/cuda-wrapper.h"
#include "emulated-device.h"
#include "real-device.h"
#include "../wrapper/hook-mgr.h"

cudaError_t REAL_DEVICE::_cudaGetLastError()
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (cu->trace_all_calls)
    {
        char * context = cu->Context();
        (*cu->output_stream) << "cudaGetLastError called, " << context << ".\n\n";
    }

    _CUDA_RUNTIME::typePtrCudaGetLastError proc = (_CUDA_RUNTIME::typePtrCudaGetLastError)cu->hook_manager->FindOriginal(
        (PROC) _CUDA_RUNTIME::_cudaGetLastError);
    return (*proc)();
}

cudaError_t REAL_DEVICE::_cudaMalloc(void ** ptr, size_t size)
{
    void * local = 0;
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    char * context = cu->Context();

    if (cu->trace_all_calls)
    {
        (*cu->output_stream) << "cudaMalloc called, " << context << ".\n\n";
        (*cu->output_stream).flush();
    }

    // Allocate a cuda memory buffer that is "bytes" long plus padding on either side.
    _CUDA_RUNTIME::typePtrCudaMalloc proc = (_CUDA_RUNTIME::typePtrCudaMalloc)cu->hook_manager->FindOriginal((PROC)_CUDA_RUNTIME::_cudaMalloc);
    cudaError_t e1 = (*proc)((void**)&local, size + 2 * cu->padding_size);
    return e1;
}

void** REAL_DEVICE::_cudaRegisterFatBinary(void *fatCubin)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    _CUDA_RUNTIME::typePtrCudaRegisterFatBinary proc = (_CUDA_RUNTIME::typePtrCudaRegisterFatBinary)cu->hook_manager->FindOriginal((PROC)_CUDA_RUNTIME::_cudaRegisterFatBinary);
    return (*proc)(fatCubin);
}

void REAL_DEVICE::ExitHandler()
{
}

CUresult REAL_DEVICE::_cuDeviceComputeCapability(int *major, int *minor, CUdevice dev)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	_CUDA::ptr_cuDeviceComputeCapability proc = (_CUDA::ptr_cuDeviceComputeCapability)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuDeviceComputeCapability);
	return (*proc)(major, minor, dev);
}

CUresult REAL_DEVICE::_cuCtxAttach(CUcontext *pctx, unsigned int flags)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    _CUDA::ptr_cuCtxAttach proc = (_CUDA::ptr_cuCtxAttach)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuCtxAttach);
    return (*proc)(pctx, flags);
}

CUresult REAL_DEVICE::_cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuCtxCreate proc = (_CUDA::ptr_cuCtxCreate)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuCtxCreate);
        return (*proc)(pctx, flags, dev);
}

CUresult REAL_DEVICE::_cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuCtxCreate_v2 proc = (_CUDA::ptr_cuCtxCreate_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuCtxCreate_v2);
        return (*proc)(pctx, flags, dev);
}

CUresult REAL_DEVICE::_cuCtxDestroy( CUcontext ctx )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuCtxDestroy proc = (_CUDA::ptr_cuCtxDestroy)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuCtxDestroy);
        return (*proc)( ctx );
}

CUresult REAL_DEVICE::_cuCtxDetach(CUcontext ctx)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuCtxDetach proc = (_CUDA::ptr_cuCtxDetach)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuCtxDetach);
        return (*proc)(ctx);
}

CUresult REAL_DEVICE::_cuCtxGetDevice(CUdevice *device)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuCtxGetDevice proc = (_CUDA::ptr_cuCtxGetDevice)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuCtxGetDevice);
        return (*proc)(device);
}

CUresult REAL_DEVICE::_cuCtxGetLimit(size_t *pvalue, CUlimit limit)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuCtxGetLimit proc = (_CUDA::ptr_cuCtxGetLimit)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuCtxGetLimit);
        return (*proc)(pvalue, limit);
}

CUresult REAL_DEVICE::_cuCtxPopCurrent( CUcontext *pctx )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuCtxPopCurrent proc = (_CUDA::ptr_cuCtxPopCurrent)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuCtxPopCurrent);
        return (*proc)( pctx );
}

CUresult REAL_DEVICE::_cuCtxPushCurrent( CUcontext ctx )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuCtxPushCurrent proc = (_CUDA::ptr_cuCtxPushCurrent)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuCtxPushCurrent);
        return (*proc)( ctx );
}

CUresult REAL_DEVICE::_cuCtxSetLimit(CUlimit limit, size_t value)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuCtxSetLimit proc = (_CUDA::ptr_cuCtxSetLimit)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuCtxSetLimit);
        return (*proc)(limit, value);
}

CUresult REAL_DEVICE::_cuCtxSynchronize(void)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuCtxSynchronize proc = (_CUDA::ptr_cuCtxSynchronize)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuCtxSynchronize);
        return (*proc)();
}

CUresult REAL_DEVICE::_cuDeviceGet(CUdevice *device, int ordinal)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuDeviceGet proc = (_CUDA::ptr_cuDeviceGet)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuDeviceGet);
        return (*proc)(device, ordinal);
}

CUresult REAL_DEVICE::_cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuDeviceGetAttribute proc = (_CUDA::ptr_cuDeviceGetAttribute)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuDeviceGetAttribute);
        return (*proc)(pi, attrib, dev);
}

CUresult REAL_DEVICE::_cuDeviceGetCount(int *count)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuDeviceGetCount proc = (_CUDA::ptr_cuDeviceGetCount)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuDeviceGetCount);
        return (*proc)(count);
}

CUresult REAL_DEVICE::_cuDeviceGetName(char *name, int len, CUdevice dev)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuDeviceGetName proc = (_CUDA::ptr_cuDeviceGetName)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuDeviceGetName);
        return (*proc)(name, len, dev);
}

CUresult REAL_DEVICE::_cuDeviceGetProperties(CUdevprop *prop, CUdevice dev)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuDeviceGetProperties proc = (_CUDA::ptr_cuDeviceGetProperties)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuDeviceGetProperties);
        return (*proc)(prop, dev);
}

CUresult REAL_DEVICE::_cuDeviceTotalMem(unsigned int *bytes, CUdevice dev)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuDeviceTotalMem proc = (_CUDA::ptr_cuDeviceTotalMem)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuDeviceTotalMem);
        return (*proc)(bytes, dev);
}

CUresult REAL_DEVICE::_cuDeviceTotalMem_v2(unsigned int *bytes, CUdevice dev)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuDeviceTotalMem_v2 proc = (_CUDA::ptr_cuDeviceTotalMem_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuDeviceTotalMem_v2);
        return (*proc)(bytes, dev);
}

CUresult REAL_DEVICE::_cuDriverGetVersion(int *driverVersion)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuDriverGetVersion proc = (_CUDA::ptr_cuDriverGetVersion)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuDriverGetVersion);
        return (*proc)(driverVersion);
}

CUresult REAL_DEVICE::_cuEventCreate( CUevent *phEvent, unsigned int Flags )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuEventCreate proc = (_CUDA::ptr_cuEventCreate)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuEventCreate);
        return (*proc)( phEvent, Flags );
}

CUresult REAL_DEVICE::_cuEventDestroy( CUevent hEvent )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuEventDestroy proc = (_CUDA::ptr_cuEventDestroy)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuEventDestroy);
        return (*proc)( hEvent );
}

CUresult REAL_DEVICE::_cuEventElapsedTime( float *pMilliseconds, CUevent hStart, CUevent hEnd )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuEventElapsedTime proc = (_CUDA::ptr_cuEventElapsedTime)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuEventElapsedTime);
        return (*proc)( pMilliseconds, hStart, hEnd );
}

CUresult REAL_DEVICE::_cuEventQuery( CUevent hEvent )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuEventQuery proc = (_CUDA::ptr_cuEventQuery)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuEventQuery);
        return (*proc)( hEvent );
}

CUresult REAL_DEVICE::_cuEventRecord( CUevent hEvent, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuEventRecord proc = (_CUDA::ptr_cuEventRecord)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuEventRecord);
        return (*proc)( hEvent, hStream );
}

CUresult REAL_DEVICE::_cuEventSynchronize( CUevent hEvent )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuEventSynchronize proc = (_CUDA::ptr_cuEventSynchronize)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuEventSynchronize);
        return (*proc)( hEvent );
}

CUresult REAL_DEVICE::_cuFuncGetAttribute (int *pi, CUfunction_attribute attrib, CUfunction hfunc)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuFuncGetAttribute proc = (_CUDA::ptr_cuFuncGetAttribute)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuFuncGetAttribute);
        return (*proc) (pi, attrib, hfunc);
}

CUresult REAL_DEVICE::_cuFuncSetBlockShape (CUfunction hfunc, int x, int y, int z)
{
    // set up dimensions, shared memory, and stream for the kernel launch.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuFuncSetBlockShape proc = (_CUDA::ptr_cuFuncSetBlockShape)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuFuncSetBlockShape);
        return (*proc)(hfunc, x, y, z);
}

CUresult REAL_DEVICE::_cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuFuncSetCacheConfig proc = (_CUDA::ptr_cuFuncSetCacheConfig)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuFuncSetCacheConfig);
        return (*proc)(hfunc, config);
}

CUresult REAL_DEVICE::_cuFuncSetSharedSize (CUfunction hfunc, unsigned int bytes)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuFuncSetSharedSize proc = (_CUDA::ptr_cuFuncSetSharedSize)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuFuncSetSharedSize);
        return (*proc) (hfunc, bytes);
}

CUresult REAL_DEVICE::_cuGetExportTable( const void **ppExportTable, const CUuuid *pExportTableId )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuGetExportTable proc = (_CUDA::ptr_cuGetExportTable)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuGetExportTable);
        return (*proc)( ppExportTable, pExportTableId );
}

CUresult REAL_DEVICE::_cuGraphicsMapResources( unsigned int count, CUgraphicsResource *resources, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuGraphicsMapResources proc = (_CUDA::ptr_cuGraphicsMapResources)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuGraphicsMapResources);
        return (*proc)( count, resources, hStream );
}

CUresult REAL_DEVICE::_cuGraphicsResourceGetMappedPointer( CUdeviceptr *pDevPtr, unsigned int *pSize, CUgraphicsResource resource )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuGraphicsResourceGetMappedPointer proc = (_CUDA::ptr_cuGraphicsResourceGetMappedPointer)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuGraphicsResourceGetMappedPointer);
        return (*proc)( pDevPtr, pSize, resource );
}

CUresult REAL_DEVICE::_cuGraphicsResourceGetMappedPointer_v2( CUdeviceptr *pDevPtr, unsigned int *pSize, CUgraphicsResource resource )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuGraphicsResourceGetMappedPointer_v2 proc = (_CUDA::ptr_cuGraphicsResourceGetMappedPointer_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuGraphicsResourceGetMappedPointer);
        return (*proc)( pDevPtr, pSize, resource );
}

CUresult REAL_DEVICE::_cuGraphicsResourceSetMapFlags( CUgraphicsResource resource, unsigned int flags )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuGraphicsResourceSetMapFlags proc = (_CUDA::ptr_cuGraphicsResourceSetMapFlags)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuGraphicsResourceSetMapFlags);
        return (*proc)( resource, flags );;
}

CUresult REAL_DEVICE::_cuGraphicsSubResourceGetMappedArray( CUarray *pArray, CUgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuGraphicsSubResourceGetMappedArray proc = (_CUDA::ptr_cuGraphicsSubResourceGetMappedArray)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuGraphicsSubResourceGetMappedArray);
        return (*proc)( pArray, resource, arrayIndex, mipLevel );
}

CUresult REAL_DEVICE::_cuGraphicsUnmapResources( unsigned int count, CUgraphicsResource *resources, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuGraphicsUnmapResources proc = (_CUDA::ptr_cuGraphicsUnmapResources)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuGraphicsUnmapResources);
        return (*proc)( count, resources, hStream );
}

CUresult REAL_DEVICE::_cuGraphicsUnregisterResource(CUgraphicsResource resource)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuGraphicsUnregisterResource proc = (_CUDA::ptr_cuGraphicsUnregisterResource)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuGraphicsUnregisterResource);
        return (*proc)(resource);
}

CUresult REAL_DEVICE::_cuInit(unsigned int Flags)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuInit proc = (_CUDA::ptr_cuInit)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuInit);
        return (*proc)(Flags);
}

CUresult REAL_DEVICE::_cuLaunch ( CUfunction f )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuLaunch proc = (_CUDA::ptr_cuLaunch)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuLaunch);
        return (*proc) ( f );
}

CUresult REAL_DEVICE::_cuLaunchGrid (CUfunction f, int grid_width, int grid_height)
{
    // set up dimensions, shared memory, and stream for the kernel launch.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuLaunchGrid proc = (_CUDA::ptr_cuLaunchGrid)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuLaunchGrid);
        return (*proc)(f, grid_width, grid_height);
}

CUresult REAL_DEVICE::_cuLaunchGridAsync( CUfunction f, int grid_width, int grid_height, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuLaunchGridAsync proc = (_CUDA::ptr_cuLaunchGridAsync)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuLaunchGridAsync);
        return (*proc)( f, grid_width, grid_height, hStream );
}

CUresult REAL_DEVICE::_cuMemAlloc( CUdeviceptr *dptr, unsigned int bytesize)
{
    // Basic, no frills, allocation.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemAlloc proc = (_CUDA::ptr_cuMemAlloc)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemAlloc);
        CUresult e1 = (*proc)(dptr, bytesize);
        return e1;
}

CUresult REAL_DEVICE::_cuMemAlloc_v2( CUdeviceptr *dptr, unsigned int bytesize)
{
    // Basic, no frills, allocation.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemAlloc_v2 proc = (_CUDA::ptr_cuMemAlloc_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemAlloc_v2);
        CUresult e1 = (*proc)(dptr, bytesize);
        return e1;
}

CUresult REAL_DEVICE::_cuMemAllocHost(void **pp, unsigned int bytesize)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemAllocHost proc = (_CUDA::ptr_cuMemAllocHost)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemAllocHost);
        return (*proc)(pp, bytesize);
}

CUresult REAL_DEVICE::_cuMemAllocHost_v2(void **pp, unsigned int bytesize)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemAllocHost_v2 proc = (_CUDA::ptr_cuMemAllocHost_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemAllocHost_v2);
        return (*proc)(pp, bytesize);
}

CUresult REAL_DEVICE::_cuMemAllocPitch( CUdeviceptr *dptr, unsigned int *pPitch, unsigned int WidthInBytes, unsigned int Height, unsigned int ElementSizeBytes)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemAllocPitch proc = (_CUDA::ptr_cuMemAllocPitch)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemAllocPitch);
        return (*proc)( dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);
}

CUresult REAL_DEVICE::_cuMemAllocPitch_v2( CUdeviceptr *dptr, unsigned int *pPitch, unsigned int WidthInBytes, unsigned int Height, unsigned int ElementSizeBytes)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemAllocPitch_v2 proc = (_CUDA::ptr_cuMemAllocPitch_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemAllocPitch_v2);
        return (*proc)( dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);
}

CUresult REAL_DEVICE::_cuMemcpy2D( const CUDA_MEMCPY2D *pCopy )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemcpy2D proc = (_CUDA::ptr_cuMemcpy2D)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpy2D);
        return (*proc)( pCopy );
}

CUresult REAL_DEVICE::_cuMemcpy2D_v2( const CUDA_MEMCPY2D *pCopy )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemcpy2D_v2 proc = (_CUDA::ptr_cuMemcpy2D_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpy2D_v2);
        return (*proc)( pCopy );
}

CUresult REAL_DEVICE::_cuMemcpy2DAsync( const CUDA_MEMCPY2D *pCopy, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemcpy2DAsync proc = (_CUDA::ptr_cuMemcpy2DAsync)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpy2DAsync);
        return (*proc)( pCopy, hStream );
}

CUresult REAL_DEVICE::_cuMemcpy2DAsync_v2( const CUDA_MEMCPY2D *pCopy, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemcpy2DAsync_v2 proc = (_CUDA::ptr_cuMemcpy2DAsync_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpy2DAsync_v2);
        return (*proc)( pCopy, hStream );
}

CUresult REAL_DEVICE::_cuMemcpy2DUnaligned( const CUDA_MEMCPY2D *pCopy )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemcpy2DUnaligned proc = (_CUDA::ptr_cuMemcpy2DUnaligned)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpy2DUnaligned);
        return (*proc)( pCopy );
}

CUresult REAL_DEVICE::_cuMemcpy2DUnaligned_v2( const CUDA_MEMCPY2D *pCopy )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemcpy2DUnaligned_v2 proc = (_CUDA::ptr_cuMemcpy2DUnaligned_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpy2DUnaligned_v2);
        return (*proc)( pCopy );
}

CUresult REAL_DEVICE::_cuMemcpy3D( const CUDA_MEMCPY3D *pCopy )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemcpy3D proc = (_CUDA::ptr_cuMemcpy3D)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpy3D);
        return (*proc)( pCopy );
}

CUresult REAL_DEVICE::_cuMemcpy3D_v2( const CUDA_MEMCPY3D *pCopy )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemcpy3D_v2 proc = (_CUDA::ptr_cuMemcpy3D_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpy3D_v2);
        return (*proc)( pCopy );
}

CUresult REAL_DEVICE::_cuMemcpy3DAsync( const CUDA_MEMCPY3D *pCopy, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemcpy3DAsync proc = (_CUDA::ptr_cuMemcpy3DAsync)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpy3DAsync);
        return (*proc)( pCopy, hStream );
}

CUresult REAL_DEVICE::_cuMemcpy3DAsync_v2( const CUDA_MEMCPY3D *pCopy, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemcpy3DAsync_v2 proc = (_CUDA::ptr_cuMemcpy3DAsync_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpy3DAsync_v2);
        return (*proc)( pCopy, hStream );
}

CUresult REAL_DEVICE::_cuMemcpyAtoA( CUarray dstArray, unsigned int dstOffset, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemcpyAtoA proc = (_CUDA::ptr_cuMemcpyAtoA)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyAtoA);
        return (*proc)( dstArray, dstOffset, srcArray, srcOffset, ByteCount );
}

CUresult REAL_DEVICE::_cuMemcpyAtoA_v2( CUarray dstArray, unsigned int dstOffset, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemcpyAtoA_v2 proc = (_CUDA::ptr_cuMemcpyAtoA_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyAtoA_v2);
        return (*proc)( dstArray, dstOffset, srcArray, srcOffset, ByteCount );
}

CUresult REAL_DEVICE::_cuMemcpyAtoD ( CUdeviceptr dstDevice, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemcpyAtoD proc = (_CUDA::ptr_cuMemcpyAtoD)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyAtoD);
        return (*proc) ( dstDevice, srcArray, srcOffset, ByteCount );
}

CUresult REAL_DEVICE::_cuMemcpyAtoD_v2 ( CUdeviceptr dstDevice, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemcpyAtoD_v2 proc = (_CUDA::ptr_cuMemcpyAtoD_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyAtoD_v2);
        return (*proc) ( dstDevice, srcArray, srcOffset, ByteCount );
}

CUresult REAL_DEVICE::_cuMemcpyAtoH( void *dstHost, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemcpyAtoH proc = (_CUDA::ptr_cuMemcpyAtoH)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyAtoH);
        return (*proc)( dstHost, srcArray, srcOffset, ByteCount );
}

CUresult REAL_DEVICE::_cuMemcpyAtoH_v2( void *dstHost, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemcpyAtoH_v2 proc = (_CUDA::ptr_cuMemcpyAtoH_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyAtoH_v2);
        return (*proc)( dstHost, srcArray, srcOffset, ByteCount );
}

CUresult REAL_DEVICE::_cuMemcpyAtoHAsync( void *dstHost, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemcpyAtoHAsync proc = (_CUDA::ptr_cuMemcpyAtoHAsync)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyAtoHAsync);
        return (*proc)( dstHost, srcArray, srcOffset, ByteCount, hStream );
}

CUresult REAL_DEVICE::_cuMemcpyAtoHAsync_v2( void *dstHost, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemcpyAtoHAsync_v2 proc = (_CUDA::ptr_cuMemcpyAtoHAsync_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyAtoHAsync_v2);
        return (*proc)( dstHost, srcArray, srcOffset, ByteCount, hStream );
}

CUresult REAL_DEVICE::_cuMemcpyDtoA ( CUarray dstArray, unsigned int dstOffset, CUdeviceptr srcDevice, unsigned int ByteCount )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemcpyDtoA proc = (_CUDA::ptr_cuMemcpyDtoA)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyDtoA);
        return (*proc) ( dstArray, dstOffset, srcDevice, ByteCount );
}

CUresult REAL_DEVICE::_cuMemcpyDtoA_v2 ( CUarray dstArray, unsigned int dstOffset, CUdeviceptr srcDevice, unsigned int ByteCount )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemcpyDtoA_v2 proc = (_CUDA::ptr_cuMemcpyDtoA_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyDtoA_v2);
        return (*proc) ( dstArray, dstOffset, srcDevice, ByteCount );
}

CUresult REAL_DEVICE::_cuMemcpyDtoD (CUdeviceptr dstDevice, CUdeviceptr srcDevice, unsigned int ByteCount )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemcpyDtoD proc = (_CUDA::ptr_cuMemcpyDtoD)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyDtoD);
        return (*proc) (dstDevice, srcDevice, ByteCount );
}

CUresult REAL_DEVICE::_cuMemcpyDtoD_v2 (CUdeviceptr dstDevice, CUdeviceptr srcDevice, unsigned int ByteCount )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemcpyDtoD_v2 proc = (_CUDA::ptr_cuMemcpyDtoD_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyDtoD_v2);
        return (*proc) (dstDevice, srcDevice, ByteCount );
}

CUresult REAL_DEVICE::_cuMemcpyDtoDAsync (CUdeviceptr dstDevice, CUdeviceptr srcDevice, unsigned int ByteCount, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemcpyDtoDAsync proc = (_CUDA::ptr_cuMemcpyDtoDAsync)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyDtoDAsync);
        return (*proc) (dstDevice, srcDevice, ByteCount, hStream );
}

CUresult REAL_DEVICE::_cuMemcpyDtoDAsync_v2 (CUdeviceptr dstDevice, CUdeviceptr srcDevice, unsigned int ByteCount, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemcpyDtoDAsync_v2 proc = (_CUDA::ptr_cuMemcpyDtoDAsync_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyDtoDAsync_v2);
        return (*proc) (dstDevice, srcDevice, ByteCount, hStream );
}

CUresult REAL_DEVICE::_cuMemcpyDtoH (void *dstHost, CUdeviceptr srcDevice, unsigned int ByteCount )
{
    // Basic, no frills.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemcpyDtoH proc = (_CUDA::ptr_cuMemcpyDtoH)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyDtoH);
        CUresult e1 = (*proc)(dstHost, srcDevice, ByteCount);
        return e1;
}

CUresult REAL_DEVICE::_cuMemcpyDtoH_v2 (void *dstHost, CUdeviceptr srcDevice, unsigned int ByteCount )
{
    // Basic, no frills.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemcpyDtoH_v2 proc = (_CUDA::ptr_cuMemcpyDtoH_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyDtoH_v2);
        CUresult e1 = (*proc)(dstHost, srcDevice, ByteCount);
        return e1;
}

CUresult REAL_DEVICE::_cuMemcpyDtoHAsync (void *dstHost, CUdeviceptr srcDevice, unsigned int ByteCount, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemcpyDtoHAsync proc = (_CUDA::ptr_cuMemcpyDtoHAsync)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyDtoHAsync);
        return (*proc) (dstHost, srcDevice, ByteCount, hStream );
}

CUresult REAL_DEVICE::_cuMemcpyDtoHAsync_v2 (void *dstHost, CUdeviceptr srcDevice, unsigned int ByteCount, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemcpyDtoHAsync_v2 proc = (_CUDA::ptr_cuMemcpyDtoHAsync_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyDtoHAsync_v2);
        return (*proc) (dstHost, srcDevice, ByteCount, hStream );
}

CUresult REAL_DEVICE::_cuMemcpyHtoA( CUarray dstArray, unsigned int dstOffset, const void *srcHost, unsigned int ByteCount )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemcpyHtoA proc = (_CUDA::ptr_cuMemcpyHtoA)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyHtoA);
        return (*proc)( dstArray, dstOffset, srcHost, ByteCount );
}

CUresult REAL_DEVICE::_cuMemcpyHtoA_v2( CUarray dstArray, unsigned int dstOffset, const void *srcHost, unsigned int ByteCount )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemcpyHtoA_v2 proc = (_CUDA::ptr_cuMemcpyHtoA_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyHtoA_v2);
        return (*proc)( dstArray, dstOffset, srcHost, ByteCount );
}

CUresult REAL_DEVICE::_cuMemcpyHtoAAsync( CUarray dstArray, unsigned int dstOffset, const void *srcHost, unsigned int ByteCount, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemcpyHtoAAsync proc = (_CUDA::ptr_cuMemcpyHtoAAsync)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyHtoAAsync);
        return (*proc)( dstArray, dstOffset, srcHost, ByteCount, hStream );
}

CUresult REAL_DEVICE::_cuMemcpyHtoAAsync_v2( CUarray dstArray, unsigned int dstOffset, const void *srcHost, unsigned int ByteCount, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemcpyHtoAAsync_v2 proc = (_CUDA::ptr_cuMemcpyHtoAAsync_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyHtoAAsync_v2);
        return (*proc)( dstArray, dstOffset, srcHost, ByteCount, hStream );
}

CUresult REAL_DEVICE::_cuMemcpyHtoD (CUdeviceptr dstDevice, const void *srcHost, unsigned int ByteCount )
{
    // Basic, no frills.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemcpyHtoD proc = (_CUDA::ptr_cuMemcpyHtoD)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyHtoD);
        CUresult e1 = (*proc)(dstDevice, srcHost, ByteCount);
        return e1;
}

CUresult REAL_DEVICE::_cuMemcpyHtoD_v2 (CUdeviceptr dstDevice, const void *srcHost, unsigned int ByteCount )
{
    // Basic, no frills.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemcpyHtoD_v2 proc = (_CUDA::ptr_cuMemcpyHtoD_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyHtoD_v2);
        CUresult e1 = (*proc)(dstDevice, srcHost, ByteCount);
        return e1;
}

CUresult REAL_DEVICE::_cuMemcpyHtoDAsync (CUdeviceptr dstDevice, const void *srcHost, unsigned int ByteCount, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemcpyHtoDAsync proc = (_CUDA::ptr_cuMemcpyHtoDAsync)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyHtoDAsync);
        return (*proc) (dstDevice, srcHost, ByteCount, hStream );
}

CUresult REAL_DEVICE::_cuMemcpyHtoDAsync_v2 (CUdeviceptr dstDevice, const void *srcHost, unsigned int ByteCount, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemcpyHtoDAsync_v2 proc = (_CUDA::ptr_cuMemcpyHtoDAsync_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyHtoDAsync_v2);
        return (*proc) (dstDevice, srcHost, ByteCount, hStream );
}

CUresult REAL_DEVICE::_cuMemFree(CUdeviceptr dptr)
{
    // Basic, no frills.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
		_CUDA::ptr_cuMemFree proc = (_CUDA::ptr_cuMemFree)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemFree);
        CUresult e1 = (*proc)(dptr);
        return e1;
}

CUresult REAL_DEVICE::_cuMemFree_v2(CUdeviceptr dptr)
{
    // Basic, no frills.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemFree_v2 proc = (_CUDA::ptr_cuMemFree_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemFree_v2);
        CUresult e1 = (*proc)(dptr);
        return e1;
}

CUresult REAL_DEVICE::_cuMemFreeHost(void *p)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemFreeHost proc = (_CUDA::ptr_cuMemFreeHost)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemFreeHost);
        return (*proc)(p);
}

CUresult REAL_DEVICE::_cuMemGetAddressRange( CUdeviceptr *pbase, unsigned int *psize, CUdeviceptr dptr )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemGetAddressRange proc = (_CUDA::ptr_cuMemGetAddressRange)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemGetAddressRange);
        return (*proc)( pbase, psize, dptr );
}

CUresult REAL_DEVICE::_cuMemGetAddressRange_v2( CUdeviceptr *pbase, unsigned int *psize, CUdeviceptr dptr )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemGetAddressRange_v2 proc = (_CUDA::ptr_cuMemGetAddressRange_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemGetAddressRange_v2);
        return (*proc)( pbase, psize, dptr );
}

CUresult REAL_DEVICE::_cuMemGetInfo(unsigned int *free, unsigned int *total)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemGetInfo proc = (_CUDA::ptr_cuMemGetInfo)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemGetInfo);
        return (*proc)(free, total);
}

CUresult REAL_DEVICE::_cuMemGetInfo_v2(unsigned int *free, unsigned int *total)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemGetInfo_v2 proc = (_CUDA::ptr_cuMemGetInfo_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemGetInfo_v2);
        return (*proc)(free, total);
}

CUresult REAL_DEVICE::_cuMemHostAlloc(void **pp, size_t bytesize, unsigned int Flags )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemHostAlloc proc = (_CUDA::ptr_cuMemHostAlloc)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemHostAlloc);
        return (*proc)(pp, bytesize, Flags );
}

CUresult REAL_DEVICE::_cuMemHostGetDevicePointer( CUdeviceptr *pdptr, void *p, unsigned int Flags )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemHostGetDevicePointer proc = (_CUDA::ptr_cuMemHostGetDevicePointer)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemHostGetDevicePointer);
        return (*proc)( pdptr, p, Flags );
}

CUresult REAL_DEVICE::_cuMemHostGetDevicePointer_v2( CUdeviceptr *pdptr, void *p, unsigned int Flags )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemHostGetDevicePointer_v2 proc = (_CUDA::ptr_cuMemHostGetDevicePointer_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemHostGetDevicePointer_v2);
        return (*proc)( pdptr, p, Flags );
}

CUresult REAL_DEVICE::_cuMemHostGetFlags( unsigned int *pFlags, void *p )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemHostGetFlags proc = (_CUDA::ptr_cuMemHostGetFlags)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemHostGetFlags);
        return (*proc)( pFlags, p );
}

CUresult REAL_DEVICE::_cuMemsetD16( CUdeviceptr dstDevice, unsigned short us, unsigned int N )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemsetD16 proc = (_CUDA::ptr_cuMemsetD16)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemsetD16);
        return (*proc)( dstDevice, us, N );
}

CUresult REAL_DEVICE::_cuMemsetD16_v2( CUdeviceptr dstDevice, unsigned short us, unsigned int N )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemsetD16_v2 proc = (_CUDA::ptr_cuMemsetD16_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemsetD16_v2);
        return (*proc)( dstDevice, us, N );
}

CUresult REAL_DEVICE::_cuMemsetD2D16( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned short us, unsigned int Width, unsigned int Height )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemsetD2D16 proc = (_CUDA::ptr_cuMemsetD2D16)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemsetD2D16);
        return (*proc)( dstDevice, dstPitch, us, Width, Height );
}

CUresult REAL_DEVICE::_cuMemsetD2D16_v2( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned short us, unsigned int Width, unsigned int Height )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemsetD2D16_v2 proc = (_CUDA::ptr_cuMemsetD2D16_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemsetD2D16_v2);
        return (*proc)( dstDevice, dstPitch, us, Width, Height );
}

CUresult REAL_DEVICE::_cuMemsetD2D32( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned int ui, unsigned int Width, unsigned int Height )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemsetD2D32 proc = (_CUDA::ptr_cuMemsetD2D32)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemsetD2D32);
        return (*proc)( dstDevice, dstPitch, ui, Width, Height );
}

CUresult REAL_DEVICE::_cuMemsetD2D32_v2( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned int ui, unsigned int Width, unsigned int Height )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemsetD2D32_v2 proc = (_CUDA::ptr_cuMemsetD2D32_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemsetD2D32_v2);
        return (*proc)( dstDevice, dstPitch, ui, Width, Height );
}

CUresult REAL_DEVICE::_cuMemsetD2D8( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned char uc, unsigned int Width, unsigned int Height )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemsetD2D8 proc = (_CUDA::ptr_cuMemsetD2D8)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemsetD2D8);
        return (*proc)( dstDevice, dstPitch, uc, Width, Height );
}

CUresult REAL_DEVICE::_cuMemsetD2D8_v2( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned char uc, unsigned int Width, unsigned int Height )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemsetD2D8_v2 proc = (_CUDA::ptr_cuMemsetD2D8_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemsetD2D8_v2);
        return (*proc)( dstDevice, dstPitch, uc, Width, Height );
}

CUresult REAL_DEVICE::_cuMemsetD32( CUdeviceptr dstDevice, unsigned int ui, unsigned int N )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemsetD32 proc = (_CUDA::ptr_cuMemsetD32)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemsetD32);
        return (*proc)( dstDevice, ui, N );
}

CUresult REAL_DEVICE::_cuMemsetD32_v2( CUdeviceptr dstDevice, unsigned int ui, unsigned int N )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemsetD32 proc = (_CUDA::ptr_cuMemsetD32)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemsetD32);
        return (*proc)( dstDevice, ui, N );
}

CUresult REAL_DEVICE::_cuMemsetD8( CUdeviceptr dstDevice, unsigned char uc, unsigned int N )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemsetD8 proc = (_CUDA::ptr_cuMemsetD8)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemsetD8);
        return (*proc)( dstDevice, uc, N );
}

CUresult REAL_DEVICE::_cuMemsetD8_v2( CUdeviceptr dstDevice, unsigned char uc, unsigned int N )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuMemsetD8_v2 proc = (_CUDA::ptr_cuMemsetD8_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemsetD8_v2);
        return (*proc)( dstDevice, uc, N );
}

CUresult REAL_DEVICE::_cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuModuleGetFunction proc = (_CUDA::ptr_cuModuleGetFunction)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuModuleGetFunction);
        return (*proc)(hfunc, hmod, name);
}

CUresult REAL_DEVICE::_cuModuleGetGlobal(CUdeviceptr *dptr, unsigned int *bytes, CUmodule hmod, const char *name)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuModuleGetGlobal proc = (_CUDA::ptr_cuModuleGetGlobal)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuModuleGetGlobal);
        return (*proc)(dptr, bytes, hmod, name);
}

CUresult REAL_DEVICE::_cuModuleGetGlobal_v2(CUdeviceptr *dptr, unsigned int *bytes, CUmodule hmod, const char *name)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuModuleGetGlobal_v2 proc = (_CUDA::ptr_cuModuleGetGlobal_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuModuleGetGlobal_v2);
        return (*proc)(dptr, bytes, hmod, name);
}

CUresult REAL_DEVICE::_cuModuleGetSurfRef(CUsurfref *pSurfRef, CUmodule hmod, const char *name)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuModuleGetSurfRef proc = (_CUDA::ptr_cuModuleGetSurfRef)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuModuleGetSurfRef);
        return (*proc)(pSurfRef, hmod, name);   ;
}

CUresult REAL_DEVICE::_cuModuleGetTexRef(CUtexref *pTexRef, CUmodule hmod, const char *name)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuModuleGetTexRef proc = (_CUDA::ptr_cuModuleGetTexRef)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuModuleGetTexRef);
        return (*proc)(pTexRef, hmod, name);
}

CUresult REAL_DEVICE::_cuModuleLoad(CUmodule *module, const char *fname)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuModuleLoad proc = (_CUDA::ptr_cuModuleLoad)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuModuleLoad);
        return (*proc)(module, fname);
}

CUresult REAL_DEVICE::_cuModuleLoadData(CUmodule *module, const void *image)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuModuleLoadData proc = (_CUDA::ptr_cuModuleLoadData)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuModuleLoadData);
        return (*proc)(module, image);
}

CUresult REAL_DEVICE::_cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuModuleLoadDataEx proc = (_CUDA::ptr_cuModuleLoadDataEx)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuModuleLoadDataEx);
        return (*proc)(module, image, numOptions, options, optionValues);
}

CUresult REAL_DEVICE::_cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuModuleLoadFatBinary proc = (_CUDA::ptr_cuModuleLoadFatBinary)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuModuleLoadFatBinary);
        return (*proc)(module, fatCubin);
}

CUresult REAL_DEVICE::_cuModuleUnload(CUmodule hmod)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuModuleUnload proc = (_CUDA::ptr_cuModuleUnload)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuModuleUnload);
        return (*proc)(hmod);
}

CUresult REAL_DEVICE::_cuParamSetf    (CUfunction hfunc, int offset, float value)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuParamSetf proc = (_CUDA::ptr_cuParamSetf)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuParamSetf);
        return (*proc)    (hfunc, offset, value);
}

CUresult REAL_DEVICE::_cuParamSeti    (CUfunction hfunc, int offset, unsigned int value)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuParamSeti proc = (_CUDA::ptr_cuParamSeti)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuParamSeti);
        return (*proc)    (hfunc, offset, value);
}

CUresult REAL_DEVICE::_cuParamSetSize (CUfunction hfunc, unsigned int numbytes)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuParamSetSize proc = (_CUDA::ptr_cuParamSetSize)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuParamSetSize);
        return (*proc)(hfunc, numbytes);
}

CUresult REAL_DEVICE::_cuParamSetTexRef(CUfunction hfunc, int texunit, CUtexref hTexRef)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuParamSetTexRef proc = (_CUDA::ptr_cuParamSetTexRef)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuParamSetTexRef);
        return (*proc)(hfunc, texunit, hTexRef);
}

CUresult REAL_DEVICE::_cuParamSetv(CUfunction hfunc, int offset, void *ptr, unsigned int numbytes)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuParamSetv proc = (_CUDA::ptr_cuParamSetv)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuParamSetv);
        return (*proc)(hfunc, offset, ptr, numbytes);
}

CUresult REAL_DEVICE::_cuStreamCreate( CUstream *phStream, unsigned int Flags )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuStreamCreate proc = (_CUDA::ptr_cuStreamCreate)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuStreamCreate);
        return (*proc)( phStream, Flags );
}

CUresult REAL_DEVICE::_cuStreamDestroy( CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuStreamDestroy proc = (_CUDA::ptr_cuStreamDestroy)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuStreamDestroy);
        return (*proc)( hStream );
}

CUresult REAL_DEVICE::_cuStreamQuery( CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuStreamQuery proc = (_CUDA::ptr_cuStreamQuery)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuStreamQuery);
        return (*proc)( hStream );
}

CUresult REAL_DEVICE::_cuStreamSynchronize( CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuStreamSynchronize proc = (_CUDA::ptr_cuStreamSynchronize)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuStreamSynchronize);
        return (*proc)( hStream );
}

CUresult REAL_DEVICE::_cuSurfRefGetArray( CUarray *phArray, CUsurfref hSurfRef )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuSurfRefGetArray proc = (_CUDA::ptr_cuSurfRefGetArray)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuSurfRefGetArray);
        return (*proc)( phArray, hSurfRef );
}

CUresult REAL_DEVICE::_cuSurfRefSetArray( CUsurfref hSurfRef, CUarray hArray, unsigned int Flags )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuSurfRefSetArray proc = (_CUDA::ptr_cuSurfRefSetArray)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuSurfRefSetArray);
        return (*proc)( hSurfRef, hArray, Flags );
}

CUresult REAL_DEVICE::_cuTexRefCreate( CUtexref *pTexRef )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuTexRefCreate proc = (_CUDA::ptr_cuTexRefCreate)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuTexRefCreate);
        return (*proc)( pTexRef );
}

CUresult REAL_DEVICE::_cuTexRefDestroy( CUtexref hTexRef )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuTexRefDestroy proc = (_CUDA::ptr_cuTexRefDestroy)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuTexRefDestroy);
        return (*proc)( hTexRef );
}

CUresult REAL_DEVICE::_cuTexRefGetAddress( CUdeviceptr *pdptr, CUtexref hTexRef )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuTexRefGetAddress proc = (_CUDA::ptr_cuTexRefGetAddress)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuTexRefGetAddress);
        return (*proc)( pdptr, hTexRef );
}

CUresult REAL_DEVICE::_cuTexRefGetAddress_v2( CUdeviceptr *pdptr, CUtexref hTexRef )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuTexRefGetAddress_v2 proc = (_CUDA::ptr_cuTexRefGetAddress_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuTexRefGetAddress_v2);
        return (*proc)( pdptr, hTexRef );
}

CUresult REAL_DEVICE::_cuTexRefGetAddressMode( CUaddress_mode *pam, CUtexref hTexRef, int dim )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuTexRefGetAddressMode proc = (_CUDA::ptr_cuTexRefGetAddressMode)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuTexRefGetAddressMode);
        return (*proc)( pam, hTexRef, dim );
}

CUresult REAL_DEVICE::_cuTexRefGetArray( CUarray *phArray, CUtexref hTexRef )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuTexRefGetArray proc = (_CUDA::ptr_cuTexRefGetArray)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuTexRefGetArray);
        return (*proc)( phArray, hTexRef );
}

CUresult REAL_DEVICE::_cuTexRefGetFilterMode( CUfilter_mode *pfm, CUtexref hTexRef )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuTexRefGetFilterMode proc = (_CUDA::ptr_cuTexRefGetFilterMode)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuTexRefGetFilterMode);
        return (*proc)( pfm, hTexRef );
}

CUresult REAL_DEVICE::_cuTexRefGetFlags( unsigned int *pFlags, CUtexref hTexRef )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuTexRefGetFlags proc = (_CUDA::ptr_cuTexRefGetFlags)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuTexRefGetFlags);
        return (*proc)( pFlags, hTexRef );
}

CUresult REAL_DEVICE::_cuTexRefGetFormat( CUarray_format *pFormat, int *pNumChannels, CUtexref hTexRef )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuTexRefGetFormat proc = (_CUDA::ptr_cuTexRefGetFormat)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuTexRefGetFormat);
        return (*proc)( pFormat, pNumChannels, hTexRef );
}

CUresult REAL_DEVICE::_cuTexRefSetAddress( unsigned int *ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, unsigned int bytes )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuTexRefSetAddress proc = (_CUDA::ptr_cuTexRefSetAddress)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuTexRefSetAddress);
        return (*proc)( ByteOffset, hTexRef, dptr, bytes );
}

CUresult REAL_DEVICE::_cuTexRefSetAddress_v2( unsigned int *ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, unsigned int bytes )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuTexRefSetAddress_v2 proc = (_CUDA::ptr_cuTexRefSetAddress_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuTexRefSetAddress);
        return (*proc)( ByteOffset, hTexRef, dptr, bytes );
}

CUresult REAL_DEVICE::_cuTexRefSetAddress2D( CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR *desc, CUdeviceptr dptr, unsigned int Pitch)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuTexRefSetAddress2D proc = (_CUDA::ptr_cuTexRefSetAddress2D)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuTexRefSetAddress2D);
        return (*proc)( hTexRef, desc, dptr, Pitch);
}

CUresult REAL_DEVICE::_cuTexRefSetAddress2D_v2( CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR *desc, CUdeviceptr dptr, unsigned int Pitch)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuTexRefSetAddress2D_v2 proc = (_CUDA::ptr_cuTexRefSetAddress2D_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuTexRefSetAddress2D_v2);
        return (*proc)( hTexRef, desc, dptr, Pitch);
}

CUresult REAL_DEVICE::_cuTexRefSetAddressMode( CUtexref hTexRef, int dim, CUaddress_mode am )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuTexRefSetAddressMode proc = (_CUDA::ptr_cuTexRefSetAddressMode)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuTexRefSetAddressMode);
        return (*proc)( hTexRef, dim, am );
}

CUresult REAL_DEVICE::_cuTexRefSetArray( CUtexref hTexRef, CUarray hArray, unsigned int Flags )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuTexRefSetArray proc = (_CUDA::ptr_cuTexRefSetArray)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuTexRefSetArray);
        return (*proc)( hTexRef, hArray, Flags );
}

CUresult REAL_DEVICE::_cuTexRefSetFilterMode( CUtexref hTexRef, CUfilter_mode fm )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuTexRefSetFilterMode proc = (_CUDA::ptr_cuTexRefSetFilterMode)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuTexRefSetFilterMode);
        return (*proc)( hTexRef, fm );
}

CUresult REAL_DEVICE::_cuTexRefSetFlags( CUtexref hTexRef, unsigned int Flags )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuTexRefSetFlags proc = (_CUDA::ptr_cuTexRefSetFlags)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuTexRefSetFlags);
        return (*proc)( hTexRef, Flags );
}

CUresult REAL_DEVICE::_cuTexRefSetFormat( CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
        _CUDA::ptr_cuTexRefSetFormat proc = (_CUDA::ptr_cuTexRefSetFormat)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuTexRefSetFormat);
        return (*proc)( hTexRef, fmt, NumPackedComponents );
}

