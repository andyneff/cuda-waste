// nvcuda.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include <cuda.h>

///////////////////////////////////////////////////////////////////////////////
//
// This project contains definitions for generating a non-functional version
// of nvcuda.dll, the CUDA Driver API.  This DLL can be built if you are
// working on a system that does not have the Nvidia drivers installed.
// Unfortunately, you cannot install the Nvidia drivers if you do not have
// Nvidia hardware present.  This DLL stub fixes that problem.
//
///////////////////////////////////////////////////////////////////////////////

CUresult CUDAAPI cuInit(unsigned int Flags)
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDriverGetVersion(int *driverVersion)
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGet(CUdevice *device, int ordinal)
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetCount(int *count)
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetName(char *name, int len, CUdevice dev)
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceComputeCapability(int *major, int *minor, CUdevice dev)
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceTotalMem(unsigned int *bytes, CUdevice dev)
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetProperties(CUdevprop *prop, CUdevice dev)
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev)
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxDestroy( CUcontext ctx )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxAttach(CUcontext *pctx, unsigned int flags)
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxDetach(CUcontext ctx)
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxPushCurrent( CUcontext ctx )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxPopCurrent( CUcontext *pctx )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxGetDevice(CUdevice *device)
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxSynchronize(void)
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuModuleLoad(CUmodule *module, const char *fname)
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuModuleLoadData(CUmodule *module, const void *image)
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues)
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin)
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuModuleUnload(CUmodule hmod)
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name)
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuModuleGetGlobal(CUdeviceptr *dptr, unsigned int *bytes, CUmodule hmod, const char *name)
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuModuleGetTexRef(CUtexref *pTexRef, CUmodule hmod, const char *name)
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuModuleGetSurfRef(CUsurfref *pSurfRef, CUmodule hmod, const char *name)
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemGetInfo(unsigned int *free, unsigned int *total)
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemAlloc( CUdeviceptr *dptr, unsigned int bytesize)
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemAllocPitch( CUdeviceptr *dptr, unsigned int *pPitch, unsigned int WidthInBytes, unsigned int Height, unsigned int ElementSizeBytes)
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemFree(CUdeviceptr dptr)
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemGetAddressRange( CUdeviceptr *pbase, unsigned int *psize, CUdeviceptr dptr )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemAllocHost(void **pp, unsigned int bytesize)
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemFreeHost(void *p)
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemHostAlloc(void **pp, size_t bytesize, unsigned int Flags )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemHostGetDevicePointer( CUdeviceptr *pdptr, void *p, unsigned int Flags )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemHostGetFlags( unsigned int *pFlags, void *p )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyHtoD (CUdeviceptr dstDevice, const void *srcHost, unsigned int ByteCount )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyDtoH (void *dstHost, CUdeviceptr srcDevice, unsigned int ByteCount )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyDtoD (CUdeviceptr dstDevice, CUdeviceptr srcDevice, unsigned int ByteCount )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyDtoA ( CUarray dstArray, unsigned int dstOffset, CUdeviceptr srcDevice, unsigned int ByteCount )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyAtoD ( CUdeviceptr dstDevice, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyHtoA( CUarray dstArray, unsigned int dstOffset, const void *srcHost, unsigned int ByteCount )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyAtoH( void *dstHost, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyAtoA( CUarray dstArray, unsigned int dstOffset, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpy2D( const CUDA_MEMCPY2D *pCopy )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpy2DUnaligned( const CUDA_MEMCPY2D *pCopy )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpy3D( const CUDA_MEMCPY3D *pCopy )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyHtoDAsync (CUdeviceptr dstDevice, const void *srcHost, unsigned int ByteCount, CUstream hStream )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyDtoHAsync (void *dstHost, CUdeviceptr srcDevice, unsigned int ByteCount, CUstream hStream )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyDtoDAsync (CUdeviceptr dstDevice, CUdeviceptr srcDevice, unsigned int ByteCount, CUstream hStream )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyHtoAAsync( CUarray dstArray, unsigned int dstOffset, const void *srcHost, unsigned int ByteCount, CUstream hStream )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyAtoHAsync( void *dstHost, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount, CUstream hStream )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpy2DAsync( const CUDA_MEMCPY2D *pCopy, CUstream hStream )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpy3DAsync( const CUDA_MEMCPY3D *pCopy, CUstream hStream )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemsetD8( CUdeviceptr dstDevice, unsigned char uc, unsigned int N )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemsetD16( CUdeviceptr dstDevice, unsigned short us, unsigned int N )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemsetD32( CUdeviceptr dstDevice, unsigned int ui, unsigned int N )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemsetD2D8( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned char uc, unsigned int Width, unsigned int Height )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemsetD2D16( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned short us, unsigned int Width, unsigned int Height )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemsetD2D32( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned int ui, unsigned int Width, unsigned int Height )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuFuncSetBlockShape (CUfunction hfunc, int x, int y, int z)
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuFuncSetSharedSize (CUfunction hfunc, unsigned int bytes)
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuFuncGetAttribute (int *pi, CUfunction_attribute attrib, CUfunction hfunc)
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config)
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuArrayCreate( CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuArrayGetDescriptor( CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor, CUarray hArray )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuArrayDestroy( CUarray hArray )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuArray3DCreate( CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuArray3DGetDescriptor( CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, CUarray hArray )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefCreate( CUtexref *pTexRef )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefDestroy( CUtexref hTexRef )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefSetArray( CUtexref hTexRef, CUarray hArray, unsigned int Flags )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefSetAddress( unsigned int *ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, unsigned int bytes )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefSetAddress2D( CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR *desc, CUdeviceptr dptr, unsigned int Pitch)
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefSetFormat( CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefSetAddressMode( CUtexref hTexRef, int dim, CUaddress_mode am )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefSetFilterMode( CUtexref hTexRef, CUfilter_mode fm )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefSetFlags( CUtexref hTexRef, unsigned int Flags )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefGetAddress( CUdeviceptr *pdptr, CUtexref hTexRef )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefGetArray( CUarray *phArray, CUtexref hTexRef )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefGetAddressMode( CUaddress_mode *pam, CUtexref hTexRef, int dim )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefGetFilterMode( CUfilter_mode *pfm, CUtexref hTexRef )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefGetFormat( CUarray_format *pFormat, int *pNumChannels, CUtexref hTexRef )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefGetFlags( unsigned int *pFlags, CUtexref hTexRef )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuSurfRefSetArray( CUsurfref hSurfRef, CUarray hArray, unsigned int Flags )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuSurfRefGetArray( CUarray *phArray, CUsurfref hSurfRef )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuParamSetSize (CUfunction hfunc, unsigned int numbytes)
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuParamSeti    (CUfunction hfunc, int offset, unsigned int value)
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuParamSetf    (CUfunction hfunc, int offset, float value)
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuParamSetv    (CUfunction hfunc, int offset, void *ptr, unsigned int numbytes)
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuParamSetTexRef(CUfunction hfunc, int texunit, CUtexref hTexRef)
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuLaunch ( CUfunction f )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuLaunchGrid (CUfunction f, int grid_width, int grid_height)
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuLaunchGridAsync( CUfunction f, int grid_width, int grid_height, CUstream hStream )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuEventCreate( CUevent *phEvent, unsigned int Flags )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuEventRecord( CUevent hEvent, CUstream hStream )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuEventQuery( CUevent hEvent )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuEventSynchronize( CUevent hEvent )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuEventDestroy( CUevent hEvent )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuEventElapsedTime( float *pMilliseconds, CUevent hStart, CUevent hEnd )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamCreate( CUstream *phStream, unsigned int Flags )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamQuery( CUstream hStream )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamSynchronize( CUstream hStream )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamDestroy( CUstream hStream )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphicsUnregisterResource(CUgraphicsResource resource)
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphicsSubResourceGetMappedArray( CUarray *pArray, CUgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphicsResourceGetMappedPointer( CUdeviceptr *pDevPtr, unsigned int *pSize, CUgraphicsResource resource )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphicsResourceSetMapFlags( CUgraphicsResource resource, unsigned int flags )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphicsMapResources( unsigned int count, CUgraphicsResource *resources, CUstream hStream )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphicsUnmapResources( unsigned int count, CUgraphicsResource *resources, CUstream hStream )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGetExportTable( const void **ppExportTable, const CUuuid *pExportTableId )
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxSetLimit(CUlimit limit, size_t value)
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxGetLimit(size_t *pvalue, CUlimit limit)
{
    return CUDA_SUCCESS;
}

