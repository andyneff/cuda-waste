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


class DLL_API _CUDA
{
private:
    bool did_wrap;
public:
	HMODULE hModule;
public:
	_CUDA() { did_wrap = false; }
    void _CUDA::WrapModule();
    // Driver API.
	typedef CUresult (CUDAAPI * ptr_cuArray3DCreate)(CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray);
	typedef CUresult (CUDAAPI * ptr_cuArray3DCreate_v2)(CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray);
	typedef CUresult (CUDAAPI * ptr_cuArray3DGetDescriptor)( CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, CUarray hArray );
	typedef CUresult (CUDAAPI * ptr_cuArray3DGetDescriptor_v2)( CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, CUarray hArray );
	typedef CUresult (CUDAAPI * ptr_cuArrayCreate)( CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray );
	typedef CUresult (CUDAAPI * ptr_cuArrayCreate_v2)( CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray );
	typedef CUresult (CUDAAPI * ptr_cuArrayDestroy)( CUarray hArray );
	typedef CUresult (CUDAAPI * ptr_cuArrayGetDescriptor)( CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor, CUarray hArray );
	typedef CUresult (CUDAAPI * ptr_cuArrayGetDescriptor_v2)( CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor, CUarray hArray );
	typedef CUresult (CUDAAPI * ptr_cuCtxAttach)(CUcontext *pctx, unsigned int flags);
	typedef CUresult (CUDAAPI * ptr_cuCtxCreate)(CUcontext *pctx, unsigned int flags, CUdevice dev );
	typedef CUresult (CUDAAPI * ptr_cuCtxCreate_v2)(CUcontext *pctx, unsigned int flags, CUdevice dev );
	typedef CUresult (CUDAAPI * ptr_cuCtxDestroy)( CUcontext ctx );
	typedef CUresult (CUDAAPI * ptr_cuCtxDetach)(CUcontext ctx);
	typedef CUresult (CUDAAPI * ptr_cuCtxGetDevice)(CUdevice *device);
	typedef CUresult (CUDAAPI * ptr_cuCtxGetLimit)(size_t *pvalue, CUlimit limit);
	typedef CUresult (CUDAAPI * ptr_cuCtxPopCurrent)( CUcontext *pctx );
	typedef CUresult (CUDAAPI * ptr_cuCtxPushCurrent)( CUcontext ctx );
	typedef CUresult (CUDAAPI * ptr_cuCtxSetLimit)(CUlimit limit, size_t value);
	typedef CUresult (CUDAAPI * ptr_cuCtxSynchronize)(void);
	typedef CUresult (CUDAAPI * ptr_cuDeviceComputeCapability)(int *major, int *minor, CUdevice dev);
	typedef CUresult (CUDAAPI * ptr_cuDeviceGet)(CUdevice *device, int ordinal);
	typedef CUresult (CUDAAPI * ptr_cuDeviceGetAttribute)(int *pi, CUdevice_attribute attrib, CUdevice dev);
	typedef CUresult (CUDAAPI * ptr_cuDeviceGetCount)(int *count);
	typedef CUresult (CUDAAPI * ptr_cuDeviceGetName)(char *name, int len, CUdevice dev);
	typedef CUresult (CUDAAPI * ptr_cuDeviceGetProperties)(CUdevprop *prop, CUdevice dev);
	typedef CUresult (CUDAAPI * ptr_cuDeviceTotalMem)(unsigned int *bytes, CUdevice dev);
	typedef CUresult (CUDAAPI * ptr_cuDeviceTotalMem_v2)(unsigned int *bytes, CUdevice dev);
	typedef CUresult (CUDAAPI * ptr_cuDriverGetVersion)(int *driverVersion);
	typedef CUresult (CUDAAPI * ptr_cuEventCreate)( CUevent *phEvent, unsigned int Flags );
	typedef CUresult (CUDAAPI * ptr_cuEventDestroy)( CUevent hEvent );
	typedef CUresult (CUDAAPI * ptr_cuEventElapsedTime)( float *pMilliseconds, CUevent hStart, CUevent hEnd );
	typedef CUresult (CUDAAPI * ptr_cuEventQuery)( CUevent hEvent );
	typedef CUresult (CUDAAPI * ptr_cuEventRecord)( CUevent hEvent, CUstream hStream );
	typedef CUresult (CUDAAPI * ptr_cuEventSynchronize)( CUevent hEvent );
	typedef CUresult (CUDAAPI * ptr_cuFuncGetAttribute) (int *pi, CUfunction_attribute attrib, CUfunction hfunc);
	typedef CUresult (CUDAAPI * ptr_cuFuncSetBlockShape) (CUfunction hfunc, int x, int y, int z);
	typedef CUresult (CUDAAPI * ptr_cuFuncSetCacheConfig)(CUfunction hfunc, CUfunc_cache config);
	typedef CUresult (CUDAAPI * ptr_cuFuncSetSharedSize) (CUfunction hfunc, unsigned int bytes);
	typedef CUresult (CUDAAPI * ptr_cuGetExportTable)( const void **ppExportTable, const CUuuid *pExportTableId );
	typedef CUresult (CUDAAPI * ptr_cuGraphicsMapResources)( unsigned int count, CUgraphicsResource *resources, CUstream hStream );
	typedef CUresult (CUDAAPI * ptr_cuGraphicsResourceGetMappedPointer)( CUdeviceptr *pDevPtr, unsigned int *pSize, CUgraphicsResource resource );
	typedef CUresult (CUDAAPI * ptr_cuGraphicsResourceGetMappedPointer_v2)( CUdeviceptr *pDevPtr, unsigned int *pSize, CUgraphicsResource resource );
	typedef CUresult (CUDAAPI * ptr_cuGraphicsResourceSetMapFlags)( CUgraphicsResource resource, unsigned int flags ); 
	typedef CUresult (CUDAAPI * ptr_cuGraphicsSubResourceGetMappedArray)( CUarray *pArray, CUgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel );
	typedef CUresult (CUDAAPI * ptr_cuGraphicsUnmapResources)( unsigned int count, CUgraphicsResource *resources, CUstream hStream );
	typedef CUresult (CUDAAPI * ptr_cuGraphicsUnregisterResource)(CUgraphicsResource resource);
	typedef CUresult (CUDAAPI * ptr_cuInit)(unsigned int Flags);
	typedef CUresult (CUDAAPI * ptr_cuLaunch) ( CUfunction f );
	typedef CUresult (CUDAAPI * ptr_cuLaunchGrid) (CUfunction f, int grid_width, int grid_height);
	typedef CUresult (CUDAAPI * ptr_cuLaunchGridAsync)( CUfunction f, int grid_width, int grid_height, CUstream hStream );
	typedef CUresult (CUDAAPI * ptr_cuMemAlloc)( CUdeviceptr *dptr, unsigned int bytesize);
	typedef CUresult (CUDAAPI * ptr_cuMemAlloc_v2)( CUdeviceptr *dptr, unsigned int bytesize);
	typedef CUresult (CUDAAPI * ptr_cuMemAllocHost)(void **pp, unsigned int bytesize);
	typedef CUresult (CUDAAPI * ptr_cuMemAllocHost_v2)(void **pp, unsigned int bytesize);
	typedef CUresult (CUDAAPI * ptr_cuMemAllocPitch)( CUdeviceptr *dptr, unsigned int *pPitch, unsigned int WidthInBytes, unsigned int Height, unsigned int ElementSizeBytes);
	typedef CUresult (CUDAAPI * ptr_cuMemAllocPitch_v2)( CUdeviceptr *dptr, unsigned int *pPitch, unsigned int WidthInBytes, unsigned int Height, unsigned int ElementSizeBytes);
	typedef CUresult (CUDAAPI * ptr_cuMemcpy2D)( const CUDA_MEMCPY2D *pCopy );
	typedef CUresult (CUDAAPI * ptr_cuMemcpy2D_v2)( const CUDA_MEMCPY2D *pCopy );
	typedef CUresult (CUDAAPI * ptr_cuMemcpy2DAsync)( const CUDA_MEMCPY2D *pCopy, CUstream hStream );
	typedef CUresult (CUDAAPI * ptr_cuMemcpy2DAsync_v2)( const CUDA_MEMCPY2D *pCopy, CUstream hStream );
	typedef CUresult (CUDAAPI * ptr_cuMemcpy2DUnaligned)( const CUDA_MEMCPY2D *pCopy );
	typedef CUresult (CUDAAPI * ptr_cuMemcpy2DUnaligned_v2)( const CUDA_MEMCPY2D *pCopy );
	typedef CUresult (CUDAAPI * ptr_cuMemcpy3D)( const CUDA_MEMCPY3D *pCopy );
	typedef CUresult (CUDAAPI * ptr_cuMemcpy3D_v2)( const CUDA_MEMCPY3D *pCopy );
	typedef CUresult (CUDAAPI * ptr_cuMemcpy3DAsync)( const CUDA_MEMCPY3D *pCopy, CUstream hStream );
	typedef CUresult (CUDAAPI * ptr_cuMemcpy3DAsync_v2)( const CUDA_MEMCPY3D *pCopy, CUstream hStream );
	typedef CUresult (CUDAAPI * ptr_cuMemcpyAtoA)( CUarray dstArray, unsigned int dstOffset, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount );
	typedef CUresult (CUDAAPI * ptr_cuMemcpyAtoA_v2)( CUarray dstArray, unsigned int dstOffset, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount );
	typedef CUresult (CUDAAPI * ptr_cuMemcpyAtoD) ( CUdeviceptr dstDevice, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount );
	typedef CUresult (CUDAAPI * ptr_cuMemcpyAtoD_v2) ( CUdeviceptr dstDevice, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount );
	typedef CUresult (CUDAAPI * ptr_cuMemcpyAtoH)( void *dstHost, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount );
	typedef CUresult (CUDAAPI * ptr_cuMemcpyAtoH_v2)( void *dstHost, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount );
	typedef CUresult (CUDAAPI * ptr_cuMemcpyAtoHAsync)( void *dstHost, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount, CUstream hStream );
	typedef CUresult (CUDAAPI * ptr_cuMemcpyAtoHAsync_v2)( void *dstHost, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount, CUstream hStream );
	typedef CUresult (CUDAAPI * ptr_cuMemcpyDtoA) ( CUarray dstArray, unsigned int dstOffset, CUdeviceptr srcDevice, unsigned int ByteCount );
	typedef CUresult (CUDAAPI * ptr_cuMemcpyDtoA_v2) ( CUarray dstArray, unsigned int dstOffset, CUdeviceptr srcDevice, unsigned int ByteCount );
	typedef CUresult (CUDAAPI * ptr_cuMemcpyDtoD) (CUdeviceptr dstDevice, CUdeviceptr srcDevice, unsigned int ByteCount );
	typedef CUresult (CUDAAPI * ptr_cuMemcpyDtoD_v2) (CUdeviceptr dstDevice, CUdeviceptr srcDevice, unsigned int ByteCount );
	typedef CUresult (CUDAAPI * ptr_cuMemcpyDtoDAsync) (CUdeviceptr dstDevice, CUdeviceptr srcDevice, unsigned int ByteCount, CUstream hStream );
	typedef CUresult (CUDAAPI * ptr_cuMemcpyDtoDAsync_v2) (CUdeviceptr dstDevice, CUdeviceptr srcDevice, unsigned int ByteCount, CUstream hStream );
	typedef CUresult (CUDAAPI * ptr_cuMemcpyDtoH) (void *dstHost, CUdeviceptr srcDevice, unsigned int ByteCount );
	typedef CUresult (CUDAAPI * ptr_cuMemcpyDtoH_v2) (void *dstHost, CUdeviceptr srcDevice, unsigned int ByteCount );
	typedef CUresult (CUDAAPI * ptr_cuMemcpyDtoHAsync) (void *dstHost, CUdeviceptr srcDevice, unsigned int ByteCount, CUstream hStream );
	typedef CUresult (CUDAAPI * ptr_cuMemcpyDtoHAsync_v2) (void *dstHost, CUdeviceptr srcDevice, unsigned int ByteCount, CUstream hStream );
	typedef CUresult (CUDAAPI * ptr_cuMemcpyHtoA)( CUarray dstArray, unsigned int dstOffset, const void *srcHost, unsigned int ByteCount );
	typedef CUresult (CUDAAPI * ptr_cuMemcpyHtoA_v2)( CUarray dstArray, unsigned int dstOffset, const void *srcHost, unsigned int ByteCount );
	typedef CUresult (CUDAAPI * ptr_cuMemcpyHtoAAsync)( CUarray dstArray, unsigned int dstOffset, const void *srcHost, unsigned int ByteCount, CUstream hStream );
	typedef CUresult (CUDAAPI * ptr_cuMemcpyHtoAAsync_v2)( CUarray dstArray, unsigned int dstOffset, const void *srcHost, unsigned int ByteCount, CUstream hStream );
	typedef CUresult (CUDAAPI * ptr_cuMemcpyHtoD) (CUdeviceptr dstDevice, const void *srcHost, unsigned int ByteCount );
	typedef CUresult (CUDAAPI * ptr_cuMemcpyHtoD_v2) (CUdeviceptr dstDevice, const void *srcHost, unsigned int ByteCount );
	typedef CUresult (CUDAAPI * ptr_cuMemcpyHtoDAsync) (CUdeviceptr dstDevice, const void *srcHost, unsigned int ByteCount, CUstream hStream );
	typedef CUresult (CUDAAPI * ptr_cuMemcpyHtoDAsync_v2) (CUdeviceptr dstDevice, const void *srcHost, unsigned int ByteCount, CUstream hStream );
	typedef CUresult (CUDAAPI * ptr_cuMemFree)(CUdeviceptr dptr);
	typedef CUresult (CUDAAPI * ptr_cuMemFree_v2)(CUdeviceptr dptr);
	typedef CUresult (CUDAAPI * ptr_cuMemFreeHost)(void *p);
	typedef CUresult (CUDAAPI * ptr_cuMemGetAddressRange)( CUdeviceptr *pbase, unsigned int *psize, CUdeviceptr dptr );
	typedef CUresult (CUDAAPI * ptr_cuMemGetAddressRange_v2)( CUdeviceptr *pbase, unsigned int *psize, CUdeviceptr dptr );
	typedef CUresult (CUDAAPI * ptr_cuMemGetInfo)(unsigned int *free, unsigned int *total);
	typedef CUresult (CUDAAPI * ptr_cuMemGetInfo_v2)(unsigned int *free, unsigned int *total);
	typedef CUresult (CUDAAPI * ptr_cuMemHostAlloc)(void **pp, size_t bytesize, unsigned int Flags );
	typedef CUresult (CUDAAPI * ptr_cuMemHostGetDevicePointer)( CUdeviceptr *pdptr, void *p, unsigned int Flags );
	typedef CUresult (CUDAAPI * ptr_cuMemHostGetDevicePointer_v2)( CUdeviceptr *pdptr, void *p, unsigned int Flags );
	typedef CUresult (CUDAAPI * ptr_cuMemHostGetFlags)( unsigned int *pFlags, void *p );
	typedef CUresult (CUDAAPI * ptr_cuMemsetD16)( CUdeviceptr dstDevice, unsigned short us, unsigned int N );
	typedef CUresult (CUDAAPI * ptr_cuMemsetD16_v2)( CUdeviceptr dstDevice, unsigned short us, unsigned int N );
	typedef CUresult (CUDAAPI * ptr_cuMemsetD2D16)( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned short us, unsigned int Width, unsigned int Height );
	typedef CUresult (CUDAAPI * ptr_cuMemsetD2D16_v2)( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned short us, unsigned int Width, unsigned int Height );
	typedef CUresult (CUDAAPI * ptr_cuMemsetD2D32)( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned int ui, unsigned int Width, unsigned int Height );
	typedef CUresult (CUDAAPI * ptr_cuMemsetD2D32_v2)( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned int ui, unsigned int Width, unsigned int Height );
	typedef CUresult (CUDAAPI * ptr_cuMemsetD2D8)( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned char uc, unsigned int Width, unsigned int Height );
	typedef CUresult (CUDAAPI * ptr_cuMemsetD2D8_v2)( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned char uc, unsigned int Width, unsigned int Height );
	typedef CUresult (CUDAAPI * ptr_cuMemsetD32)( CUdeviceptr dstDevice, unsigned int ui, unsigned int N );
	typedef CUresult (CUDAAPI * ptr_cuMemsetD32_v2)( CUdeviceptr dstDevice, unsigned int ui, unsigned int N );
	typedef CUresult (CUDAAPI * ptr_cuMemsetD8)( CUdeviceptr dstDevice, unsigned char uc, unsigned int N );
	typedef CUresult (CUDAAPI * ptr_cuMemsetD8_v2)( CUdeviceptr dstDevice, unsigned char uc, unsigned int N );
	typedef CUresult (CUDAAPI * ptr_cuModuleGetFunction)(CUfunction *hfunc, CUmodule hmod, const char *name);
	typedef CUresult (CUDAAPI * ptr_cuModuleGetGlobal)(CUdeviceptr *dptr, unsigned int *bytes, CUmodule hmod, const char *name);
	typedef CUresult (CUDAAPI * ptr_cuModuleGetGlobal_v2)(CUdeviceptr *dptr, unsigned int *bytes, CUmodule hmod, const char *name);
	typedef CUresult (CUDAAPI * ptr_cuModuleGetSurfRef)(CUsurfref *pSurfRef, CUmodule hmod, const char *name);    
	typedef CUresult (CUDAAPI * ptr_cuModuleGetTexRef)(CUtexref *pTexRef, CUmodule hmod, const char *name);
	typedef CUresult (CUDAAPI * ptr_cuModuleLoad)(CUmodule *module, const char *fname);
	typedef CUresult (CUDAAPI * ptr_cuModuleLoadData)(CUmodule *module, const void *image);
	typedef CUresult (CUDAAPI * ptr_cuModuleLoadDataEx)(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues);
	typedef CUresult (CUDAAPI * ptr_cuModuleLoadFatBinary)(CUmodule *module, const void *fatCubin);
	typedef CUresult (CUDAAPI * ptr_cuModuleUnload)(CUmodule hmod);
	typedef CUresult (CUDAAPI * ptr_cuParamSetf)    (CUfunction hfunc, int offset, float value);
	typedef CUresult (CUDAAPI * ptr_cuParamSeti)    (CUfunction hfunc, int offset, unsigned int value);
	typedef CUresult (CUDAAPI * ptr_cuParamSetSize) (CUfunction hfunc, unsigned int numbytes);
	typedef CUresult (CUDAAPI * ptr_cuParamSetTexRef)(CUfunction hfunc, int texunit, CUtexref hTexRef);
	typedef CUresult (CUDAAPI * ptr_cuParamSetv)    (CUfunction hfunc, int offset, void *ptr, unsigned int numbytes);
	typedef CUresult (CUDAAPI * ptr_cuStreamCreate)( CUstream *phStream, unsigned int Flags );
	typedef CUresult (CUDAAPI * ptr_cuStreamDestroy)( CUstream hStream );
	typedef CUresult (CUDAAPI * ptr_cuStreamQuery)( CUstream hStream );
	typedef CUresult (CUDAAPI * ptr_cuStreamSynchronize)( CUstream hStream );
	typedef CUresult (CUDAAPI * ptr_cuSurfRefGetArray)( CUarray *phArray, CUsurfref hSurfRef );
	typedef CUresult (CUDAAPI * ptr_cuSurfRefSetArray)( CUsurfref hSurfRef, CUarray hArray, unsigned int Flags );
	typedef CUresult (CUDAAPI * ptr_cuTexRefCreate)( CUtexref *pTexRef );
	typedef CUresult (CUDAAPI * ptr_cuTexRefDestroy)( CUtexref hTexRef );
	typedef CUresult (CUDAAPI * ptr_cuTexRefGetAddress)( CUdeviceptr *pdptr, CUtexref hTexRef );
	typedef CUresult (CUDAAPI * ptr_cuTexRefGetAddress_v2)( CUdeviceptr *pdptr, CUtexref hTexRef );
	typedef CUresult (CUDAAPI * ptr_cuTexRefGetAddressMode)( CUaddress_mode *pam, CUtexref hTexRef, int dim );
	typedef CUresult (CUDAAPI * ptr_cuTexRefGetArray)( CUarray *phArray, CUtexref hTexRef );
	typedef CUresult (CUDAAPI * ptr_cuTexRefGetFilterMode)( CUfilter_mode *pfm, CUtexref hTexRef );
	typedef CUresult (CUDAAPI * ptr_cuTexRefGetFlags)( unsigned int *pFlags, CUtexref hTexRef );
	typedef CUresult (CUDAAPI * ptr_cuTexRefGetFormat)( CUarray_format *pFormat, int *pNumChannels, CUtexref hTexRef );
	typedef CUresult (CUDAAPI * ptr_cuTexRefSetAddress)( unsigned int *ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, unsigned int bytes );
	typedef CUresult (CUDAAPI * ptr_cuTexRefSetAddress_v2)( unsigned int *ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, unsigned int bytes );
	typedef CUresult (CUDAAPI * ptr_cuTexRefSetAddress2D)( CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR *desc, CUdeviceptr dptr, unsigned int Pitch);
	typedef CUresult (CUDAAPI * ptr_cuTexRefSetAddress2D_v2)( CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR *desc, CUdeviceptr dptr, unsigned int Pitch);
	typedef CUresult (CUDAAPI * ptr_cuTexRefSetAddressMode)( CUtexref hTexRef, int dim, CUaddress_mode am );
	typedef CUresult (CUDAAPI * ptr_cuTexRefSetArray)( CUtexref hTexRef, CUarray hArray, unsigned int Flags );
	typedef CUresult (CUDAAPI * ptr_cuTexRefSetFilterMode)( CUtexref hTexRef, CUfilter_mode fm );
	typedef CUresult (CUDAAPI * ptr_cuTexRefSetFlags)( CUtexref hTexRef, unsigned int Flags );
	typedef CUresult (CUDAAPI * ptr_cuTexRefSetFormat)( CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents );

	static CUresult CUDAAPI _cuArray3DCreate(CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray);
	static CUresult CUDAAPI _cuArray3DCreate_v2(CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray);
	static CUresult CUDAAPI _cuArray3DGetDescriptor_v2( CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, CUarray hArray );
	static CUresult CUDAAPI _cuArray3DGetDescriptor( CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, CUarray hArray );
	static CUresult CUDAAPI _cuArrayCreate( CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray );
	static CUresult CUDAAPI _cuArrayCreate_v2( CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray );
	static CUresult CUDAAPI _cuArrayDestroy( CUarray hArray );
	static CUresult CUDAAPI _cuArrayGetDescriptor( CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor, CUarray hArray );
	static CUresult CUDAAPI _cuArrayGetDescriptor_v2( CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor, CUarray hArray );
	static CUresult CUDAAPI _cuCtxAttach(CUcontext *pctx, unsigned int flags);
	static CUresult CUDAAPI _cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev );
	static CUresult CUDAAPI _cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev );
	static CUresult CUDAAPI _cuCtxDestroy( CUcontext ctx );
	static CUresult CUDAAPI _cuCtxDetach(CUcontext ctx);
	static CUresult CUDAAPI _cuCtxGetDevice(CUdevice *device);
	static CUresult CUDAAPI _cuCtxGetLimit(size_t *pvalue, CUlimit limit);
	static CUresult CUDAAPI _cuCtxPopCurrent( CUcontext *pctx );
	static CUresult CUDAAPI _cuCtxPushCurrent( CUcontext ctx );
	static CUresult CUDAAPI _cuCtxSetLimit(CUlimit limit, size_t value);
	static CUresult CUDAAPI _cuCtxSynchronize(void);
	static CUresult CUDAAPI _cuDeviceComputeCapability(int *major, int *minor, CUdevice dev);
	static CUresult CUDAAPI _cuDeviceGet(CUdevice *device, int ordinal);
	static CUresult CUDAAPI _cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev);
	static CUresult CUDAAPI _cuDeviceGetCount(int *count);
	static CUresult CUDAAPI _cuDeviceGetName(char *name, int len, CUdevice dev);
	static CUresult CUDAAPI _cuDeviceGetProperties(CUdevprop *prop, CUdevice dev);
	static CUresult CUDAAPI _cuDeviceTotalMem(unsigned int *bytes, CUdevice dev);
	static CUresult CUDAAPI _cuDeviceTotalMem_v2(unsigned int *bytes, CUdevice dev);
	static CUresult CUDAAPI _cuDriverGetVersion(int *driverVersion);
	static CUresult CUDAAPI _cuEventCreate( CUevent *phEvent, unsigned int Flags );
	static CUresult CUDAAPI _cuEventDestroy( CUevent hEvent );
	static CUresult CUDAAPI _cuEventElapsedTime( float *pMilliseconds, CUevent hStart, CUevent hEnd );
	static CUresult CUDAAPI _cuEventQuery( CUevent hEvent );
	static CUresult CUDAAPI _cuEventRecord( CUevent hEvent, CUstream hStream );
	static CUresult CUDAAPI _cuEventSynchronize( CUevent hEvent );
	static CUresult CUDAAPI _cuFuncGetAttribute (int *pi, CUfunction_attribute attrib, CUfunction hfunc);
	static CUresult CUDAAPI _cuFuncSetBlockShape (CUfunction hfunc, int x, int y, int z);
	static CUresult CUDAAPI _cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config);
	static CUresult CUDAAPI _cuFuncSetSharedSize (CUfunction hfunc, unsigned int bytes);
	static CUresult CUDAAPI _cuGetExportTable( const void **ppExportTable, const CUuuid *pExportTableId );
	static CUresult CUDAAPI _cuGraphicsMapResources( unsigned int count, CUgraphicsResource *resources, CUstream hStream );
	static CUresult CUDAAPI _cuGraphicsResourceGetMappedPointer( CUdeviceptr *pDevPtr, unsigned int *pSize, CUgraphicsResource resource );
	static CUresult CUDAAPI _cuGraphicsResourceGetMappedPointer_v2( CUdeviceptr *pDevPtr, unsigned int *pSize, CUgraphicsResource resource );
	static CUresult CUDAAPI _cuGraphicsResourceSetMapFlags( CUgraphicsResource resource, unsigned int flags );
	static CUresult CUDAAPI _cuGraphicsSubResourceGetMappedArray( CUarray *pArray, CUgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel );
	static CUresult CUDAAPI _cuGraphicsUnmapResources( unsigned int count, CUgraphicsResource *resources, CUstream hStream );
	static CUresult CUDAAPI _cuGraphicsUnregisterResource(CUgraphicsResource resource);
	static CUresult CUDAAPI _cuInit(unsigned int Flags);
	static CUresult CUDAAPI _cuLaunch ( CUfunction f );
	static CUresult CUDAAPI _cuLaunchGrid (CUfunction f, int grid_width, int grid_height);
	static CUresult CUDAAPI _cuLaunchGridAsync( CUfunction f, int grid_width, int grid_height, CUstream hStream );
	static CUresult CUDAAPI _cuMemAlloc( CUdeviceptr *dptr, unsigned int bytesize);
	static CUresult CUDAAPI _cuMemAlloc_v2( CUdeviceptr *dptr, unsigned int bytesize);
	static CUresult CUDAAPI _cuMemAllocHost(void **pp, unsigned int bytesize);
	static CUresult CUDAAPI _cuMemAllocHost_v2(void **pp, unsigned int bytesize);
	static CUresult CUDAAPI _cuMemAllocPitch( CUdeviceptr *dptr, unsigned int *pPitch, unsigned int WidthInBytes, unsigned int Height, unsigned int ElementSizeBytes);
	static CUresult CUDAAPI _cuMemAllocPitch_v2( CUdeviceptr *dptr, unsigned int *pPitch, unsigned int WidthInBytes, unsigned int Height, unsigned int ElementSizeBytes);
	static CUresult CUDAAPI _cuMemcpy2D( const CUDA_MEMCPY2D *pCopy );
	static CUresult CUDAAPI _cuMemcpy2D_v2( const CUDA_MEMCPY2D *pCopy );
	static CUresult CUDAAPI _cuMemcpy2DAsync( const CUDA_MEMCPY2D *pCopy, CUstream hStream );
	static CUresult CUDAAPI _cuMemcpy2DAsync_v2( const CUDA_MEMCPY2D *pCopy, CUstream hStream );
	static CUresult CUDAAPI _cuMemcpy2DUnaligned( const CUDA_MEMCPY2D *pCopy );
	static CUresult CUDAAPI _cuMemcpy2DUnaligned_v2( const CUDA_MEMCPY2D *pCopy );
	static CUresult CUDAAPI _cuMemcpy3D( const CUDA_MEMCPY3D *pCopy );
	static CUresult CUDAAPI _cuMemcpy3D_v2( const CUDA_MEMCPY3D *pCopy );
	static CUresult CUDAAPI _cuMemcpy3DAsync( const CUDA_MEMCPY3D *pCopy, CUstream hStream );
	static CUresult CUDAAPI _cuMemcpy3DAsync_v2( const CUDA_MEMCPY3D *pCopy, CUstream hStream );
	static CUresult CUDAAPI _cuMemcpyAtoA( CUarray dstArray, unsigned int dstOffset, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount );
	static CUresult CUDAAPI _cuMemcpyAtoA_v2( CUarray dstArray, unsigned int dstOffset, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount );
	static CUresult CUDAAPI _cuMemcpyAtoD ( CUdeviceptr dstDevice, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount );
	static CUresult CUDAAPI _cuMemcpyAtoD_v2 ( CUdeviceptr dstDevice, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount );
	static CUresult CUDAAPI _cuMemcpyAtoH( void *dstHost, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount );
	static CUresult CUDAAPI _cuMemcpyAtoH_v2( void *dstHost, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount );
	static CUresult CUDAAPI _cuMemcpyAtoHAsync( void *dstHost, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount, CUstream hStream );
	static CUresult CUDAAPI _cuMemcpyAtoHAsync_v2( void *dstHost, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount, CUstream hStream );
	static CUresult CUDAAPI _cuMemcpyDtoA ( CUarray dstArray, unsigned int dstOffset, CUdeviceptr srcDevice, unsigned int ByteCount );
	static CUresult CUDAAPI _cuMemcpyDtoA_v2 ( CUarray dstArray, unsigned int dstOffset, CUdeviceptr srcDevice, unsigned int ByteCount );
	static CUresult CUDAAPI _cuMemcpyDtoD (CUdeviceptr dstDevice, CUdeviceptr srcDevice, unsigned int ByteCount );
	static CUresult CUDAAPI _cuMemcpyDtoD_v2 (CUdeviceptr dstDevice, CUdeviceptr srcDevice, unsigned int ByteCount );
	static CUresult CUDAAPI _cuMemcpyDtoDAsync (CUdeviceptr dstDevice, CUdeviceptr srcDevice, unsigned int ByteCount, CUstream hStream );
	static CUresult CUDAAPI _cuMemcpyDtoDAsync_v2 (CUdeviceptr dstDevice, CUdeviceptr srcDevice, unsigned int ByteCount, CUstream hStream );
	static CUresult CUDAAPI _cuMemcpyDtoH (void *dstHost, CUdeviceptr srcDevice, unsigned int ByteCount );
	static CUresult CUDAAPI _cuMemcpyDtoH_v2 (void *dstHost, CUdeviceptr srcDevice, unsigned int ByteCount );
	static CUresult CUDAAPI _cuMemcpyDtoHAsync (void *dstHost, CUdeviceptr srcDevice, unsigned int ByteCount, CUstream hStream );
	static CUresult CUDAAPI _cuMemcpyDtoHAsync_v2 (void *dstHost, CUdeviceptr srcDevice, unsigned int ByteCount, CUstream hStream );
	static CUresult CUDAAPI _cuMemcpyHtoA( CUarray dstArray, unsigned int dstOffset, const void *srcHost, unsigned int ByteCount );
	static CUresult CUDAAPI _cuMemcpyHtoA_v2( CUarray dstArray, unsigned int dstOffset, const void *srcHost, unsigned int ByteCount );
	static CUresult CUDAAPI _cuMemcpyHtoAAsync( CUarray dstArray, unsigned int dstOffset, const void *srcHost, unsigned int ByteCount, CUstream hStream );
	static CUresult CUDAAPI _cuMemcpyHtoAAsync_v2( CUarray dstArray, unsigned int dstOffset, const void *srcHost, unsigned int ByteCount, CUstream hStream );
	static CUresult CUDAAPI _cuMemcpyHtoD (CUdeviceptr dstDevice, const void *srcHost, unsigned int ByteCount );
	static CUresult CUDAAPI _cuMemcpyHtoD_v2 (CUdeviceptr dstDevice, const void *srcHost, unsigned int ByteCount );
	static CUresult CUDAAPI _cuMemcpyHtoDAsync (CUdeviceptr dstDevice, const void *srcHost, unsigned int ByteCount, CUstream hStream );
	static CUresult CUDAAPI _cuMemcpyHtoDAsync_v2 (CUdeviceptr dstDevice, const void *srcHost, unsigned int ByteCount, CUstream hStream );
	static CUresult CUDAAPI _cuMemFree(CUdeviceptr dptr);
	static CUresult CUDAAPI _cuMemFree_v2(CUdeviceptr dptr);
	static CUresult CUDAAPI _cuMemFreeHost(void *p);
	static CUresult CUDAAPI _cuMemGetAddressRange( CUdeviceptr *pbase, unsigned int *psize, CUdeviceptr dptr );
	static CUresult CUDAAPI _cuMemGetAddressRange_v2( CUdeviceptr *pbase, unsigned int *psize, CUdeviceptr dptr );
	static CUresult CUDAAPI _cuMemGetInfo(unsigned int *free, unsigned int *total);
	static CUresult CUDAAPI _cuMemGetInfo_v2(unsigned int *free, unsigned int *total);
	static CUresult CUDAAPI _cuMemHostAlloc(void **pp, size_t bytesize, unsigned int Flags );
	static CUresult CUDAAPI _cuMemHostGetDevicePointer( CUdeviceptr *pdptr, void *p, unsigned int Flags );
	static CUresult CUDAAPI _cuMemHostGetDevicePointer_v2( CUdeviceptr *pdptr, void *p, unsigned int Flags );
	static CUresult CUDAAPI _cuMemHostGetFlags( unsigned int *pFlags, void *p );
	static CUresult CUDAAPI _cuMemsetD16( CUdeviceptr dstDevice, unsigned short us, unsigned int N );
	static CUresult CUDAAPI _cuMemsetD16_v2( CUdeviceptr dstDevice, unsigned short us, unsigned int N );
	static CUresult CUDAAPI _cuMemsetD2D16( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned short us, unsigned int Width, unsigned int Height );
	static CUresult CUDAAPI _cuMemsetD2D16_v2( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned short us, unsigned int Width, unsigned int Height );
	static CUresult CUDAAPI _cuMemsetD2D32( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned int ui, unsigned int Width, unsigned int Height );
	static CUresult CUDAAPI _cuMemsetD2D32_v2( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned int ui, unsigned int Width, unsigned int Height );
	static CUresult CUDAAPI _cuMemsetD2D8( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned char uc, unsigned int Width, unsigned int Height );
	static CUresult CUDAAPI _cuMemsetD2D8_v2( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned char uc, unsigned int Width, unsigned int Height );
	static CUresult CUDAAPI _cuMemsetD32( CUdeviceptr dstDevice, unsigned int ui, unsigned int N );
	static CUresult CUDAAPI _cuMemsetD32_v2( CUdeviceptr dstDevice, unsigned int ui, unsigned int N );
	static CUresult CUDAAPI _cuMemsetD8( CUdeviceptr dstDevice, unsigned char uc, unsigned int N );
	static CUresult CUDAAPI _cuMemsetD8_v2( CUdeviceptr dstDevice, unsigned char uc, unsigned int N );
	static CUresult CUDAAPI _cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name);
	static CUresult CUDAAPI _cuModuleGetGlobal(CUdeviceptr *dptr, unsigned int *bytes, CUmodule hmod, const char *name);
	static CUresult CUDAAPI _cuModuleGetGlobal_v2(CUdeviceptr *dptr, unsigned int *bytes, CUmodule hmod, const char *name);
	static CUresult CUDAAPI _cuModuleGetSurfRef(CUsurfref *pSurfRef, CUmodule hmod, const char *name);
	static CUresult CUDAAPI _cuModuleGetTexRef(CUtexref *pTexRef, CUmodule hmod, const char *name);
	static CUresult CUDAAPI _cuModuleLoad(CUmodule *module, const char *fname);
	static CUresult CUDAAPI _cuModuleLoadData(CUmodule *module, const void *image);
	static CUresult CUDAAPI _cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues);
	static CUresult CUDAAPI _cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin);
	static CUresult CUDAAPI _cuModuleUnload(CUmodule hmod);
	static CUresult CUDAAPI _cuParamSetf    (CUfunction hfunc, int offset, float value);
	static CUresult CUDAAPI _cuParamSeti    (CUfunction hfunc, int offset, unsigned int value);
	static CUresult CUDAAPI _cuParamSetSize (CUfunction hfunc, unsigned int numbytes);
	static CUresult CUDAAPI _cuParamSetTexRef(CUfunction hfunc, int texunit, CUtexref hTexRef);
	static CUresult CUDAAPI _cuParamSetv    (CUfunction hfunc, int offset, void *ptr, unsigned int numbytes);
	static CUresult CUDAAPI _cuStreamCreate( CUstream *phStream, unsigned int Flags );
	static CUresult CUDAAPI _cuStreamDestroy( CUstream hStream );
	static CUresult CUDAAPI _cuStreamQuery( CUstream hStream );
	static CUresult CUDAAPI _cuStreamSynchronize( CUstream hStream );
	static CUresult CUDAAPI _cuSurfRefGetArray( CUarray *phArray, CUsurfref hSurfRef );
	static CUresult CUDAAPI _cuSurfRefSetArray( CUsurfref hSurfRef, CUarray hArray, unsigned int Flags );
	static CUresult CUDAAPI _cuTexRefCreate( CUtexref *pTexRef );
	static CUresult CUDAAPI _cuTexRefDestroy( CUtexref hTexRef );
	static CUresult CUDAAPI _cuTexRefGetAddress( CUdeviceptr *pdptr, CUtexref hTexRef );
	static CUresult CUDAAPI _cuTexRefGetAddress_v2( CUdeviceptr *pdptr, CUtexref hTexRef );
	static CUresult CUDAAPI _cuTexRefGetAddressMode( CUaddress_mode *pam, CUtexref hTexRef, int dim );
	static CUresult CUDAAPI _cuTexRefGetArray( CUarray *phArray, CUtexref hTexRef );
	static CUresult CUDAAPI _cuTexRefGetFilterMode( CUfilter_mode *pfm, CUtexref hTexRef );
	static CUresult CUDAAPI _cuTexRefGetFlags( unsigned int *pFlags, CUtexref hTexRef );
	static CUresult CUDAAPI _cuTexRefGetFormat( CUarray_format *pFormat, int *pNumChannels, CUtexref hTexRef );
	static CUresult CUDAAPI _cuTexRefSetAddress( unsigned int *ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, unsigned int bytes );
	static CUresult CUDAAPI _cuTexRefSetAddress_v2( unsigned int *ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, unsigned int bytes );
	static CUresult CUDAAPI _cuTexRefSetAddress2D( CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR *desc, CUdeviceptr dptr, unsigned int Pitch);
	static CUresult CUDAAPI _cuTexRefSetAddress2D_v2( CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR *desc, CUdeviceptr dptr, unsigned int Pitch);
	static CUresult CUDAAPI _cuTexRefSetAddressMode( CUtexref hTexRef, int dim, CUaddress_mode am );
	static CUresult CUDAAPI _cuTexRefSetArray( CUtexref hTexRef, CUarray hArray, unsigned int Flags );
	static CUresult CUDAAPI _cuTexRefSetFilterMode( CUtexref hTexRef, CUfilter_mode fm );
	static CUresult CUDAAPI _cuTexRefSetFlags( CUtexref hTexRef, unsigned int Flags );
	static CUresult CUDAAPI _cuTexRefSetFormat( CUtexref hTexRef, CUarray_format fmt, int static );
};
