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

class SYMBOL_TABLE;
class STRING_TABLE;
class SYMBOL;
class TYPES;

class EMULATOR
{
private:
    EMULATOR();
    ~EMULATOR();
    static EMULATOR * singleton;

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

public:

    int trace_level;

    struct arg
    {
        const void * argument;
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
    std::map<char*, TREE *, ltstr> entry;
    std::map<char*, TREE *, ltstr> func;
    std::map<void*, char*> fun_to_name;
    std::list<TREE *> modules;
    std::list<arg*> arguments;
    config conf;
    void * extern_memory_buffer;

    
public:
    class Unimplemented {
    private:
        char * the_reason;
    public:
        Unimplemented(char * reason)
        {
            the_reason = strdup(reason);
        };
        Unimplemented(const Unimplemented & r)
        {
            this->the_reason = r.the_reason;
        }
        ~Unimplemented()
        {
           free(the_reason);
        };
        const char *ShowReason() const {
            return the_reason;
        }
    };
    void unimplemented(bool condition, char * text);
    void unimplemented(char * text);
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
    void Extract_From_Tree(TREE * node);
    void SetupSingleVar(SYMBOL_TABLE * symbol_table, TREE * var, int * desired_storage_classes, bool externed);
    static EMULATOR * Singleton();
    TREE * Parse(char * module_name, char * source);
    
    // cuda_runtime.h equivalents.
    void ** _cudaRegisterFunction(void * fun, char * name);
    cudaError_t _cudaSetupArgument(const void *arg, size_t size, size_t offset);
    cudaError_t _cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream);
    cudaError_t _cudaThreadSynchronize();
    cudaError_t _cudaLaunch(const char *entry);
    cudaError_t _cudaGetDevice(int * device);
    cudaError_t _cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device);
    cudaError_t _cudaStreamCreate(cudaStream_t *pStream);
    cudaError_t _cudaStreamDestroy(cudaStream_t stream);
    cudaError_t _cudaStreamSynchronize(cudaStream_t stream);
    cudaError_t _cudaStreamQuery(cudaStream_t stream);

    // cuda.h equivalents.
    CUresult _cuLaunchGrid(CUfunction f, int grid_width, int grid_height);
    CUresult _cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name);
    CUresult _cuModuleLoad(CUmodule *module, const char *fname);
    CUresult _cuParamSetSize(CUfunction hfunc, unsigned int numbytes);
    CUresult _cuParamSetv(CUfunction hfunc, int offset, void *ptr, unsigned int numbytes);
    CUresult _cuDeviceGet(CUdevice *device, int ordinal);
    CUresult _cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev);
    CUresult _cuDeviceGetName(char *name, int len, CUdevice dev);
    CUresult _cuDeviceGetProperties(CUdevprop *prop, CUdevice dev);
    CUresult _cuDriverGetVersion(int * driverVersion);
    CUresult _cuGetExportTable( const void **ppExportTable, const CUuuid *pExportTableId );
    CUresult _cuDeviceTotalMem(unsigned int *bytes, CUdevice dev);
    CUresult _cuDeviceComputeCapability(int *major, int *minor, CUdevice dev);
    CUresult _cuCtxAttach(CUcontext *pctx, unsigned int flags);
    CUresult _cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev );
    CUresult _cuCtxDestroy( CUcontext ctx );
    CUresult _cuCtxDetach(CUcontext ctx);


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
};
