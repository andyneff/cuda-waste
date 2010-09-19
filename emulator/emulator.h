#pragma once
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#include <list>
#include <map>
#include "../ptxp/PtxLexer.h"
//#include "../ptxp/PtxParser.h"
#include "tree.h"
#include <cuda.h>
#include <cuda_runtime.h> // cudaError_t, CUDARTAPI, etc.

class CUDA_EMULATOR
{
private:
    struct ltstr
    {
        bool operator()(const char* s1, const char* s2) const
        {
            return strcmp(s1, s2) < 0;
        }
    };

    typedef union TYPES {
        signed __int64 s64;
        signed __int32 s32;
        signed __int16 s16;;
        signed __int8 s8;
        unsigned __int64 u64;
        unsigned __int32 u32;
        unsigned __int16 u16;
        unsigned __int8 u8;
        unsigned __int64 b64;
        unsigned __int32 b32;
        unsigned __int16 b16;
        unsigned __int8 b8;
        float f16;  // not really supported.
        float f32;
        double f64;
        bool pred;
        void * pvoid;
    } TYPES;

    class Symbol
    {
    public:
        char * name;
        void * pvalue;
        size_t size;
        char * typestring;
        int type;
        bool array;
        size_t index_max;
        int storage_class;
        CUDA_EMULATOR * emulator;
        ~Symbol()
        {
            if (this->array && emulator->extern_memory_buffer != ((TYPES*)this->pvalue)->pvoid)
                free(((TYPES*)this->pvalue)->pvoid);
            free(this->pvalue);
        }
    };

    class SymbolTable
    {
        public:
            std::map<char*, Symbol*, ltstr> symbols;
            SymbolTable * parent_block_symbol_table;
            SymbolTable();
            ~SymbolTable();
    };

    SymbolTable * root;
    int carry;

    class StringTable
    {
        public:
            char * Entry(char * node);
        private:
            std::map<char *, char*, ltstr> table;
    };

    StringTable * string_table;

    char * device;

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
    

    class ThreadQueue
    {
        // context information for location of thread in code.
        int pc;
        SymbolTable * root;
    };

    class Constant
    {
        public:
            int type;
            TYPES value;
            Constant(int i)
            {
                type = K_S32;
                value.s32 = i;
            }
            Constant()
            {
                memset(&this->value, 0, sizeof(value));
            }
    };

    class Thread
    {
        public:
            Thread(CUDA_EMULATOR * emulator, TREE * block, int pc, SymbolTable * root);
            ~Thread();
            bool Execute();
            bool Finished();
            void Reset();
            bool Waiting();
        private:
            TREE * block;
            int pc;
            bool finished;
            bool wait;
            SymbolTable * root;
            int carry;
            CUDA_EMULATOR * emulator;
    };

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

private:
    CUDA_EMULATOR();
    ~CUDA_EMULATOR();
    static CUDA_EMULATOR * singleton;
    TREE * FindBlock(TREE * node);
    TREE * GetInst(TREE * block, int pc);
    int Dispatch(TREE * inst);
    void SetupParams(TREE * entry);
    void SetupVariables(TREE * block, int * desired_storage_classes);
    size_t Sizeof(int type);
    int GetType(TREE * tree_type);
    int GetSize(TREE * tree_par_register);
    TREE * GetChild(TREE * node, int n);
    Symbol * FindSymbol(char * name);
    void SetupDimensionLocals();
    void SetupPredefined(dim3 tid, dim3 bid);
    void CreateSymbol(char * name, char * typestring, int type, void * value, size_t size, int storage_class);
    void SetupGotos(TREE * block);
    void Print(TREE * node, int level);
    void PrintName(TREE * node);
    void Dump(char * comment, int pc, TREE * inst);

    std::map<char*, TREE *, ltstr> entry;
    std::map<char*, TREE *, ltstr> func;
    std::map<void*, char*> fun_to_name;
    std::list<TREE *> modules;
    std::list<Symbol*> symbol_table;
    std::list<arg*> arguments;
    config conf;
    void * extern_memory_buffer;

    void SetupThreadQueue();
    void ProcessThreadQueue();
    int DoAbs(TREE * inst);
    int DoAdd(TREE * inst);
    int DoAddc(TREE * inst);
    int DoAnd(TREE * inst);
    int DoAtom(TREE * inst);
    int DoBar(TREE * inst);
    int DoBfe(TREE * inst);
    int DoBfi(TREE * inst);
    int DoBfind(TREE * inst);
    int DoBra(TREE * inst);
    int DoBrev(TREE * inst);
    int DoBrkpt(TREE * inst);
    int DoCall(TREE * inst);
    int DoClz(TREE * inst);
    int DoCnot(TREE * inst);
    int DoCopysign(TREE * inst);
    int DoCos(TREE * inst);
    int DoCvt(TREE * inst);
    int DoCvta(TREE * inst);
    int DoDiv(TREE * inst);
    int DoEx2(TREE * inst);
    int DoExit(TREE * inst);
    int DoFma(TREE * inst);
    int DoIsspacep(TREE * inst);
    int DoLd(TREE * inst);
    int DoLdu(TREE * inst);
    int DoLg2(TREE * inst);
    int DoMad(TREE * inst);
    int DoMad24(TREE * inst);
    int DoMax(TREE * inst);
    int DoMembar(TREE * inst);
    int DoMin(TREE * inst);
    int DoMov(TREE * inst);
    int DoMul(TREE * inst);
    int DoMul24(TREE * inst);
    int DoNeg(TREE * inst);
    int DoNot(TREE * inst);
    int DoOr(TREE * inst);
    int DoPmevent(TREE * inst);
    int DoPopc(TREE * inst);
    int DoPrefetch(TREE * inst);
    int DoPrefetchu(TREE * inst);
    int DoPrmt(TREE * inst);
    int DoRcp(TREE * inst);
    int DoRed(TREE * inst);
    int DoRem(TREE * inst);
    int DoRet(TREE * inst);
    int DoRsqrt(TREE * inst);
    int DoSad(TREE * inst);
    int DoSelp(TREE * inst);
    int DoSet(TREE * inst);
    int DoSetp(TREE * inst);
    int DoShl(TREE * inst);
    int DoShr(TREE * inst);
    int DoSin(TREE * inst);
    int DoSlct(TREE * inst);
    int DoSqrt(TREE * inst);
    int DoSt(TREE * inst);
    int DoSub(TREE * inst);
    int DoSubc(TREE * inst);
    int DoSuld(TREE * inst);
    int DoSuq(TREE * inst);
    int DoSured(TREE * inst);
    int DoSust(TREE * inst);
    int DoTestp(TREE * inst);
    int DoTex(TREE * inst);
    int DoTrap(TREE * inst);
    int DoTxq(TREE * inst);
    int DoVabsdiff(TREE * inst);
    int DoVadd(TREE * inst);
    int DoVmad(TREE * inst);
    int DoVmax(TREE * inst);
    int DoVmin(TREE * inst);
    int DoVote(TREE * inst);
    int DoVset(TREE * inst);
    int DoVshl(TREE * inst);
    int DoVshr(TREE * inst);
    int DoVsub(TREE * inst);
    int DoXor(TREE * inst);
    
    int FindFirstInst(TREE * block, int first);
    Constant Eval(int expected_type, TREE * const_expr);

    SymbolTable * PushSymbolTable();
    void PopSymbolTable();
    void ExecuteBlocks(bool do_thread_synch, TREE * code);
    void ExecuteSingleBlock(bool do_thread_synch, TREE * code, int bidx, int bidy, int bidz);
    bool CodeRequiresThreadSynchronization(TREE * code);
    void unimplemented(bool condition, char * text);
    void unimplemented(char * text);

    void SetupExternShared(TREE * code);
    void Extract_From_Tree(TREE * node);
    void SetupSingleVar(TREE * var, int * desired_storage_classes, bool externed);

public:
    static CUDA_EMULATOR * Singleton();
    
    // Parse
    TREE * Extract_From_Source(char * module_name, char * source);
    
    // cuda_runtime.h equivalents.
    void ** _cudaRegisterFunction(void * fun, char * name);
    cudaError_t _cudaSetupArgument(const void *arg, size_t size, size_t offset);
    cudaError_t _cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream);
    cudaError_t _cudaThreadSynchronize();
    cudaError_t _cudaLaunch(const char *entry);
    void _cudaSetDevice(char * device);
    cudaError_t _cudaGetDevice(int * device);
    cudaError_t _cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device);

    // cuda.h equivalents.
    CUresult _cuLaunchGrid(CUfunction f, int grid_width, int grid_height);
    CUresult _cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name);
    CUresult _cuModuleLoad(CUmodule *module, const char *fname);
    CUresult _cuParamSetSize(CUfunction hfunc, unsigned int numbytes);
    CUresult _cuParamSetv(CUfunction hfunc, int offset, void *ptr, unsigned int numbytes);

    // Generic setup, execution.
    void ConfigureBlock(dim3 dim);
    void ConfigureGrid(dim3 dim);
    void ConfigureSharedMemory(size_t sharedMem);
    void ConfigureStream(cudaStream_t stream);
    void Execute(TREE * entry);

    void SetTrace(int level);
    char * StringTableEntry(char * text);
};
