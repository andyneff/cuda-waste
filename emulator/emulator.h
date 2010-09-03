#pragma once

#include <list>
#include <map>
#include "../ptxp/PtxLexer.h"
#include "../ptxp/PtxParser.h"
#include <cuda_runtime.h> // cudaError_t, CUDARTAPI, etc.

class CUDA_EMULATOR
{
    struct ltstr
    {
        bool operator()(const char* s1, const char* s2) const
        {
            return strcmp(s1, s2) < 0;
        }
    };

    class Symbol
    {
    public:
        char * name;
        void * lvalue;
        size_t size;
        char * type;
    };

    class SymbolTable
    {
    public:
        std::map<char*, Symbol*, ltstr> symbols;
        SymbolTable * parent_block_symbol_table;
    };
    SymbolTable * root;
    char * device;

public:
    class Constant
    {
        public:
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
				float f16;	// not really supported.
				float f32;
				double f64;
				bool pred;
			} TYPES;
			int type;
			TYPES value;
            Constant(int i)
            {
                value.s64 = i;
            }
			Constant()
			{
				memset(&this->value, 0, sizeof(value));
			}
    };

private:
    CUDA_EMULATOR();
    ~CUDA_EMULATOR();
    static CUDA_EMULATOR * singleton;
    pANTLR3_BASE_TREE FindBlock(pANTLR3_BASE_TREE node);
    pANTLR3_BASE_TREE GetInst(pANTLR3_BASE_TREE block, int pc);
    int Dispatch(pANTLR3_BASE_TREE inst);
    void BindArguments(pANTLR3_BASE_TREE entry);
    void SetupLocals(pANTLR3_BASE_TREE block);
    size_t Sizeof(int type);
    int GetType(pANTLR3_BASE_TREE tree_type);
    int GetSize(pANTLR3_BASE_TREE tree_par_register);
    pANTLR3_BASE_TREE GetChild(pANTLR3_BASE_TREE node, int n);
    char * GetText(pANTLR3_BASE_TREE node);
    Symbol * FindSymbol(char * name);
    void SetupDimensionLocals();
    void SetupPredefined(dim3 tid, dim3 bid);
    void CreateSymbol(char * name, char * type, void * value, size_t size);
    void SetupGotos(pANTLR3_BASE_TREE block);
    void Print(pANTLR3_BASE_TREE node, int level);
    void Dump(char * comment, int pc, pANTLR3_BASE_TREE inst);

public:
    static CUDA_EMULATOR * Singleton();
    void Extract_From_Source(char * module_name, char * source);
    void Extract_From_Tree(pANTLR3_BASE_TREE node);
    void Execute(void * hostfun);
    void ** RegisterFunction(void * fun, char * name);
    cudaError_t SetupArgument(const void *arg, size_t size, size_t offset);
    cudaError_t ConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream);
    cudaError_t ThreadSynchronize();
    void SetDevice(char * device);
    cudaError_t GetDevice(int * device);
    cudaError_t GetDeviceProperties(struct cudaDeviceProp *prop, int device);

private:
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
    std::map<char*, pANTLR3_BASE_TREE, ltstr> entry;
    std::map<char*, pANTLR3_BASE_TREE, ltstr> func;
    std::map<void*, char*> fun_to_name;
    std::list<pANTLR3_BASE_TREE> modules;
    std::list<Symbol*> symbol_table;
    std::list<arg*> arguments;
    config conf;

    void DoAdd(pANTLR3_BASE_TREE inst);
    int DoBra(pANTLR3_BASE_TREE inst);
    void DoCvt(pANTLR3_BASE_TREE inst);
    void DoExit(pANTLR3_BASE_TREE inst);
    void DoMov(pANTLR3_BASE_TREE inst);
    void DoMul(pANTLR3_BASE_TREE inst);
    void DoLd(pANTLR3_BASE_TREE inst);
    void DoSetp(pANTLR3_BASE_TREE inst);
    void DoSt(pANTLR3_BASE_TREE inst);
    void DoDiv(pANTLR3_BASE_TREE inst);
    int FindFirstInst(pANTLR3_BASE_TREE block, int first);
    Constant Eval(int expected_type, pANTLR3_BASE_TREE const_expr);
};
