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
    void Extract_From_Source(char * source);
    void Extract_From_Tree(pANTLR3_BASE_TREE node);
    void Execute(void * hostfun);
    void ** RegisterFunction(void * fun, char * name);
    cudaError_t SetupArgument(const void *arg, size_t size, size_t offset);
    cudaError_t ConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream);
    cudaError_t ThreadSynchronize();

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
    void DoExit(pANTLR3_BASE_TREE inst);
    void DoMov(pANTLR3_BASE_TREE inst);
    void DoMul(pANTLR3_BASE_TREE inst);
    void DoLd(pANTLR3_BASE_TREE inst);
    void DoSetp(pANTLR3_BASE_TREE inst);
    void DoSt(pANTLR3_BASE_TREE inst);
    int FindFirstInst(pANTLR3_BASE_TREE block, int first);
};
