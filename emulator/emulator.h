#pragma once

#include <list>
#include <map>
#include "../ptxp/PtxLexer.h"
//#include "../ptxp/PtxParser.h"
#include "tree.h"
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
    } TYPES;

    class Symbol
    {
    public:
        char * name;
        void * pvalue;
        size_t size;
        char * type;
        int storage_class;
        ~Symbol()
        {
            free(pvalue);
        }
    };

    class SymbolTable
    {
    public:
        std::map<char*, Symbol*, ltstr> symbols;
        SymbolTable * parent_block_symbol_table;
    };

	SymbolTable * root;

	class StringTable
	{
		public:
			char * Entry(char * node);
		private:
			std::map<char *, char*> table;
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
                value.s64 = i;
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
			CUDA_EMULATOR * emulator;
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
    void CreateSymbol(char * name, char * type, void * value, size_t size, int storage_class);
    void SetupGotos(TREE * block);
    void Print(TREE * node, int level);
    void PrintName(TREE * node);
    void Dump(char * comment, int pc, TREE * inst);

public:
    char * StringTableEntry(char * text);
    static CUDA_EMULATOR * Singleton();
    void Extract_From_Source(char * module_name, char * source);
    void Extract_From_Tree(TREE * node);
    void Execute(void * hostfun);
    void ** RegisterFunction(void * fun, char * name);
    cudaError_t SetupArgument(const void *arg, size_t size, size_t offset);
    cudaError_t ConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream);
    cudaError_t ThreadSynchronize();
    void SetDevice(char * device);
    cudaError_t GetDevice(int * device);
    cudaError_t GetDeviceProperties(struct cudaDeviceProp *prop, int device);

private:
    std::map<char*, TREE *, ltstr> entry;
    std::map<char*, TREE *, ltstr> func;
    std::map<void*, char*> fun_to_name;
    std::list<TREE *> modules;
    std::list<Symbol*> symbol_table;
    std::list<arg*> arguments;
    config conf;

    void SetupThreadQueue();
    void ProcessThreadQueue();
    int DoAdd(TREE * inst);
    int DoBar(TREE * inst);
    int DoBra(TREE * inst);
    int DoCvt(TREE * inst);
    int DoCvta(TREE * inst);
    int DoExit(TREE * inst);
    int DoFma(TREE * inst);
    int DoMov(TREE * inst);
    int DoMad(TREE * inst);
    int DoMul(TREE * inst);
    int DoMul24(TREE * inst);
    int DoLd(TREE * inst);
    int DoLdu(TREE * inst);
    int DoSetp(TREE * inst);
    int DoSt(TREE * inst);
    int DoSub(TREE * inst);
    int DoDiv(TREE * inst);
    int FindFirstInst(TREE * block, int first);
    Constant Eval(int expected_type, TREE * const_expr);

    void PushSymbolTable();
    void PopSymbolTable();
    void ExecuteBlocks(bool do_thread_synch, TREE * code);
    void ExecuteSingleBlock(bool do_thread_synch, TREE * code, int bidx, int bidy, int bidz);
    bool CodeRequiresThreadSynchronization(TREE * code);

};
