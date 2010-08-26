#pragma once

#include <list>
#include <map>
#include "../ptxp/PtxLexer.h"
#include "../ptxp/PtxParser.h"
#include <cuda_runtime.h> // cudaError_t, CUDARTAPI, etc.

class CUDA_EMULATOR
{
	class Symbol
	{
	};
private:
	CUDA_EMULATOR();
	~CUDA_EMULATOR();
	static CUDA_EMULATOR * singleton;
	pANTLR3_BASE_TREE FindBlock(pANTLR3_BASE_TREE node);
	pANTLR3_BASE_TREE GetInst(pANTLR3_BASE_TREE block, int pc);
	bool Dispatch(pANTLR3_BASE_TREE inst);


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
	struct ltstr
	{
	  bool operator()(const char* s1, const char* s2) const
	  {
		return strcmp(s1, s2) < 0;
	  }
	};
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
	std::list<arg> arguments;
	config conf;

	void DoExit(pANTLR3_BASE_TREE inst);
	void DoMov(pANTLR3_BASE_TREE inst);
	void DoLd(pANTLR3_BASE_TREE inst);
	void DoSt(pANTLR3_BASE_TREE inst);
	int FindFirstInst(pANTLR3_BASE_TREE block, int first);
};
