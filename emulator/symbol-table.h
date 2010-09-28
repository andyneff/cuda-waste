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

class Symbol;

class SymbolTable
{
private:
    struct ltstr
    {
        bool operator()(const char* s1, const char* s2) const
        {
            return strcmp(s1, s2) < 0;
        }
    };

public:
    std::map<char *, Symbol *, ltstr> symbols;
    SymbolTable * parent_block_symbol_table;
    SymbolTable();
    ~SymbolTable();
	void Dump();
    Symbol * FindSymbol(char * name);
};

