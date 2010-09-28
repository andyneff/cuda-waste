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

class SYMBOL;

class SYMBOL_TABLE
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
    std::map<char *, SYMBOL *, ltstr> symbols;
    SYMBOL_TABLE * parent_block_symbol_table;
    SYMBOL_TABLE();
    ~SYMBOL_TABLE();
	void Dump();
    SYMBOL * FindSymbol(char * name);
};

