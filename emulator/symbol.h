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

class EMULATOR;

class SYMBOL
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
    EMULATOR * emulator;
    ~SYMBOL();
};

