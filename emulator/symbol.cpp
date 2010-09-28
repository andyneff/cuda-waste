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
#include "symbol.h"
#include "emulator.h"

SYMBOL::~SYMBOL()
{
    // Do not free here if this is shared memory.
    if (this->emulator->extern_memory_buffer != (TYPES::Types*)this->pvalue)
        free(this->pvalue);
}
