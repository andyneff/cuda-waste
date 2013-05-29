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
#include "emulated-device.h"
#include <assert.h>
#define new new(_CLIENT_BLOCK,__FILE__, __LINE__)

SYMBOL::~SYMBOL()
{
    // Do not free here if this is shared memory.
    if (this->emulator->extern_memory_buffer == (TYPES::Types*)this->pvalue)
        ;
    else if (this->size == 0 && this->storage_class == 0)
        ;
    else
        free(this->pvalue);
    this->pvalue = 0;
}
