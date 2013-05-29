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
    SYMBOL * FindAddr(int storage_class, void * addr);
    void EnterSymbol(SYMBOL * sym);
    SYMBOL_TABLE(const SYMBOL_TABLE & original);
        void CachePvalues();
    void CheckCachedPvalues();
};

