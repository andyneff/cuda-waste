#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include "../ptxp/PtxLexer.h"
//#include "../ptxp/PtxParser.h"
#include "tree.h"
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
#include <cuda.h>
#include <cuda_runtime.h> // cudaError_t, CUDARTAPI, etc.
#include "symbol.h"
#include "emulator.h"
#include "symbol-table.h"
#define new new(_CLIENT_BLOCK,__FILE__, __LINE__)


SYMBOL_TABLE::SYMBOL_TABLE()
{
}

SYMBOL_TABLE::SYMBOL_TABLE(const SYMBOL_TABLE & original)
{
    this->symbols = original.symbols;
    this->parent_block_symbol_table = original.parent_block_symbol_table;
}

SYMBOL_TABLE::~SYMBOL_TABLE()
{
    std::map<char*, SYMBOL*, ltstr>::iterator it = this->symbols.begin();
    for ( ; it != this->symbols.end(); ++it)
    {
        delete it->second;
	}
	this->symbols.clear();
}

void SYMBOL_TABLE::Dump()
{
    int level = 0;
    for (SYMBOL_TABLE * st = this; st != 0; st = st->parent_block_symbol_table, level++)
    {
        std::cout << "---- Level " << level << " ----\n";
        std::map<char*, SYMBOL*, ltstr>::iterator it;
        for (it = st->symbols.begin(); it != st->symbols.end(); ++it)
        {
            SYMBOL * s = (*it).second;
            std::cout << "name: " << s->name << " ";
            std::cout << "size: " << s->size << " ";
            std::cout << "stor: " << s->storage_class << " ";
            std::cout << "type: " << s->typestring << " ";
            if (strcmp(s->typestring, "label") == 0)
                std::cout << "val:  " << (int)s->pvalue << "\n";
            else if (strcmp(s->typestring, "dim3") == 0)
                std::cout << "val:  " << ((dim3*)s->pvalue)->x
                        << " " << ((dim3*)s->pvalue)->y
                        << " " << ((dim3*)s->pvalue)->z
                        << "\n";
            else if (strcmp(s->typestring, ".pred") == 0)
                std::cout << "val:  " << ((TYPES::Types*)s->pvalue)->pred << "\n";
            else if (strcmp(s->typestring, ".u8") == 0)
                std::cout << "val:  " << ((TYPES::Types*)s->pvalue)->u8 << "\n";
            else if (strcmp(s->typestring, ".u16") == 0)
                std::cout << "val:  " << ((TYPES::Types*)s->pvalue)->u16 << "\n";
            else if (strcmp(s->typestring, ".u32") == 0)
            {
                std::cout << "val:  " << ((TYPES::Types*)s->pvalue)->u32 << " ";
                std::cout << std::hex << ((TYPES::Types*)s->pvalue)->u32;
                std::cout << std::dec << "\n";
            }
            else if (strcmp(s->typestring, ".u64") == 0)
                std::cout << "val:  " << ((TYPES::Types*)s->pvalue)->u64 << "\n";
            else if (strcmp(s->typestring, ".s8") == 0)
                std::cout << "val:  " << ((TYPES::Types*)s->pvalue)->s8 << "\n";
            else if (strcmp(s->typestring, ".s16") == 0)
                std::cout << "val:  " << ((TYPES::Types*)s->pvalue)->s16 << "\n";
            else if (strcmp(s->typestring, ".s32") == 0)
                std::cout << "val:  " << ((TYPES::Types*)s->pvalue)->s32 << "\n";
            else if (strcmp(s->typestring, ".s64") == 0)
                std::cout << "val:  " << ((TYPES::Types*)s->pvalue)->s64 << "\n";
            else if (strcmp(s->typestring, ".b8") == 0)
                std::cout << "val:  " << ((TYPES::Types*)s->pvalue)->u8 << "\n";
            else if (strcmp(s->typestring, ".b16") == 0)
                std::cout << "val:  " << ((TYPES::Types*)s->pvalue)->u16 << "\n";
            else if (strcmp(s->typestring, ".b32") == 0)
            {
                std::cout << "val:  " << ((TYPES::Types*)s->pvalue)->u32 << " ";
                std::cout << std::hex << ((TYPES::Types*)s->pvalue)->u32;
                std::cout << std::dec << "\n";
            }
            else if (strcmp(s->typestring, ".b64") == 0)
                std::cout << "val:  " << ((TYPES::Types*)s->pvalue)->u64 << "\n";
            else if (strcmp(s->typestring, ".f32") == 0)
                std::cout << "val:  " << ((TYPES::Types*)s->pvalue)->f32 << "\n";
            else if (strcmp(s->typestring, ".f64") == 0)
                std::cout << "val:  " << ((TYPES::Types*)s->pvalue)->f64 << "\n";
            else assert(false);
        }
    }

}


SYMBOL * SYMBOL_TABLE::FindSymbol(char * name)
{
    SYMBOL_TABLE * st = this;
    while (st)
    {
        std::map<char*, SYMBOL*, ltstr>::iterator it = st->symbols.find(name);
        if (it != st->symbols.end())
        {
            return it->second;
        }
        st = st->parent_block_symbol_table;
    }
    return 0;
}

void SYMBOL_TABLE::EnterSymbol(SYMBOL * s)
{
	assert(FindSymbol(s->name) == 0);
    // Add the entry into the symbol table.
    std::pair<char*, SYMBOL*> sym;
    sym.first = s->name;
    sym.second = s;
    this->symbols.insert(sym);
}


void SYMBOL_TABLE::CachePvalues()
{
	// Cache pvalues in the symbol tables.
	std::map<char*, SYMBOL*, ltstr>::iterator it = this->symbols.begin();
	for ( ; it != this->symbols.end(); ++it)
	{
		SYMBOL * s = it->second;
		s->cache = s->pvalue;
	}
	if (this->parent_block_symbol_table)
		this->parent_block_symbol_table->CachePvalues();
}

void SYMBOL_TABLE::CheckCachedPvalues()
{
	// Cache pvalues in the symbol tables.
	std::map<char*, SYMBOL*, ltstr>::iterator it = this->symbols.begin();
	for ( ; it != this->symbols.end(); ++it)
	{
		SYMBOL * s = it->second;
		if (s->cache != s->pvalue)
			throw new EMULATOR::Unimplemented("Cache overwrite!");
	}
	if (this->parent_block_symbol_table)
		this->parent_block_symbol_table->CheckCachedPvalues();
}

