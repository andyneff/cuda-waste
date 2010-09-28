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
#include <cuda.h>
#include <cuda_runtime.h> // cudaError_t, CUDARTAPI, etc.
#include "symbol.h"
#include "emulator.h"
#include "symbol-table.h"


SYMBOL_TABLE::SYMBOL_TABLE()
{
}

SYMBOL_TABLE::~SYMBOL_TABLE()
{
	std::map<char*, SYMBOL*, ltstr>::iterator it = this->symbols.begin();
	for ( ; it != this->symbols.end(); ++it)
	{
		delete it->second;
	}
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
