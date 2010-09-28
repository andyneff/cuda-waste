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


class StringTable
{
public:
    char * Entry(char * text)
	{
		char * result = 0;
		std::map<char *, char*, ltstr>::iterator it = this->table.find(text);
		if (it == this->table.end())
		{
			std::pair<char *, char*> p;
			char * the_text = strdup(text);
			p.first = the_text;
			p.second = the_text;
			this->table.insert(p);
			result = the_text;
		}
		else
		{
			result = it->second;
		}
		return result;
	}

private:
    struct ltstr
    {
        bool operator()(const char* s1, const char* s2) const
        {
            return strcmp(s1, s2) < 0;
        }
    };

	std::map<char *, char*, ltstr> table;
};

