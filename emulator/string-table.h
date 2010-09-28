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


class STRING_TABLE
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

