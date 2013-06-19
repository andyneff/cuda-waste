#pragma once

#include <map>
#include "ltstr.h"

class TREE;
class ENTRY;

/*
 * MOD is used to represent an entire PTX module. The class contains the name of the module
 * (e.g., "compile_sm10"), the entire IR representing the parse tree of the module PTX,
 * a list of entry points, and list of functions, both of which are contained in the PTX.
 */
class MODULE
{
public:
	MODULE()
	{
	};

    char * name;
    TREE * tree;
    std::map<char*, ENTRY *, ltstr> entry;
    std::map<char*, ENTRY *, ltstr> func;
};
