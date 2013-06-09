#pragma once

#include <map>

class TREE;
class ENTRY;

/*
 * MOD is used to represent an entire PTX module. The class contains the name of the module
 * (e.g., "compile_sm10"), the entire IR representing the parse tree of the module PTX,
 * a list of entry points, and list of functions, both of which are contained in the PTX.
 */
class MOD
{
public:
	MOD()
	{
	};

	struct ltstr
    {
        bool operator()(const char* s1, const char* s2) const
        {
            return strcmp(s1, s2) < 0;
        }
    };
    char * name;
    TREE * tree;
    std::map<char*, ENTRY *, ltstr> entry;
    std::map<char*, ENTRY *, ltstr> func;
};
