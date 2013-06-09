#pragma once

#include <map>

class MOD;
class TREE;

/*
 * ENTRY is used to represent all entry points (.entry or .func) in PTX modules.
 * The entry contains the module, the sub-tree representing the entry point in the PTX,
 * and the name of the entry point.
 */
class ENTRY
{
public:
	ENTRY()
	{
	};

    struct ltstr
    {
        bool operator()(const char* s1, const char* s2) const
        {
            return strcmp(s1, s2) < 0;
        }
    };
    MOD * module;
    TREE * tree;
    char * name;
};
