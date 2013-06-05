#pragma once

#include <map>

class MOD;
class TREE;

class ENTRY
{
public:
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
