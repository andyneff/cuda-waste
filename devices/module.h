#pragma once

#include <map>

class TREE;
class ENTRY;

class MOD
{
public:
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
