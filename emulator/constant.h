#pragma once
#include "types.h"

class CONSTANT
{
    public:
        int type;
        TYPES::Types value;
        CONSTANT(int i)
        {
            type = K_S32;
            value.s32 = i;
        }
        CONSTANT()
        {
            memset(&this->value, 0, sizeof(value));
        }
};

