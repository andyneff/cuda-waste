#pragma once
#include "types.h"

class Constant
{
    public:
        int type;
        TYPES::Types value;
        Constant(int i)
        {
            type = K_S32;
            value.s32 = i;
        }
        Constant()
        {
            memset(&this->value, 0, sizeof(value));
        }
};

