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

#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#include <assert.h>
#include <iostream>
#include "emulated-device.h"
#include "thread.h"
#include "symbol.h"
#include "symbol-table.h"
#include "constant.h"
#include "../wrapper/lock-mgr.h"
#include "../wrapper/cuda-wrapper.h"
#include "texref.h"
#include "texture.h"
#include "array.h"
#include "texarr.h"
#include "../libc/k_stdio.h"
#include <bitset>

#define new new(_CLIENT_BLOCK,__FILE__, __LINE__)

int THREAD::DoAbs(TREE * inst)
{
    int start = 0;
    if (inst->GetChild(start)->GetType() == TREE_PRED)
        start++;
    assert(inst->GetChild( start)->GetType() == KI_ABS);
    start++;
    TREE * ttype = 0;
    TREE * odst = 0;
    TREE * osrc1 = 0;
    for (;; ++start)
    {
        TREE * t = inst->GetChild(start);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == TREE_TYPE)
            ttype = t;
        else if (gt == TREE_OPR)
        {
            if (odst == 0)
            {
                odst = t;
            } else if (osrc1 == 0)
            {
                osrc1 = t;
            }
            else
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ABS instruction");
        }
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ABS instruction");
    }
    assert(ttype != 0);
    assert(odst != 0);
    assert(osrc1 != 0);
    bool ftz = false;
    for (int i = 0; ; ++i)
    {
        TREE * t = ttype->GetChild(i);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == K_S16 || gt == K_S32 || gt == K_S64)
            ttype = t;
        else if (gt == K_F32 || gt == K_F64)
            ttype = t;
        else if (gt == K_FTZ)
            ftz = true;
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ABS instruction");
    }
    assert(ttype != 0);
    this->device->unimplemented(ftz, "ABS.ftz not implemented.");

    int type = ttype->GetType();
    TREE * dst = odst->GetChild(0);
    TREE * src1 = osrc1->GetChild(0);

    SYMBOL * sdst = 0;
    if (dst->GetType() == T_WORD)
    {
        sdst = this->symbol_table->FindSymbol(dst->GetText());
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ABS instruction");

    TYPES::Types value1;
    char * dummy;
    TYPES::Types * d = (TYPES::Types*)sdst->pvalue;
    TYPES::Types * s1 = &value1;

    if (src1->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src1->GetChild(0));
        switch (type)
        {
            case K_S16:
                s1->s16 = c.value.s16;
                break;
            case K_S32:
                s1->s32 = c.value.s32;
                break;
            case K_S64:
                s1->s64 = c.value.s64;
                break;
            case K_F32:
                s1->f32 = c.value.f32;
                break;
            case K_F64:
                s1->f64 = c.value.f64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ABS instruction");
        }
    } else if (src1->GetType() == T_WORD)
    {
        SYMBOL * ssrc1 = this->symbol_table->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        assert(ssrc1->size == this->device->Sizeof(type));
        TYPES::Types * psrc1_value = (TYPES::Types*)ssrc1->pvalue;
        switch (type)
        {
            case K_S16:
                s1->s16 = psrc1_value->s16;
                break;
            case K_S32:
                s1->s32 = psrc1_value->s32;
                break;
            case K_S64:
                s1->s64 = psrc1_value->s64;
                break;
            case K_F32:
                s1->f32 = psrc1_value->f32;
                break;
            case K_F64:
                s1->f64 = psrc1_value->f64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ABS instruction");
        }
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ABS instruction");

    switch (type)
    {
        case K_S16:
            d->s16 = abs(s1->s16);
            break;
        case K_S32:
            d->s32 = abs(s1->s32);
            break;
        case K_S64:
            d->s64 = abs(s1->s64);
            break;
        case K_F32:
            d->f32 = abs(s1->f32);
            break;
        case K_F64:
            d->f64 = abs(s1->f64);
            break;
        default:
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ABS instruction");
    }
    return 0;
}

int THREAD::DoAdd(TREE * inst)
{
    int start = 0;
    if (inst->GetChild(start)->GetType() == TREE_PRED)
        start++;
    assert(inst->GetChild(start)->GetType() == KI_ADD);
    start++;
    TREE * ttype = 0;
    TREE * odst = 0;
    TREE * osrc1 = 0;
    TREE * osrc2 = 0;
    for (;; ++start)
    {
        TREE * t = inst->GetChild(start);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == TREE_TYPE)
            ttype = t;
        else if (gt == TREE_OPR)
        {
            if (odst == 0)
            {
                odst = t;
            } else if (osrc1 == 0)
            {
                osrc1 = t;
            } else if (osrc2 == 0)
            {
                osrc2 = t;
            }
            else
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ADD instruction");
        }
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ADD instruction");
    }
    assert(ttype != 0);
    assert(odst != 0);
    assert(osrc1 != 0);
    assert(osrc2 != 0);
    bool sat = false;
    bool cc = false;
    int rnd = 0;
    bool ftz = false;
    for (int i = 0; ; ++i)
    {
        TREE * t = ttype->GetChild(i);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == K_SAT)
            sat = true;
        else if (gt == K_U16 || gt == K_U32 || gt == K_U64
                 || gt == K_S16 || gt == K_S32 || gt == K_S64)
            ttype = t;
        else if (gt == K_F32 || gt == K_F64)
            ttype = t;
        else if (gt == K_CC)
            cc = true;
        else if (gt == K_RN || gt == K_RZ || gt == K_RM || gt == K_RP)
            rnd = gt;
        else if (gt == K_FTZ)
            ftz = true;
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ADD instruction");
    }
    assert(ttype != 0);
    this->device->unimplemented(ftz, "ADD.ftz not implemented.");
    int type = ttype->GetType();
    TREE * dst = odst->GetChild(0);
    TREE * src1 = osrc1->GetChild(0);
    TREE * src2 = osrc2->GetChild(0);

    SYMBOL * sdst = 0;
    if (dst->GetType() == T_WORD)
    {
        sdst = this->symbol_table->FindSymbol(dst->GetText());
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ADD instruction");

    TYPES::Types value1;
    TYPES::Types value2;
    TYPES::Types value3;
    char * dummy;
    TYPES::Types * d = (TYPES::Types*)sdst->pvalue;
    TYPES::Types * s1 = &value1;
    TYPES::Types * s2 = &value2;
    // used for carry out calculation.
    TYPES::Types * temp = &value3;

    if (src1->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src1->GetChild(0));
        switch (type)
        {
            case K_U16:
                s1->u16 = c.value.u16;
                break;
            case K_S16:
                s1->s16 = c.value.s16;
                break;
            case K_U32:
                s1->u32 = c.value.u32;
                break;
            case K_S32:
                s1->s32 = c.value.s32;
                break;
            case K_U64:
                s1->u64 = c.value.u64;
                break;
            case K_S64:
                s1->s64 = c.value.s64;
                break;
            case K_F32:
                s1->f32 = c.value.f32;
                break;
            case K_F64:
                s1->f64 = c.value.f64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ADD instruction");
        }
    } else if (src1->GetType() == T_WORD)
    {
        SYMBOL * ssrc1 = this->symbol_table->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        assert(ssrc1->size == this->device->Sizeof(type));
        TYPES::Types * psrc1_value = (TYPES::Types*)ssrc1->pvalue;
        switch (type)
        {
            case K_U16:
                s1->u16 = psrc1_value->u16;
                break;
            case K_S16:
                s1->s16 = psrc1_value->s16;
                break;
            case K_U32:
                s1->u32 = psrc1_value->u32;
                break;
            case K_S32:
                s1->s32 = psrc1_value->s32;
                break;
            case K_U64:
                s1->u64 = psrc1_value->u64;
                break;
            case K_S64:
                s1->s64 = psrc1_value->s64;
                break;
            case K_F32:
                s1->f32 = psrc1_value->f32;
                break;
            case K_F64:
                s1->f64 = psrc1_value->f64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ADD instruction");
        }
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ADD instruction");

    if (src2->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src2->GetChild(0));
        switch (type)
        {
            case K_U16:
                s2->u16 = c.value.u16;
                break;
            case K_S16:
                s2->s16 = c.value.s16;
                break;
            case K_U32:
                s2->u32 = c.value.u32;
                break;
            case K_S32:
                s2->s32 = c.value.s32;
                break;
            case K_U64:
                s2->u64 = c.value.u64;
                break;
            case K_S64:
                s2->s64 = c.value.s64;
                break;
            case K_F32:
                s1->f32 = c.value.f32;
                break;
            case K_F64:
                s1->f64 = c.value.f64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ADD instruction");
        }
    } else if (src2->GetType() == T_WORD)
    {
        SYMBOL * ssrc2 = this->symbol_table->FindSymbol(src2->GetText());
        assert(ssrc2 != 0);
        assert(ssrc2->size == this->device->Sizeof(type));
        TYPES::Types * psrc2_value = (TYPES::Types*)ssrc2->pvalue;
        switch (type)
        {
            case K_U16:
                s2->u16 = psrc2_value->u16;
                break;
            case K_S16:
                s2->s16 = psrc2_value->s16;
                break;
            case K_U32:
                s2->u32 = psrc2_value->u32;
                break;
            case K_S32:
                s2->s32 = psrc2_value->s32;
                break;
            case K_U64:
                s2->u64 = psrc2_value->u64;
                break;
            case K_S64:
                s2->s64 = psrc2_value->s64;
                break;
            case K_F32:
                s2->f32 = psrc2_value->f32;
                break;
            case K_F64:
                s2->f64 = psrc2_value->f64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ADD instruction");
        }
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ADD instruction");

    switch (type)
    {
        case K_U16:
            temp->u32 = s1->u16 + (unsigned __int32)s2->u16;
            if (temp->u32 >> 16)
                this->carry = 1;
            else
                this->carry = 0;
            d->u16 = temp->u32;
            break;
        case K_S16:
            temp->s32 = s1->s16 + (signed __int32)s2->s16;
            if ((temp->s32 >> 16) && 0xffff)
                this->carry = 1;
            else
                this->carry = 0;
            d->s16 = temp->s32;
            break;
        case K_U32:
            temp->u64 = s1->u32 + (unsigned __int64)s2->u32;
            if ((temp->u64 >> 32) && 0xffffffff)
                this->carry = 1;
            else
                this->carry = 0;
            d->u32 = temp->u64;
            break;
        case K_S32:
            temp->s64 = s1->s32 + (signed __int64)s2->s32;
            if ( temp->s64 > (__int64)0x7fffffff
                || temp->s64 < (__int64)0xffffffff80000000)
            {
                if (sat && temp->s64 > (__int64)0x7fffffff)
                {
                    temp->s64 = (__int64)0x7fffffff;
                }
                else if (sat && temp->s64 < (__int64)0xffffffff80000000)
                {
                    temp->s64 = (__int64)0xffffffff80000000;
                }
                this->carry = 1;
            }
            else
            {
                this->carry = 0;
            }
            d->s32 = temp->s64;
            break;
        case K_S64:
            d->s64 = s1->s64 + s2->s64;
            break;
        case K_U64:
            d->u64 = s1->u64 + s2->u64;
            break;
        case K_F32:
            temp->f64 = s1->f32 + (double)s2->f32;
            // Round.
            switch (rnd)
            {
                case K_RN:
                case K_RZ:
                case K_RM:
                    d->f32 = temp->f64;
                    break;
                case K_RP:
                    // test low bits of mantissa, round up.
                    if (temp->b64 & 0x00000000ffffffff)
                        temp->b64 |= 0x0000000100000000;
                    d->f32 = temp->f64;
                    break;
                default:
                    d->f32 = temp->f64;
                    break;
            }
            if (sat)
            {
                if (d->f32 > 1.0)
                    d->f32 = 1.0;
                else if (d->f32 < 0.0)
                    d->f32 = 0.0;
            }
            break;
        case K_F64:
            d->f64 = s1->f64 + s2->f64;
            if (sat)
            {
                if (d->f64 > 1.0)
                    d->f64 = 1.0;
                else if (d->f64 < 0.0)
                    d->f64 = 0.0;
            }
            break;
        default:
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ADD instruction");
    }
    return 0;
}

int THREAD::DoAddc(TREE * inst)
{
    int start = 0;
    if (inst->GetChild(start)->GetType() == TREE_PRED)
        start++;
    assert(inst->GetChild(start)->GetType() == KI_ADDC);
    start++;
    TREE * ttype = 0;
    TREE * odst = 0;
    TREE * osrc1 = 0;
    TREE * osrc2 = 0;
    for (;; ++start)
    {
        TREE * t = inst->GetChild(start);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == TREE_TYPE)
            ttype = t;
        else if (gt == TREE_OPR)
        {
            if (odst == 0)
            {
                odst = t;
            } else if (osrc1 == 0)
            {
                osrc1 = t;
            } else if (osrc2 == 0)
            {
                osrc2 = t;
            }
            else
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ADDC instruction");
        }
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ADDC instruction");
    }
    assert(ttype != 0);
    assert(odst != 0);
    assert(osrc1 != 0);
    assert(osrc2 != 0);
    bool cc = false;
    int type = 0;
    for (int i = 0; ; ++i)
    {
        TREE * t = ttype->GetChild(i);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == K_U32 || gt == K_S32 || gt == K_U64 || K_U16 )
            type = gt;
        else if (gt == K_CC)
            cc = true;
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ADDC instruction");
    }
    assert(type != 0);
    TREE * dst = odst->GetChild(0);
    TREE * src1 = osrc1->GetChild(0);
    TREE * src2 = osrc2->GetChild(0);

    SYMBOL * sdst = 0;
    if (dst->GetType() == T_WORD)
    {
        sdst = this->symbol_table->FindSymbol(dst->GetText());
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ADDC instruction");

    TYPES::Types value1;
    TYPES::Types value2;
    TYPES::Types value3;
    char * dummy;
    TYPES::Types * d = (TYPES::Types*)sdst->pvalue;
    TYPES::Types * s1 = &value1;
    TYPES::Types * s2 = &value2;
    // used for carry out calculation.
    TYPES::Types * temp = &value3;

    if (src1->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src1->GetChild(0));
        switch (type)
        {
            case K_U32:
                s1->u32 = c.value.u32;
                break;
            case K_S32:
                s1->s32 = c.value.s32;
                break;
            case K_U16:
                s1->u16 = c.value.u16;
                break;
            case K_U64:
                s1->s32 = c.value.u64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ADDC instruction");
        }
    } else if (src1->GetType() == T_WORD)
    {
        SYMBOL * ssrc1 = this->symbol_table->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        assert(ssrc1->size == this->device->Sizeof(type));
        TYPES::Types * psrc1_value = (TYPES::Types*)ssrc1->pvalue;
        switch (type)
        {
            case K_U32:
                s1->u32 = psrc1_value->u32;
                break;
            case K_S32:
                s1->s32 = psrc1_value->s32;
                break;
            case K_U16:
                s1->u16 = psrc1_value->u16;
                break;
            case K_U64:
                s1->u64 = psrc1_value->u64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ADDC instruction");
        }
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ADDC instruction");

    if (src2->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src2->GetChild(0));
        switch (type)
        {
            case K_U32:
                s2->u32 = c.value.u32;
                break;
            case K_S32:
                s2->s32 = c.value.s32;
                break;
            case K_U16:
                s1->u16 = c.value.u16;
                break;
            case K_U64:
                s1->u64 = c.value.u64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ADDC instruction");
        }
    } else if (src2->GetType() == T_WORD)
    {
        SYMBOL * ssrc2 = this->symbol_table->FindSymbol(src2->GetText());
        assert(ssrc2 != 0);
        assert(ssrc2->size == this->device->Sizeof(type));
        TYPES::Types * psrc2_value = (TYPES::Types*)ssrc2->pvalue;
        switch (type)
        {
            case K_U32:
                s2->u32 = psrc2_value->u32;
                break;
            case K_S32:
                s2->s32 = psrc2_value->s32;
                break;
            case K_U16:
                s2->u16 = psrc2_value->u16;
                break;
            case K_U64:
                s2->u64 = psrc2_value->u64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ADDC instruction");
        }
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ADDC instruction");

    switch (type)
    {
        case K_U32:
            temp->u64 = s1->u32 + (unsigned __int64)s2->u32 + this->carry;
            if ((temp->u64 >> 32) && 0xffffffff)
                this->carry = 1;
            else
                this->carry = 0;
            d->u32 = temp->u32;
            break;
        case K_U16:
            temp->u32 = s1->u16 + (unsigned __int64)s2->u16 + this->carry;
            if ((temp->u32 >> 16) && 0xffff)
                this->carry = 1;
            else
                this->carry = 0;
            d->u16 = temp->u32;
            break;
        case K_U64:
            temp->u64 = s1->u64 + (unsigned __int64)s2->u64 + this->carry;
            d->u64 = temp->u64;
            break;
        case K_S32:
            temp->s64 = s1->s32 + (signed __int64)s2->s32 + this->carry;
            if ((temp->s64 >> 32) && 0xffffffff)
                this->carry = 1;
            else
                this->carry = 0;
            d->s32 = temp->s32;
            break;
        default:
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ADDC instruction");
    }
    return 0;
}

int THREAD::DoAnd(TREE * inst)
{
    int start = 0;
    if (inst->GetChild(start)->GetType() == TREE_PRED)
        start++;
    assert(inst->GetChild(start)->GetType() == KI_AND);
    start++;
    TREE * ttype = 0;
    TREE * odst = 0;
    TREE * osrc1 = 0;
    TREE * osrc2 = 0;
    for (;; ++start)
    {
        TREE * t = inst->GetChild(start);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == TREE_TYPE)
            ttype = t;
        else if (gt == TREE_OPR)
        {
            if (odst == 0)
            {
                odst = t;
            } else if (osrc1 == 0)
            {
                osrc1 = t;
            } else if (osrc2 == 0)
            {
                osrc2 = t;
            }
            else
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected AND instruction");
        }
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected AND instruction");
    }
    assert(ttype != 0);
    assert(odst != 0);
    assert(osrc1 != 0);
    assert(osrc2 != 0);
    int type = 0;
    for (int i = 0; ; ++i)
    {
        TREE * t = ttype->GetChild(i);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == K_PRED || gt == K_B16 || gt == K_B32 || gt == K_B64)
            type = gt;
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected AND instruction");
    }
    assert(type != 0);
    TREE * dst = odst->GetChild(0);
    TREE * src1 = osrc1->GetChild(0);
    TREE * src2 = osrc2->GetChild(0);

    SYMBOL * sdst = 0;
    if (dst->GetType() == T_WORD)
    {
        sdst = this->symbol_table->FindSymbol(dst->GetText());
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected AND instruction");

    TYPES::Types value1;
    TYPES::Types value2;
    char * dummy;
    TYPES::Types * d = (TYPES::Types*)sdst->pvalue;
    TYPES::Types * s1 = &value1;
    TYPES::Types * s2 = &value2;

    if (src1->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src1->GetChild(0));
        switch (type)
        {
            case K_B16:
                s1->b16 = c.value.b16;
                break;
            case K_B32:
                s1->b32 = c.value.b32;
                break;
            case K_B64:
                s1->b64 = c.value.b64;
                break;
            case K_PRED:
                s1->pred = c.value.pred;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected AND instruction");
        }
    } else if (src1->GetType() == T_WORD)
    {
        SYMBOL * ssrc1 = this->symbol_table->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        assert(ssrc1->size == this->device->Sizeof(type));
        TYPES::Types * psrc1_value = (TYPES::Types*)ssrc1->pvalue;
        switch (type)
        {
            case K_B16:
                s1->b16 = psrc1_value->b16;
                break;
            case K_B32:
                s1->b32 = psrc1_value->b32;
                break;
            case K_B64:
                s1->b64 = psrc1_value->b64;
                break;
            case K_PRED:
                s1->pred = psrc1_value->pred;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected AND instruction");
        }
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected AND instruction");

    if (src2->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src2->GetChild(0));
        switch (type)
        {
            case K_B16:
                s2->b16 = c.value.b16;
                break;
            case K_B32:
                s2->b32 = c.value.b32;
                break;
            case K_B64:
                s2->b64 = c.value.b64;
                break;
            case K_PRED:
                s2->pred = c.value.pred;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected AND instruction");
        }
    } else if (src2->GetType() == T_WORD)
    {
        SYMBOL * ssrc2 = this->symbol_table->FindSymbol(src2->GetText());
        assert(ssrc2 != 0);
        assert(ssrc2->size == this->device->Sizeof(type));
        TYPES::Types * psrc2_value = (TYPES::Types*)ssrc2->pvalue;
        switch (type)
        {
            case K_B16:
                s2->b16 = psrc2_value->b16;
                break;
            case K_B32:
                s2->b32 = psrc2_value->b32;
                break;
            case K_B64:
                s2->b64 = psrc2_value->b64;
                break;
            case K_PRED:
                s2->pred = psrc2_value->pred;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected AND instruction");
        }
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected AND instruction");

    switch (type)
    {
        case K_B16:
            d->b16 = s1->b16 & s2->b16;
            break;
        case K_B32:
            d->b32 = s1->b32 & s2->b32;
            break;
        case K_S64:
            d->b64 = s1->b64 & s2->b64;
            break;
        case K_PRED:
            d->pred = s1->pred & s2->pred;
            break;
        default:
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected AND instruction");
    }
    return 0;
}

int THREAD::DoAtom(TREE * inst)
{
    LOCK_MANAGER<CRIT_SECTION>  lockMgr(this->device->sm_CritSec, TRUE);
    int start = 0;
    if (inst->GetChild(start)->GetType() == TREE_PRED)
        start++;
    assert(inst->GetChild(start)->GetType() == KI_ATOM);
    start++;
    TREE * ttype = 0;
    TREE * odst = 0;
    TREE * osrc1 = 0;
    TREE * osrc2 = 0;
    TREE * osrc3 = 0;
    for (;; ++start)
    {
        TREE * t = inst->GetChild(start);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == TREE_TYPE)
            ttype = t;
        else if (gt == TREE_OPR)
        {
            if (odst == 0)
            {
                odst = t;
            } else if (osrc1 == 0)
            {
                osrc1 = t;
            } else if (osrc2 == 0)
            {
                osrc2 = t;
            } else if (osrc3 == 0)
            {
                osrc3 = t;
            }
            else
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ATOM instruction");
        }
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ATOM instruction");
    }
    assert(ttype != 0);
    assert(odst != 0);
    assert(osrc1 != 0);
    assert(osrc2 != 0);
    int op = 0;
    int type = 0;
    int storage_class = 0;
    for (int i = 0; ; ++i)
    {
        TREE * t = ttype->GetChild(i);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == K_SHARED || gt == K_GLOBAL)
            storage_class = gt;
        else if (gt == K_B32 || gt == K_B64
            || gt == K_U32 || gt == K_U64
            || gt == K_S32)
            type = gt;
        else if (gt == K_F32)
            type = gt;
        else if (gt == K_AND || gt == K_OR || gt == K_XOR ||
                 gt == K_CAS || gt == K_EXCH ||
                 gt == K_ADD ||
                 gt == K_INC || gt == K_DEC ||
                 gt == K_MIN || gt == K_MAX)
            op = gt;
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ATOM instruction");
    }
    assert(op != 0);
    assert(type != 0);

    TREE * dst = odst->GetChild(0);
    // Note src1 is not really only a source operand. It is a
    // destination operand, too.
    TREE * src1 = osrc1->GetChild(0);
    TREE * src2 = osrc2->GetChild(0);
    TREE * src3 = osrc3 ? osrc3->GetChild(0) : 0;

    assert(src1->GetType() == T_WORD);
    SYMBOL * ssrc1 = this->symbol_table->FindSymbol(src1->GetText());
    assert(ssrc1 != 0);
    TREE * plus = osrc1->GetChild(1);
    CONSTANT value(0);
    if (plus != 0)
    {
        TREE * const_expr_tree = osrc1->GetChild(2);
        assert(const_expr_tree != 0);
        assert(const_expr_tree->GetType() == TREE_CONSTANT_EXPR);
        TREE * const_expr = const_expr_tree->GetChild(0);
        assert(const_expr != 0);
        value = this->device->Eval(K_S32, const_expr);
    }

    // Different semantics for different storage classes.
    TYPES::Types * s = 0;
    unsigned char * addr = 0;
    switch (ssrc1->storage_class)
    {
        case K_GLOBAL:
        case K_LOCAL:
        case K_PARAM:
        case K_SHARED:
        case K_CONST:
        {
            addr = (unsigned char*)ssrc1->pvalue;
        }
        break;
        case K_REG:
        {
            addr = *(unsigned char**)ssrc1->pvalue;
        }
        break;
        default:
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ATOM instruction");
    }
    switch (value.type)
    {
        case K_U8:
            addr = (addr + value.value.s32);
            break;
        case K_U16:
            addr = (addr + value.value.s32);
            break;
        case K_U32:
            addr = (addr + value.value.s32);
            break;
        case K_U64:
            addr = (addr + value.value.s32);
            break;
        case K_S8:
            addr = (addr + value.value.s32);
            break;
        case K_S16:
            addr = (addr + value.value.s32);
            break;
        case K_S32:
            addr = (addr + value.value.s32);
            break;
        case K_S64:
            addr = (addr + value.value.s32);
            break;
        case K_B8:
            addr = (addr + value.value.s32);
            break;
        case K_B16:
            addr = (addr + value.value.s32);
            break;
        case K_B32:
            addr = (addr + value.value.s32);
            break;
        case K_B64:
            addr = (addr + value.value.s32);
            break;
        case K_F32:
            addr = (addr + value.value.s32);
            break;
        case K_F64:
            addr = (addr + value.value.s32);
            break;
        default:
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ATOM instruction");
    }

    SYMBOL * sdst = 0;
    assert(dst->GetType() == T_WORD);
    sdst = this->symbol_table->FindSymbol(dst->GetText());
    assert(sdst != 0);
    TYPES::Types * d = (TYPES::Types*)sdst->pvalue;
    TYPES::Types * s1 = (TYPES::Types*)addr;
    TYPES::Types value2;
    TYPES::Types * s2 = &value2;

    if (src2->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src2->GetChild(0));
        switch (type)
        {
            case K_B32:
                s2->b32 = c.value.b32;
                break;
            case K_B64:
                s2->b64 = c.value.b64;
                break;
            case K_U32:
                s2->u32 = c.value.u32;
                break;
            case K_U64:
                s2->u64 = c.value.u64;
                break;
            case K_S32:
                s2->s32 = c.value.s32;
                break;
            case K_F32:
                s2->f32 = c.value.f32;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ATOM instruction");
        }
    } else if (src2->GetType() == T_WORD)
    {
        SYMBOL * ssrc2 = this->symbol_table->FindSymbol(src2->GetText());
        assert(ssrc2 != 0);
        assert(ssrc2->size == this->device->Sizeof(type));
        TYPES::Types * psrc2_value = (TYPES::Types*)ssrc2->pvalue;
        switch (type)
        {
            case K_B32:
                s2->b32 = psrc2_value->b32;
                break;
            case K_B64:
                s2->b64 = psrc2_value->b64;
                break;
            case K_U32:
                s2->u32 = psrc2_value->u32;
                break;
            case K_U64:
                s2->u64 = psrc2_value->u64;
                break;
            case K_S32:
                s2->s32 = psrc2_value->s32;
                break;
            case K_F32:
                s2->f32 = psrc2_value->f32;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ATOM instruction");
        }
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ATOM instruction");

    switch (type)
    {
        case K_B32:
            switch (op)
            {
                case K_AND:
                    d->b32 = s1->b32;
                    s1->b32 = s1->b32 & s2->b32;
                    break;
                case K_OR:
                    d->b32 = s1->b32;
                    s1->b32 = s1->b32 | s2->b32;
                    break;
                case K_XOR:
                    d->b32 = s1->b32;
                    s1->b32 = s1->b32 ^ s2->b32;
                    break;
                case K_ADD:
                    d->b32 = s1->b32;
                    s1->b32 = s1->b32 + s2->b32;
                    break;
                default:
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ATOM instruction");
            }
            break;
        case K_B64:
            switch (op)
            {
                case K_AND:
                    d->b64 = s1->b64;
                    s1->b64 = s1->b64 & s2->b64;
                    break;
                case K_OR:
                    d->b64 = s1->b64;
                    s1->b64 = s1->b64 | s2->b64;
                    break;
                case K_XOR:
                    d->b64 = s1->b64;
                    s1->b64 = s1->b64 ^ s2->b64;
                    break;
                case K_ADD:
                    d->b64 = s1->b64;
                    s1->b64 = s1->b64 + s2->b64;
                    break;
                default:
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ATOM instruction");
            }
            break;
        case K_U32:
            switch (op)
            {
                case K_AND:
                    d->u32 = s1->u32;
                    s1->u32 = s1->u32 & s2->u32;
                    break;
                case K_OR:
                    d->u32 = s1->u32;
                    s1->u32 = s1->u32 | s2->u32;
                    break;
                case K_XOR:
                    d->u32 = s1->u32;
                    s1->u32 = s1->u32 ^ s2->u32;
                    break;
                case K_ADD:
                    d->u32 = s1->u32;
                    s1->u32 = s1->u32 + s2->u32;
                    break;
                default:
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ATOM instruction");
            }
            break;
        case K_U64:
            switch (op)
            {
                case K_AND:
                    d->u64 = s1->u64;
                    s1->u64 = s1->u64 & s2->u64;
                    break;
                case K_OR:
                    d->u64 = s1->u64;
                    s1->u64 = s1->u64 | s2->u64;
                    break;
                case K_XOR:
                    d->u64 = s1->u64;
                    s1->u64 = s1->u64 ^ s2->u64;
                    break;
                case K_ADD:
                    d->u64 = s1->u64;
                    s1->u64 = s1->u64 + s2->u64;
                    break;
                default:
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ATOM instruction");
            }
            break;
        case K_S32:
            switch (op)
            {
                case K_AND:
                    d->s32 = s1->s32;
                    s1->s32 = s1->s32 & s2->s32;
                    break;
                case K_OR:
                    d->s32 = s1->s32;
                    s1->s32 = s1->s32 | s2->s32;
                    break;
                case K_XOR:
                    d->s32 = s1->s32;
                    s1->s32 = s1->s32 ^ s2->s32;
                    break;
                case K_ADD:
                    d->s32 = s1->s32;
                    s1->s32 = s1->s32 + s2->s32;
                    break;
                default:
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ATOM instruction");
            }
            break;
        case K_F32:
            switch (op)
            {
                case K_ADD:
                    d->f32 = s1->f32;
                    s1->f32 = s1->f32 + s2->f32;
                    break;
                default:
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ATOM instruction");
            }
            break;
        default:
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ATOM instruction");
    }

    return 0;
}

int THREAD::DoBar(TREE * inst)
{
    // ONLY VERY SIMPLE SYNCHRONIZATION IMPLEMENTED!!!
    int start = 0;
    if (inst->GetChild(start)->GetType() == TREE_PRED)
        start++;
    assert(inst->GetChild(start)->GetType() == KI_BAR);
    start++;
    TREE * ttype = 0;
    TREE * osrc1 = 0;
    for (;; ++start)
    {
        TREE * t = inst->GetChild(start);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == TREE_TYPE)
            ttype = t;
        else if (gt == TREE_OPR)
        {
            if (osrc1 == 0)
            {
                osrc1 = t;
            }
            else
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected BAR instruction");
        }
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected BAR instruction");
    }
    assert(ttype != 0);
    assert(osrc1 != 0);
    bool sync = false;
    bool arrive = false;
    bool red = false;
    bool popc = false;
    int op = 0;
    int size = 0;
    for (int i = 0; ; ++i)
    {
        TREE * t = ttype->GetChild(i);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == K_SYNC)
            sync = true;
        else if (gt == K_ARRIVE)
            arrive = true;
        else if (gt == K_RED)
            red = true;
        else if (gt == K_POPC)
            popc = true;
        else if (gt == K_AND)
            op = K_AND;
        else if (gt == K_OR)
            op = K_OR;
        else if (gt = K_U32)
            size = K_U32;
    }
    this->device->unimplemented(arrive, "BAR.arrive not implemented.");
    this->device->unimplemented(red, "BAR.red not implemented.");
    
    int type = K_U32;
    TREE * src1 = osrc1->GetChild(0);

    TYPES::Types value1;
    char * dummy;
    TYPES::Types * s1 = &value1;

    if (src1->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src1->GetChild(0));
        switch (type)
        {
            case K_U32:
                s1->u32 = c.value.u32;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected BAR instruction");
        }
        this->device->unimplemented(c.value.u32 != 0, "BAR with non-zero argument not implemented.");
    } else if (src1->GetType() == T_WORD)
    {
        SYMBOL * ssrc1 = this->symbol_table->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        assert(ssrc1->size == this->device->Sizeof(type));
        TYPES::Types * psrc1_value = (TYPES::Types*)ssrc1->pvalue;
        switch (type)
        {
            case K_U32:
                s1->u32 = psrc1_value->u32;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected BAR instruction");
        }
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected BAR instruction");

    this->device->unimplemented(s1->u32 != 0, "BAR with non-zero argument not implemented.");

    // pack up the THREAD context.  Synchronize all assumed.

    return -KI_BAR;
}

int THREAD::DoBfe(TREE * inst)
{
    int start = 0;
    if (inst->GetChild(start)->GetType() == TREE_PRED)
        start++;
    assert(inst->GetChild(start)->GetType() == KI_BFE);
    start++;
    TREE * ttype = 0;
    TREE * odst = 0;
    TREE * osrc1 = 0;
    TREE * osrc2 = 0;
    TREE * osrc3 = 0;
    for (;; ++start)
    {
        TREE * t = inst->GetChild(start);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == TREE_TYPE)
            ttype = t;
        else if (gt == TREE_OPR)
        {
            if (odst == 0)
            {
                odst = t;
            } else if (osrc1 == 0)
            {
                osrc1 = t;
            } else if (osrc2 == 0)
            {
                osrc2 = t;
            } else if (osrc3 == 0)
            {
                osrc3 = t;
            }
            else
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected BFE instruction");
        }
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected BFE instruction");
    }
    assert(ttype != 0);
    assert(odst != 0);
    assert(osrc1 != 0);
    assert(osrc2 != 0);
    assert(osrc3 != 0);
    int type = 0;
    int storage_class = 0;
    for (int i = 0; ; ++i)
    {
        TREE * t = ttype->GetChild(i);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == K_U32 || gt == K_U64
            || gt == K_S32)
            type = gt;
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected BFE instruction");
    }
    assert(type != 0);

    TREE * dst = odst->GetChild(0);
    // Note src1 is not really only a source operand. It is a
    // destination operand, too.
    TREE * src1 = osrc1->GetChild(0);
    TREE * src2 = osrc2->GetChild(0);
    TREE * src3 = osrc3->GetChild(0);

    SYMBOL * sdst = 0;
    assert(dst->GetType() == T_WORD);
    sdst = this->symbol_table->FindSymbol(dst->GetText());
    assert(sdst != 0);
    TYPES::Types * d = (TYPES::Types*)sdst->pvalue;
    TYPES::Types value1;
    TYPES::Types * s1 = &value1;
    TYPES::Types value2;
    TYPES::Types * s2 = &value2;
    TYPES::Types value3;
    TYPES::Types * s3 = &value3;

    if (src1->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src1->GetChild(0));
        switch (type)
        {
            case K_U32:
                s1->u32 = c.value.u32;
                break;
            case K_U64:
                s1->u64 = c.value.u64;
                break;
            case K_S32:
                s1->s32 = c.value.s32;
                break;
            case K_S64:
                s1->s64 = c.value.s64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected BFE instruction");
        }
    } else if (src1->GetType() == T_WORD)
    {
        SYMBOL * ssrc1 = this->symbol_table->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        assert(ssrc1->size == this->device->Sizeof(type));
        TYPES::Types * psrc1_value = (TYPES::Types*)ssrc1->pvalue;
        switch (type)
        {
            case K_U32:
                s1->u32 = psrc1_value->u32;
                break;
            case K_U64:
                s1->u64 = psrc1_value->u64;
                break;
            case K_S32:
                s1->s32 = psrc1_value->s32;
                break;
            case K_S64:
                s1->s64 = psrc1_value->s64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected BFE instruction");
        }
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected BFE instruction");

    if (src2->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src2->GetChild(0));
        s2->u32 = c.value.u32;
    } else if (src2->GetType() == T_WORD)
    {
        SYMBOL * ssrc2 = this->symbol_table->FindSymbol(src2->GetText());
        assert(ssrc2 != 0);
        TYPES::Types * psrc2_value = (TYPES::Types*)ssrc2->pvalue;
        s2->u32 = psrc2_value->u32;
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected BFE instruction");

    if (src3->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src3->GetChild(0));
        s3->u32 = c.value.u32;
    } else if (src2->GetType() == T_WORD)
    {
        SYMBOL * ssrc3 = this->symbol_table->FindSymbol(src3->GetText());
        assert(ssrc3 != 0);
        TYPES::Types * psrc3_value = (TYPES::Types*)ssrc3->pvalue;
        s3->u32 = psrc3_value->u32;
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected BFE instruction");


    // Bits are numbered least significant bit to most significant bit.
    switch (type)
    {
        case K_U32:
            {
                std::bitset<32> in_bs;
                std::bitset<32> out_bs;
                unsigned _int32 a = s1->u32;
                in_bs = a;
                int msb = 31;
                int pos = s2->u32 & 0xff;
                int len = s3->u32 & 0xff;
                int sbit = 0; // U32
                unsigned __int32 r = 0;
                a = a >> pos;
                for (int i = 0; i <= msb; ++i)
                {
                    if (i < len && pos + i <= msb)
                        out_bs[i] = in_bs[pos+i];
                    else
                        out_bs[i] = sbit;
                }
                d->u32 = 0xffffffff & out_bs.to_ulong();
            }
            break;
        case K_U64:
            {
                std::bitset<64> in_bs;
                std::bitset<64> out_bs;
                unsigned _int64 a = s1->u64;
                in_bs = a;
                int msb = 63;
                int pos = s2->u32 & 0xff;
                int len = s3->u32 & 0xff;
                int sbit = 0; // U64
                unsigned __int64 r = 0;
                a = a >> pos;
                for (int i = 0; i <= msb; ++i)
                {
                    if (i < len && pos + i <= msb)
                        out_bs[i] = in_bs[pos+i];
                    else
                        out_bs[i] = sbit;
                }
                d->u64 = out_bs.to_ulong();
            }
            break;
        case K_S32:
            {
                std::bitset<32> in_bs;
                std::bitset<32> out_bs;
                _int32 a = s1->s32;
                in_bs = a;
                int msb = 31;
                int pos = s2->u32 & 0xff;
                int len = s3->u32 & 0xff;
                int sbit;
                if (len == 0)
                    sbit = 0;
                else
                    sbit = (a >> (min(pos + len - 1, msb))) & 1;
                __int32 r = 0;
                a = a >> pos;
                for (int i = 0; i <= msb; ++i)
                {
                    if (i < len && pos + i <= msb)
                        out_bs[i] = in_bs[pos+i];
                    else
                        out_bs[i] = sbit;
                }
                d->u32 = (__int32)(0xffffffff & out_bs.to_ulong());
            }
            break;
        case K_S64:
            {
                std::bitset<64> in_bs;
                std::bitset<64> out_bs;
                _int64 a = s1->s64;
                int msb = 63;
                int pos = s2->u32 & 0xff;
                int len = s3->u32 & 0xff;
                int sbit;
                if (len == 0)
                    sbit = 0;
                else
                    sbit = (a >> (min(pos + len - 1, msb))) & 1;
                __int64 r = 0;
                a = a >> pos;
                for (int i = 0; i <= msb; ++i)
                {
                    if (i < len && pos + i <= msb)
                        out_bs[i] = in_bs[pos+i];
                    else
                        out_bs[i] = sbit;
                }
                d->s64 = (__int64) out_bs.to_ulong();
            }
            break;
        default:
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected BFE instruction");
    }

    return 0;
}

int THREAD::DoBfi(TREE * inst)
{
    throw new EMULATED_DEVICE::EMU_ERROR("BFI unimplemented");
}

int THREAD::DoBfind(TREE * inst)
{
    throw new EMULATED_DEVICE::EMU_ERROR("BFIND unimplemented");
}

int THREAD::DoBrev(TREE * inst)
{
    throw new EMULATED_DEVICE::EMU_ERROR("BREV unimplemented");
}

int THREAD::DoBra(TREE * inst)
{
    int start = 0;
    for (;;)
    {
        TREE * t = inst->GetChild(start);
        assert(t != 0);
        if (t->GetType() == TREE_OPR)
            break;
        start++;
    }
    TREE * opr = inst->GetChild(start);
    TREE * dst = opr->GetChild(0);
    assert(dst->GetType() == T_WORD);
    SYMBOL * sdst = this->symbol_table->FindSymbol(dst->GetText());
    assert (sdst != 0);
    return (int)sdst->pvalue;
}

int THREAD::DoBrkpt(TREE * inst)
{
    throw new EMULATED_DEVICE::EMU_ERROR("BRKPT unimplemented");
}


int THREAD::DoCall(TREE * inst)
{
    int start = 0;
    if (inst->GetChild(start)->GetType() == TREE_PRED)
        start++;
    assert(inst->GetChild(start)->GetType() == KI_CALL);
    start++;
    TREE * ttype = 0;
    int start_rets = 0;
    int start_params = 0;
    TREE * tfunc = 0;
    for (;; ++start)
    {
        TREE * t = inst->GetChild(start);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == TREE_TYPE)
            ttype = t;
        else if (gt == TREE_OPR)
        {
            if (start_rets == 0)
                start_rets = start;
            else if (start_params == 0)
                start_params = start;
        } else if (gt == T_WORD)
            tfunc = t;
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected CALL instruction");
    }

    // Call of vprintf only thing supported...
    if (strcmp(tfunc->GetText(), "vprintf") != 0)
    {
        throw new EMULATED_DEVICE::EMU_ERROR("Only CALL of vprintf implemented.\n");
    }

    bool uni = false;
    int i = 0;
    for (;; ++i)
    {
        TREE * t = ttype->GetChild(i);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == K_UNI)
            uni = true;
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected CALL instruction");
    }

    // allow only _ for return; allow only direct calls.
    int once = 0;
    for (;; ++start_rets)
    {
        TREE * t = inst->GetChild( start_rets);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == TREE_OPR)
        {
            TREE * u = t->GetChild(0);
            //if (u->GetType() == T_UNDERSCORE)
                once++;
            //else
            //    throw new EMULATED_DEVICE::EMU_ERROR("CALL non-underscore return unimplemented.");
        } else
            break;
    }
    if (once != 1)
        throw new EMULATED_DEVICE::EMU_ERROR("CALL with multiple returns unimplemented.");

    std::vector<SYMBOL*> stack;
    for (;; ++start_params)
    {
        TREE * t = inst->GetChild( start_params);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == TREE_OPR)
        {
            TREE * u = t->GetChild(0);
            if (u->GetType() == T_WORD)
            {
                // load value into stack...
                SYMBOL * param = this->symbol_table->FindSymbol(u->GetText());
                assert(param != 0);
                this->device->unimplemented(param->storage_class != K_PARAM, "CALL indirect form unimplemented.");
                stack.push_back(param);
            }
            else
                this->device->unimplemented("CALL unimplemented parameter type.");
        } else
            break;
    }

//#define va_start_assign(list, var) ((list) = ((char*)&var))
//#define va_assign(lvalp, type, rval) \
//*((type*)lvalp) = rval; \
//va_arg(lvalp, type);
//
//    va_list out;
//    void * aligned[100]; /* aligned buffer of ridiculously large size */
//    va_start_assign(out, aligned[0]);
//    for (int i = 1; i < stack.size(); ++i)
//    {
//        SYMBOL * s = stack[i];
//        switch (s->type)
//        {
//        case K_U32:
//            va_assign(out, unsigned __int32, ((TYPES::Types*)s->pvalue)->u32);
//            break;
//        case K_S32:
//            va_assign(out, signed __int32, ((TYPES::Types*)s->pvalue)->u32);
//            break;
//        case K_F32:
//            va_assign(out, float, ((TYPES::Types*)s->pvalue)->f32);
//            break;
//        case K_F64:
//            va_assign(out, double, ((TYPES::Types*)s->pvalue)->f64);
//            break;
//        default:
//            throw new EMULATED_DEVICE::EMU_ERROR("EMULATED_DEVICE::EMU_ERROR type for vprintf\n");
//            break;
//        }
//    }

    TYPES::Types* p1 = (TYPES::Types*)stack[0]->pvalue;
    TYPES::Types* p2 = (TYPES::Types*)stack[1]->pvalue;

    // Call depending on size of pointer.
    k_vprintf<__int32>( p1->pchar, static_cast<va_list>(p2->pvoid));
//    va_end(out);

    return 0;
}

int THREAD::DoClz(TREE * inst)
{
    throw new EMULATED_DEVICE::EMU_ERROR("CLZ unimplemented");
}

int THREAD::DoCnot(TREE * inst)
{
    int start = 0;
    if (inst->GetChild( start)->GetType() == TREE_PRED)
        start++;
    assert(inst->GetChild(start)->GetType() == KI_CNOT);
    start++;
    TREE * ttype = 0;
    TREE * odst = 0;
    TREE * osrc1 = 0;
    for (;; ++start)
    {
        TREE * t = inst->GetChild(start);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == TREE_TYPE)
            ttype = t;
        else if (gt == TREE_OPR)
        {
            if (odst == 0)
            {
                odst = t;
            } else if (osrc1 == 0)
            {
                osrc1 = t;
            }
            else
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected CNOT instruction");
        } else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected CNOT instruction");
    }
    assert(ttype != 0);
    assert(odst != 0);
    assert(osrc1 != 0);
    int type = 0;
    for (int i = 0; ; ++i)
    {
        TREE * t = ttype->GetChild(i);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == K_PRED || gt == K_B16 || gt == K_B32 || gt == K_B64)
            type = gt;
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected CNOT instruction");
    }
    assert(type != 0);
    TREE * dst = odst->GetChild(0);
    TREE * src1 = osrc1->GetChild(0);

    SYMBOL * sdst = 0;
    if (dst->GetType() == T_WORD)
    {
        sdst = this->symbol_table->FindSymbol(dst->GetText());
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected CNOT instruction");

    TYPES::Types value1;
    char * dummy;
    TYPES::Types * d = (TYPES::Types*)sdst->pvalue;
    TYPES::Types * s1 = &value1;

    if (src1->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src1->GetChild( 0));
        switch (type)
        {
            case K_B16:
                s1->b16 = c.value.b16;
                break;
            case K_B32:
                s1->b32 = c.value.b32;
                break;
            case K_B64:
                s1->b64 = c.value.b64;
                break;
            case K_PRED:
                s1->pred = c.value.pred;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected CNOT instruction");
        }
    } else if (src1->GetType() == T_WORD)
    {
        SYMBOL * ssrc1 = this->symbol_table->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        assert(ssrc1->size == this->device->Sizeof(type));
        TYPES::Types * psrc1_value = (TYPES::Types*)ssrc1->pvalue;
        switch (type)
        {
            case K_B16:
                s1->b16 = psrc1_value->b16;
                break;
            case K_B32:
                s1->b32 = psrc1_value->b32;
                break;
            case K_B64:
                s1->b64 = psrc1_value->b64;
                break;
            case K_PRED:
                s1->pred = psrc1_value->pred;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected CNOT instruction");
        }
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected CNOT instruction");

    switch (type)
    {
        case K_B16:
            d->b16 = s1->b16 == 0 ? 1 : 0;
            break;
        case K_B32:
            d->b32 = s1->b32 == 0 ? 1 : 0;
            break;
        case K_S64:
            d->b64 = s1->b64 == 0 ? 1 : 0;
            break;
        case K_PRED:
            d->pred = s1->pred == 0 ? 1 : 0;
            break;
        default:
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected CNOT instruction");
    }
    return 0;
}

int THREAD::DoCopysign(TREE * inst)
{
    throw new EMULATED_DEVICE::EMU_ERROR("COPYSIGN unimplemented");
}

int THREAD::DoCos(TREE * inst)
{
    throw new EMULATED_DEVICE::EMU_ERROR("COS unimplemented");
}

int THREAD::DoCvt(TREE * inst)
{
    int start = 0;
    if (inst->GetChild(start)->GetType() == TREE_PRED)
        start++;
    assert(inst->GetChild(start)->GetType() == KI_CVT);
    start++;
    assert(inst->GetChild(start)->GetType() == TREE_TYPE);
    TREE * type = inst->GetChild(start);
    start++;
    bool ftz = false;
    bool sat = false;
    TREE * tirnd = 0;
    TREE * tfrnd = 0;
    int irnd = 0;
    int frnd = 0;
    int src_type = 0;
    int dst_type = 0;
    int i = 0;
    for (;; ++i)
    {
        TREE * t = type->GetChild(i);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == TREE_IRND)
            tirnd = t;
        else if (gt == TREE_FRND)
            tfrnd = t;
        else if (gt == K_FTZ)
            ftz = true;
        else if (gt == K_SAT)
            sat = true;
        else {
            if (dst_type == 0)
                dst_type = gt;
            else
                src_type = gt;
        }
    }
    assert(ftz == false); // unimplemented.
    assert(sat == false); // unimplemented.
    if (tirnd != 0)
    {
        for (int i = 0; ; ++i)
        {
            TREE * t = tirnd->GetChild(i);
            if (t == 0)
                break;
            int gt = t->GetType();
            if (gt == K_RNI || gt == K_RZI || gt == K_RMI || gt == K_RPI)
                irnd = gt;
            else
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected CVT instruction");
        }
    }
    if (tfrnd != 0)
    {
        for (int i = 0; ; ++i)
        {
            TREE * t = tfrnd->GetChild(i);
            if (t == 0)
                break;
            int gt = t->GetType();
            if (gt == K_RN || gt == K_RZ || gt == K_RM || gt == K_RP)
                frnd = gt;
            else
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected CVT instruction");
        }
    }

    TYPES::Types * dst_value;
    TYPES::Types * src_value;

    TREE * o1 = inst->GetChild(start++);
    assert(o1->GetType() == TREE_OPR);
    assert(o1->GetChild(0)->GetType() == T_WORD);
    TREE * o2 = inst->GetChild(start++);
    assert(o2->GetType() == TREE_OPR);
    assert(o2->GetChild(0)->GetType() == T_WORD);

    SYMBOL * s1 = this->symbol_table->FindSymbol(o1->GetChild(0)->GetText());
    assert(s1 != 0);
    SYMBOL * s2 = this->symbol_table->FindSymbol(o2->GetChild(0)->GetText());
    assert(s2 != 0);

    dst_value = (TYPES::Types*)s1->pvalue;

    // handle .x, .y, .z stuff.
    if (strcmp(s2->typestring, "dim3") == 0)
    {
        // Get qualifier of the structure.
        TREE * tqual = o2->GetChild(1);
        assert(tqual != 0);
        int qual = tqual->GetType();
        if (qual == K_X)
        {
            src_value = (TYPES::Types*) &(((dim3*)s2->pvalue)->x);
        } else if (qual == K_Y)
        {
            src_value = (TYPES::Types*) &(((dim3*)s2->pvalue)->y);
        } else if (qual == K_Z)
        {
            src_value = (TYPES::Types*) &(((dim3*)s2->pvalue)->z);
        }
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected CVT instruction");
    } else
        src_value = (TYPES::Types*)s2->pvalue;

    switch (src_type)
    {
        case K_U8:
            switch (dst_type)
            {
                case K_U8:
                    dst_value->u8 = src_value->u8;
                    break;
                case K_U16:
                    dst_value->u16 = src_value->u8;
                    break;
                case K_U32:
                    dst_value->u32 = src_value->u8;
                    break;
                case K_U64:
                    dst_value->u64 = src_value->u8;
                    break;
                case K_S8:
                    dst_value->s8 = src_value->u8;
                    break;
                case K_S16:
                    dst_value->s16 = src_value->u8;
                    break;
                case K_S32:
                    dst_value->s32 = src_value->u8;
                    break;
                case K_S64:
                    dst_value->s64 = src_value->u8;
                    break;
                case K_F32:
                    dst_value->f32 = src_value->u8;
                    break;
                case K_F64:
                    dst_value->f64 = src_value->u8;
                    break;
                default:
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected CVT instruction");
            }
            break;
        case K_U16:
            switch (dst_type)
            {
                case K_U8:
                    dst_value->u8 = src_value->u16;
                    break;
                case K_U16:
                    dst_value->u16 = src_value->u16;
                    break;
                case K_U32:
                    dst_value->u32 = src_value->u16;
                    break;
                case K_U64:
                    dst_value->u64 = src_value->u16;
                    break;
                case K_S8:
                    dst_value->s8 = src_value->u16;
                    break;
                case K_S16:
                    dst_value->s16 = src_value->u16;
                    break;
                case K_S32:
                    dst_value->s32 = src_value->u16;
                    break;
                case K_S64:
                    dst_value->s64 = src_value->u16;
                    break;
                case K_F32:
                    dst_value->f32 = src_value->u16;
                    break;
                case K_F64:
                    dst_value->f64 = src_value->u16;
                    break;
                default:
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected CVT instruction");
            }
            break;
        case K_U32:
            switch (dst_type)
            {
                case K_U8:
                    dst_value->u8 = src_value->u32;
                    break;
                case K_U16:
                    dst_value->u16 = src_value->u32;
                    break;
                case K_U32:
                    dst_value->u32 = src_value->u32;
                    break;
                case K_U64:
                    dst_value->u64 = src_value->u32;
                    break;
                case K_S8:
                    dst_value->s8 = src_value->u32;
                    break;
                case K_S16:
                    dst_value->s16 = src_value->u32;
                    break;
                case K_S32:
                    dst_value->s32 = src_value->u32;
                    break;
                case K_S64:
                    dst_value->s64 = src_value->u32;
                    break;
                case K_F32:
                    dst_value->f32 = src_value->u32;
                    break;
                case K_F64:
                    dst_value->f64 = src_value->u32;
                    break;
                default:
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected CVT instruction");
            }
            break;
        case K_U64:
            switch (dst_type)
            {
                case K_U8:
                    dst_value->u8 = src_value->u64;
                    break;
                case K_U16:
                    dst_value->u16 = src_value->u64;
                    break;
                case K_U32:
                    dst_value->u32 = src_value->u64;
                    break;
                case K_U64:
                    dst_value->u64 = src_value->u64;
                    break;
                case K_S8:
                    dst_value->s8 = src_value->u64;
                    break;
                case K_S16:
                    dst_value->s16 = src_value->u64;
                    break;
                case K_S32:
                    dst_value->s32 = src_value->u64;
                    break;
                case K_S64:
                    dst_value->s64 = src_value->u64;
                    break;
                case K_F32:
                    dst_value->f32 = src_value->u64;
                    break;
                case K_F64:
                    dst_value->f64 = src_value->u64;
                    break;
                default:
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected CVT instruction");
            }
            break;
        case K_S8:
            switch (dst_type)
            {
                case K_U8:
                    dst_value->u8 = src_value->s8;
                    break;
                case K_U16:
                    dst_value->u16 = src_value->s8;
                    break;
                case K_U32:
                    dst_value->u32 = src_value->s8;
                    break;
                case K_U64:
                    dst_value->u64 = src_value->s8;
                    break;
                case K_S8:
                    dst_value->s8 = src_value->s8;
                    break;
                case K_S16:
                    dst_value->s16 = src_value->s8;
                    break;
                case K_S32:
                    dst_value->s32 = src_value->s8;
                    break;
                case K_S64:
                    dst_value->s64 = src_value->s8;
                    break;
                case K_F32:
                    dst_value->f32 = src_value->s8;
                    break;
                case K_F64:
                    dst_value->f64 = src_value->s8;
                    break;
                default:
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected CVT instruction");
            }
            break;
        case K_S16:
            switch (dst_type)
            {
                case K_U8:
                    dst_value->u8 = src_value->s16;
                    break;
                case K_U16:
                    dst_value->u16 = src_value->s16;
                    break;
                case K_U32:
                    dst_value->u32 = src_value->s16;
                    break;
                case K_U64:
                    dst_value->u64 = src_value->s16;
                    break;
                case K_S8:
                    dst_value->s8 = src_value->s16;
                    break;
                case K_S16:
                    dst_value->s16 = src_value->s16;
                    break;
                case K_S32:
                    dst_value->s32 = src_value->s16;
                    break;
                case K_S64:
                    dst_value->s64 = src_value->s16;
                    break;
                case K_F32:
                    dst_value->f32 = src_value->s16;
                    break;
                case K_F64:
                    dst_value->f64 = src_value->s16;
                    break;
                default:
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected CVT instruction");
            }
            break;
        case K_S32:
            switch (dst_type)
            {
                case K_U8:
                    dst_value->u8 = src_value->s32;
                    break;
                case K_U16:
                    dst_value->u16 = src_value->s32;
                    break;
                case K_U32:
                    dst_value->u32 = src_value->s32;
                    break;
                case K_U64:
                    dst_value->u64 = src_value->s32;
                    break;
                case K_S8:
                    dst_value->s8 = src_value->s32;
                    break;
                case K_S16:
                    dst_value->s16 = src_value->s32;
                    break;
                case K_S32:
                    dst_value->s32 = src_value->s32;
                    break;
                case K_S64:
                    dst_value->s64 = src_value->s32;
                    break;
                case K_F32:
                    dst_value->f32 = src_value->s32;
                    break;
                case K_F64:
                    dst_value->f64 = src_value->s32;
                    break;
                default:
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected CVT instruction");
            }
            break;
        case K_S64:
            switch (dst_type)
            {
                case K_U8:
                    dst_value->u8 = src_value->s64;
                    break;
                case K_U16:
                    dst_value->u16 = src_value->s64;
                    break;
                case K_U32:
                    dst_value->u32 = src_value->s64;
                    break;
                case K_U64:
                    dst_value->u64 = src_value->s64;
                    break;
                case K_S8:
                    dst_value->s8 = src_value->s64;
                    break;
                case K_S16:
                    dst_value->s16 = src_value->s64;
                    break;
                case K_S32:
                    dst_value->s32 = src_value->s64;
                    break;
                case K_S64:
                    dst_value->s64 = src_value->s64;
                    break;
                case K_F32:
                    dst_value->f32 = src_value->s64;
                    break;
                case K_F64:
                    dst_value->f64 = src_value->s64;
                    break;
                default:
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected CVT instruction");
            }
            break;
        case K_F32:
            switch (dst_type)
            {
                case K_U8:
                    dst_value->u8 = src_value->f32;
                    break;
                case K_U16:
                    dst_value->u16 = src_value->f32;
                    break;
                case K_U32:
                    dst_value->u32 = src_value->f32;
                    break;
                case K_U64:
                    dst_value->u64 = src_value->f32;
                    break;
                case K_S8:
                    dst_value->s8 = src_value->f32;
                    break;
                case K_S16:
                    dst_value->s16 = src_value->f32;
                    break;
                case K_S32:
                    dst_value->s32 = src_value->f32;
                    break;
                case K_S64:
                    dst_value->s64 = src_value->f32;
                    break;
                case K_F32:
                    dst_value->f32 = src_value->f32;
                    break;
                case K_F64:
                    dst_value->f64 = src_value->f32;
                    break;
                default:
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected CVT instruction");
            }
            break;
        case K_F64:
            switch (dst_type)
            {
                case K_U8:
                    dst_value->u8 = src_value->f64;
                    break;
                case K_U16:
                    dst_value->u16 = src_value->f64;
                    break;
                case K_U32:
                    dst_value->u32 = src_value->f64;
                    break;
                case K_U64:
                    dst_value->u64 = src_value->f64;
                    break;
                case K_S8:
                    dst_value->s8 = src_value->f64;
                    break;
                case K_S16:
                    dst_value->s16 = src_value->f64;
                    break;
                case K_S32:
                    dst_value->s32 = src_value->f64;
                    break;
                case K_S64:
                    dst_value->s64 = src_value->f64;
                    break;
                case K_F32:
                    dst_value->f32 = src_value->f64;
                    break;
                case K_F64:
                    dst_value->f64 = src_value->f64;
                    break;
                default:
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected CVT instruction");
            }
            break;

        default:
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected CVT instruction");
    }
    return 0;
}


int THREAD::DoCvta(TREE * inst)
{
    // Assign source to destination.
    int start = 0;
    if (inst->GetChild(start)->GetType() == TREE_PRED)
        start++;
    assert(inst->GetChild(start)->GetType() == KI_CVTA);
    start++;
    TREE * ttype = 0;
    TREE * odst = 0;
    TREE * osrc = 0;
    for (;; ++start)
    {
        TREE * t = inst->GetChild(start);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == TREE_TYPE)
            ttype = t;
        else if (gt == TREE_OPR)
        {
            if (odst == 0)
            {
                odst = t;
            } else if (osrc == 0)
            {
                osrc = t;
            }
            else
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected CVTA instruction");
        }
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected CVTA instruction");
    }
    assert(ttype != 0);
    assert(odst != 0);
    assert(osrc != 0);
    int storage_class = 0;
    bool to = false;
    for (int i = 0; ; ++i)
    {
        TREE * t = ttype->GetChild(i);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == K_U32 || gt == K_U64)
            ttype = t;
        else if (gt == K_SHARED || gt == K_GLOBAL || gt == K_LOCAL || gt == K_CONST)
            storage_class = gt;
        else if (gt == K_TO)
            to = true;
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected CVTA instruction");
    }
    assert(ttype);
    int type = ttype->GetType();
    TREE * dst = odst->GetChild( 0);
    TREE * src = osrc->GetChild(0);
    SYMBOL * sdst = 0;
    SYMBOL * ssrc = 0;
    if (dst->GetType() == T_WORD)
    {
        sdst = this->symbol_table->FindSymbol(dst->GetText());
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected CVTA instruction");

    TYPES::Types * d;
    TYPES::Types * s;
    TYPES::Types value;
    char * dummy;

    d = (TYPES::Types*)sdst->pvalue;

    // different semantics depending on if register, global, etc.
    if (src->GetType() == T_WORD)
    {
        ssrc = this->symbol_table->FindSymbol(src->GetText());
        // Various types of id's to handle:
        assert(ssrc != 0);
        switch (ssrc->storage_class)
        {
            case K_GLOBAL:
            case K_LOCAL:
            case K_PARAM:
            case K_SHARED:
            case K_CONST:
                // names in instructions refer to the address of the
                // variable, not the contents.
                s = (TYPES::Types*)&ssrc->pvalue;
                break;
            case K_REG:
                // names in instructions refer to the contents of the
                // register.
                s = (TYPES::Types*)ssrc->pvalue;
                break;
        }
    }

    switch (type)
    {
        case K_U32:
            d->u32 = s->u32;
            break;
        case K_U64:
            d->u64 = s->u64;
            break;
        default:
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected CVTA instruction");
    }
    return 0;
}

int THREAD::DoDiv(TREE * inst)
{
    int start = 0;
    if (inst->GetChild(start)->GetType() == TREE_PRED)
        start++;
    assert(inst->GetChild(start)->GetType() == KI_DIV);
    start++;
    assert(inst->GetChild(start)->GetType() == TREE_TYPE);
    TREE * ttype = inst->GetChild(start);
    start++;
    TREE * odst = 0;
    TREE * osrc1 = 0;
    TREE * osrc2 = 0;
    for (;; ++start)
    {
        TREE * t = inst->GetChild(start);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == TREE_OPR)
        {
            if (odst == 0)
            {
                odst = t;
            } else if (osrc1 == 0)
            {
                osrc1 = t;
            } else if (osrc2 == 0)
            {
                osrc2 = t;
            } else assert(false);
        } else assert(false);
    }
    assert(ttype != 0);
    assert(odst != 0);
    assert(osrc1 != 0);
    assert(osrc2 != 0);
    bool ftz = false;
    TREE * tfrnd = 0;
    for (int i = 0; ; ++i)
    {
        TREE * t = ttype->GetChild(i);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == K_FTZ)
            ftz = true;
        else if (gt == K_U16 || gt == K_U32 || gt == K_U64
                 || gt == K_S16 || gt == K_S32 || gt == K_S64)
            ttype = t;
        else if (gt == K_F32 || gt == K_F64)
            ttype = t;
        else if (gt == K_RN || gt == K_RZ || gt == K_RM || gt == K_RP)
            tfrnd = t;
        else if (gt == K_FULL || gt == K_APPROX)
            ;
        else assert(false);
    }
    assert(ttype != 0);
    assert(ftz == 0);  // unimplemented.
    int type = ttype->GetType();

    TREE * dst = odst->GetChild(0);
    TREE * src1 = osrc1->GetChild(0);
    TREE * src2 = osrc2->GetChild(0);

    TYPES::Types * pdst_value;
    TYPES::Types * psrc1_value;
    TYPES::Types * psrc2_value;
    TYPES::Types src1_value;// used if literal
    TYPES::Types src2_value;// used if literal

    SYMBOL * sdst = 0;
    SYMBOL * ssrc1 = 0;
    SYMBOL * ssrc2 = 0;
    assert(dst->GetType() == T_WORD);
    sdst = this->symbol_table->FindSymbol(dst->GetText());
    char * dummy;

    TYPES::Types value1; // used if literal
    TYPES::Types value2; // used if literal
    TYPES::Types * d = (TYPES::Types*)sdst->pvalue;
    TYPES::Types * s1 = &value1;
    TYPES::Types * s2 = &value2;

    if (src1->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src1->GetChild( 0));
        switch (type)
        {
            case K_U16:
                s1->u16 = c.value.u16;
                break;
            case K_S16:
                s1->s16 = c.value.s16;
                break;
            case K_U32:
                s1->u32 = c.value.u32;
                break;
            case K_S32:
                s1->s32 = c.value.s32;
                break;
            case K_U64:
                s1->u64 = c.value.u64;
                break;
            case K_S64:
                s1->s64 = c.value.s64;
                break;
            case K_F32:
                s1->f32 = c.value.f32;
                break;
            case K_F64:
                s1->f64 = c.value.f64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected DIV instruction");
        }
    } else if (src1->GetType() == T_WORD)
    {
        ssrc1 = this->symbol_table->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        s1 = (TYPES::Types*)ssrc1->pvalue;
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected DIV instruction");

    if (src2->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src2->GetChild(0));
        switch (type)
        {
            case K_U16:
                s2->u16 = c.value.u16;
                break;
            case K_S16:
                s2->s16 = c.value.s16;
                break;
            case K_U32:
                s2->u32 = c.value.u32;
                break;
            case K_S32:
                s2->s32 = c.value.s32;
                break;
            case K_U64:
                s2->u64 = c.value.u64;
                break;
            case K_S64:
                s2->s64 = c.value.s64;
                break;
            case K_F32:
                s2->f32 = c.value.f32;
                break;
            case K_F64:
                s2->f64 = c.value.f64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected DIV instruction");
        }
    } else if (src2->GetType() == T_WORD)
    {
        ssrc2 = this->symbol_table->FindSymbol(src2->GetText());
        assert(ssrc2 != 0);
        s2 = (TYPES::Types*)ssrc2->pvalue;
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected DIV instruction");

    switch (type)
    {
        case K_U16:
            d->u16 = s1->u16 / s2->u16;
            break;
        case K_S16:
            d->s16 = s1->s16 / s2->s16;
            break;
        case K_U32:
            d->u32 = s1->u32 / s2->u32;
            break;
        case K_S32:
            d->s32 = s1->s32 / s2->s32;
            break;
        case K_U64:
            d->u64 = s1->u64 / s2->u64;
            break;
        case K_S64:
            d->s64 = s1->s64 / s2->s64;
            break;
        case K_F32:
            d->f32 = s1->f32 / s2->f32;
            break;
        case K_F64:
            d->f64 = s1->f64 / s2->f64;
            break;
        default:
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected DIV instruction");
    }
    return 0;
}

int THREAD::DoEx2(TREE * inst)
{
    int start = 0;
    if (inst->GetChild(start)->GetType() == TREE_PRED)
        start++;
    assert(inst->GetChild(start)->GetType() == KI_EX2);
    start++;
    TREE * ttype = 0;
    TREE * odst = 0;
    TREE * osrc1 = 0;
    for (;; ++start)
    {
        TREE * t = inst->GetChild(start);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == TREE_TYPE)
            ttype = t;
        else if (gt == TREE_OPR)
        {
            if (odst == 0)
            {
                odst = t;
            } else if (osrc1 == 0)
            {
                osrc1 = t;
            }
            else
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected EX2 instruction");
        }
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected EX2 instruction");
    }
    assert(ttype != 0);
    assert(odst != 0);
    assert(osrc1 != 0);
    bool ftz = false;
    int rnd = 0;
    for (int i = 0; ; ++i)
    {
        TREE * t = ttype->GetChild(i);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == K_F32 || gt == K_F64)
            ttype = t;
        else if (gt == K_FTZ)
            ftz = true;
        else if (gt == K_APPROX)
            ;
        else if (gt == K_RN || gt == K_RZ || gt == K_RM || gt == K_RP)
            rnd = gt;
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected EX2 instruction");
    }
    assert(ttype != 0);
    this->device->unimplemented(ftz, "EX2.ftz not implemented.");

    int type = ttype->GetType();
    TREE * dst = odst->GetChild(0);
    TREE * src1 = osrc1->GetChild(0);

    SYMBOL * sdst = 0;
    if (dst->GetType() == T_WORD)
    {
        sdst = this->symbol_table->FindSymbol(dst->GetText());
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected EX2 instruction");

    TYPES::Types value1;
    char * dummy;
    TYPES::Types * d = (TYPES::Types*)sdst->pvalue;
    TYPES::Types * s1 = &value1;

    if (src1->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src1->GetChild( 0));
        switch (type)
        {
            case K_F32:
                s1->f32 = c.value.f32;
                break;
            case K_F64:
                s1->f64 = c.value.f64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected EX2 instruction");
        }
    } else if (src1->GetType() == T_WORD)
    {
        SYMBOL * ssrc1 = this->symbol_table->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        assert(ssrc1->size == this->device->Sizeof(type));
        TYPES::Types * psrc1_value = (TYPES::Types*)ssrc1->pvalue;
        switch (type)
        {
            case K_F32:
                s1->f32 = psrc1_value->f32;
                break;
            case K_F64:
                s1->f64 = psrc1_value->f64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected EX2 instruction");
        }
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected EX2 instruction");

    switch (type)
    {
        case K_F32:
            d->f32 = pow(2, s1->f32);
            break;
        case K_F64:
            d->f64 = pow(2, s1->f64);
            break;
        default:
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected EX2 instruction");
    }
    return 0;
}

int THREAD::DoExit(TREE * inst)
{
    return -KI_EXIT;
}

int THREAD::DoFma(TREE * inst)
{
    // Multiply register and/or constants, and store in a register.
    int start = 0;
    if (inst->GetChild(start)->GetType() == TREE_PRED)
        start++;
    assert(inst->GetChild(start)->GetType() == KI_FMA);
    start++;
    assert(inst->GetChild(start)->GetType() == TREE_TYPE);
    TREE * ttype = inst->GetChild(start);
    start++;
    TREE * odst = 0;
    TREE * osrc1 = 0;
    TREE * osrc2 = 0;
    TREE * osrc3 = 0;
    for (;; ++start)
    {
        TREE * t = inst->GetChild(start);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == TREE_OPR)
        {
            if (odst == 0)
            {
                odst = t;
            } else if (osrc1 == 0)
            {
                osrc1 = t;
            } else if (osrc2 == 0)
            {
                osrc2 = t;
            } else if (osrc3 == 0)
            {
                osrc3 = t;
            } else assert(false);
        } else assert(false);
    }
    assert(ttype != 0);
    assert(odst != 0);
    assert(osrc1 != 0);
    assert(osrc2 != 0);
    assert(osrc3 != 0);
    bool sat = false;
    bool ftz = false;
    TREE * tfrnd = 0;
    for (int i = 0; ; ++i)
    {
        TREE * t = ttype->GetChild(i);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == K_SAT)
            sat = true;
        else if (gt == K_FTZ)
            ftz = true;
        else if (gt == K_RN || gt == K_RZ || gt == K_RM || gt == K_RP)
            tfrnd = t;
        else if (gt == K_F32 || gt == K_F64)
            ttype = t;
        else assert(false);
    }
    assert(ttype != 0);
    assert(sat == 0); // unimplemented
    assert(ftz == 0);  // unimplemented.
    int type = ttype->GetType();

    TREE * dst = odst->GetChild(0);
    TREE * src1 = osrc1->GetChild(0);
    TREE * src2 = osrc2->GetChild(0);
    TREE * src3 = osrc3->GetChild(0);

    SYMBOL * sdst = 0;
    SYMBOL * ssrc1 = 0;
    SYMBOL * ssrc2 = 0;
    SYMBOL * ssrc3 = 0;
    assert(dst->GetType() == T_WORD);
    sdst = this->symbol_table->FindSymbol(dst->GetText());
    char * dummy;

    TYPES::Types value1; // used if literal
    TYPES::Types value2; // used if literal
    TYPES::Types value3; // used if literal
    TYPES::Types * d = (TYPES::Types*)sdst->pvalue;
    TYPES::Types * s1 = &value1;
    TYPES::Types * s2 = &value2;
    TYPES::Types * s3 = &value2;

    if (src1->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src1->GetChild( 0));
        switch (type)
        {
            case K_F32:
                s1->f32 = c.value.f32;
                break;
            case K_F64:
                s1->f64 = c.value.f64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected FMA instruction");
        }
    } else if (src1->GetType() == T_WORD)
    {
        ssrc1 = this->symbol_table->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        s1 = (TYPES::Types*)ssrc1->pvalue;
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected FMA instruction");

    if (src2->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src2->GetChild(0));
        switch (type)
        {
            case K_F32:
                s2->f32 = c.value.f32;
                break;
            case K_F64:
                s2->f64 = c.value.f64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected FMA instruction");
        }
    } else if (src2->GetType() == T_WORD)
    {
        ssrc2 = this->symbol_table->FindSymbol(src2->GetText());
        assert(ssrc2 != 0);
        s2 = (TYPES::Types*)ssrc2->pvalue;
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected FMA instruction");

    if (src3->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src3->GetChild(0));
        switch (type)
        {
            case K_F32:
                s3->f32 = c.value.f32;
                break;
            case K_F64:
                s3->f64 = c.value.f64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected FMA instruction");
        }
    } else if (src3->GetType() == T_WORD)
    {
        ssrc3 = this->symbol_table->FindSymbol(src3->GetText());
        assert(ssrc3 != 0);
        s3 = (TYPES::Types*)ssrc3->pvalue;
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected FMA instruction");

    switch (type)
    {
        case K_F32:
            d->f32 = s1->f32 * s2->f32 + s3->f32;
            break;
        case K_F64:
            d->f64 = s1->f64 * s2->f64 + s3->f64;
            break;
        default:
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected FMA instruction");
    }
    return 0;
}

int THREAD::DoIsspacep(TREE * inst)
{
    throw new EMULATED_DEVICE::EMU_ERROR("ISSPACEP unimplemented");
}

int THREAD::DoLd(TREE * inst)
{
    int start = 0;
    if (inst->GetChild(start)->GetType() == TREE_PRED)
        start++;
    assert(inst->GetChild(start)->GetType() == KI_LD);
    start++;
    TREE * ttype = 0;
    TREE * odst = 0;
    TREE * osrc = 0;
    for (;; ++start)
    {
        TREE * t = inst->GetChild(start);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == TREE_TYPE)
            ttype = t;
        else if (gt == TREE_OPR)
        {
            if (odst == 0)
            {
                odst = t;
            } else if (osrc == 0)
            {
                osrc = t;
            }
            else
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected LD instruction");
        }
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected LD instruction");
    }
    assert(ttype != 0);
    assert(odst != 0);
    assert(osrc != 0);
    int ss = 0;
    int cop = 0;
    int vec = 0;
    int vol = 0;
    for (int i = 0; ; ++i)
    {
        TREE * t = ttype->GetChild(i);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == K_CONST || gt == K_GLOBAL || gt == K_LOCAL || gt == K_PARAM || gt == K_SHARED)
            ss = gt;
        else if (gt == K_U8 || gt == K_U16 || gt == K_U32 || gt == K_U64
                 || gt == K_S8 || gt == K_S16 || gt == K_S32 || gt == K_S64
                 || gt == K_F32 || gt == K_F64
                 || gt == K_B8 || gt == K_B16 || gt == K_B32 || gt == K_B64)
            ttype = t;
        else if (gt == K_CA || gt == K_CG || gt == K_CS || gt == K_LU || gt == K_CV)
            cop = gt;
        else if (gt == K_V2 || gt == K_V4)
            vec = gt;
        else if (gt == K_VOLATILE)
            vol = gt;
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected LD instruction");
    }
    assert(ttype != 0);
    assert(cop == 0);
    int type = ttype->GetType();

    // Get two operands, assign source to destination.
    TREE * src = osrc->GetChild( 0);
    SYMBOL * ssrc = 0;
    
    assert(src->GetType() == T_WORD);
    ssrc = this->symbol_table->FindSymbol(src->GetText());
    assert(ssrc != 0);
    TREE * plus = osrc->GetChild( 1);
    CONSTANT value(0);
    if (plus != 0)
    {
        TREE * const_expr_tree = osrc->GetChild( 2);
        assert(const_expr_tree != 0);
        assert(const_expr_tree->GetType() == TREE_CONSTANT_EXPR);
        TREE * const_expr = const_expr_tree->GetChild(0);
        assert(const_expr != 0);
        value = this->device->Eval(K_S32, const_expr);
    }

    int times = 1;
    if (vec == K_V2)
        times = 2;
    else if (vec == K_V4)
        times = 4;

    // Different semantics for different storage classes.
    TYPES::Types * s = 0;
    unsigned char * addr = 0;
    switch (ssrc->storage_class)
    {
        case K_GLOBAL:
        case K_LOCAL:
        case K_PARAM:
        case K_SHARED:
        case K_CONST:
            {
                addr = (unsigned char*)ssrc->pvalue;
            }
            break;
        case K_REG:
            {
                addr = *(unsigned char**)ssrc->pvalue;
            }
            break;
        default:
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected LD instruction");
    }
    switch (value.type)
    {
        case K_U8:
            addr = (addr + value.value.s32);
            break;
        case K_U16:
            addr = (addr + value.value.s32);
            break;
        case K_U32:
            addr = (addr + value.value.s32);
            break;
        case K_U64:
            addr = (addr + value.value.s32);
            break;
        case K_S8:
            addr = (addr + value.value.s32);
            break;
        case K_S16:
            addr = (addr + value.value.s32);
            break;
        case K_S32:
            addr = (addr + value.value.s32);
            break;
        case K_S64:
            addr = (addr + value.value.s32);
            break;
        case K_B8:
            addr = (addr + value.value.s32);
            break;
        case K_B16:
            addr = (addr + value.value.s32);
            break;
        case K_B32:
            addr = (addr + value.value.s32);
            break;
        case K_B64:
            addr = (addr + value.value.s32);
            break;
        case K_F32:
            addr = (addr + value.value.s32);
            break;
        case K_F64:
            addr = (addr + value.value.s32);
            break;
        default:
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected LD instruction");
    }

    for (int i = 0; i < times; ++i)
    {
        TREE * dst = odst->GetChild( i);
        SYMBOL * sdst = 0;
        if (dst->GetType() == T_UNDERSCORE)
        {
            // ignore destination store--bit bucket.
        }
        else
        {
            assert(dst->GetType() == T_WORD);
            sdst = this->symbol_table->FindSymbol(dst->GetText());
            assert(sdst != 0);
            TYPES::Types * d = (TYPES::Types*)sdst->pvalue;

            TYPES::Types * s = (TYPES::Types*)addr;

            // For targets that are register, we may widen to the size of the register.
            if (sdst->storage_class == K_REG)
            {
                switch (sdst->type)
                {
                    case K_U8:
                        switch (type)
                        {
                            case K_U8:
                                d->u8 = s->u8;
                                break;
                            case K_B8:
                                d->u8 = s->b8;
                                break;
                            case K_S8:
                                d->u8 = s->s8;
                                break;
                            default:
                                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected LD instruction");
                        }
                        break;
                    case K_U16:
                        switch (type)
                        {
                            case K_U8:
                                d->u16 = s->u8;
                                break;
                            case K_B8:
                                d->u16 = s->b8;
                                break;
                            case K_S8:
                                d->u16 = s->s8;
                                break;
                            case K_U16:
                                d->u16 = s->u16;
                                break;
                            case K_B16:
                                d->u16 = s->b16;
                                break;
                            case K_S16:
                                d->u16 = s->s16;
                                break;
                            default:
                                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected LD instruction");
                        }
                        break;
                    case K_U32:
                        switch (type)
                        {
                            case K_U8:
                                d->u32 = s->u8;
                                break;
                            case K_B8:
                                d->u32 = s->b8;
                                break;
                            case K_S8:
                                d->u32 = s->s8;
                                break;
                            case K_U16:
                                d->u32 = s->u16;
                                break;
                            case K_B16:
                                d->u32 = s->b16;
                                break;
                            case K_S16:
                                d->u32 = s->s16;
                                break;
                            case K_U32:
                                d->u32 = s->u32;
                                break;
                            case K_B32:
                                d->u32 = s->b32;
                                break;
                            case K_S32:
                                d->u32 = s->s32;
                                break;
                            default:
                                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected LD instruction");
                        }
                        break;
                    case K_U64:
                        switch (type)
                        {
                            case K_U8:
                                d->u64 = s->u8;
                                break;
                            case K_B8:
                                d->u64 = s->b8;
                                break;
                            case K_S8:
                                d->u64 = s->s8;
                                break;
                            case K_U16:
                                d->u64 = s->u16;
                                break;
                            case K_B16:
                                d->u64 = s->b16;
                                break;
                            case K_S16:
                                d->u64 = s->s16;
                                break;
                            case K_U32:
                                d->u64 = s->u32;
                                break;
                            case K_B32:
                                d->u64 = s->b32;
                                break;
                            case K_S32:
                                d->u64 = s->s32;
                                break;
                            case K_U64:
                                d->u64 = s->u64;
                                break;
                            case K_B64:
                                d->u64 = s->b64;
                                break;
                            case K_S64:
                                d->u64 = s->s64;
                                break;
                            default:
                                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected LD instruction");
                        }
                        break;
                    case K_S8:
                        switch (type)
                        {
                            case K_U8:
                                d->s8 = s->u8;
                                break;
                            case K_B8:
                                d->s8 = s->b8;
                                break;
                            case K_S8:
                                d->s8 = s->s8;
                                break;
                            default:
                                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected LD instruction");
                        }
                        break;
                    case K_S16:
                        switch (type)
                        {
                            case K_U8:
                                d->s16 = s->u8;
                                break;
                            case K_B8:
                                d->s16 = s->b8;
                                break;
                            case K_S8:
                                d->s16 = s->s8;
                                break;
                            case K_U16:
                                d->s16 = s->u16;
                                break;
                            case K_B16:
                                d->s16 = s->b16;
                                break;
                            case K_S16:
                                d->s16 = s->s16;
                                break;
                            default:
                                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected LD instruction");
                        }
                        break;
                    case K_S32:
                        switch (type)
                        {
                            case K_U8:
                                d->s32 = s->u8;
                                break;
                            case K_B8:
                                d->s32 = s->b8;
                                break;
                            case K_S8:
                                d->s32 = s->s8;
                                break;
                            case K_U16:
                                d->s32 = s->u16;
                                break;
                            case K_B16:
                                d->s32 = s->b16;
                                break;
                            case K_S16:
                                d->s32 = s->s16;
                                break;
                            case K_U32:
                                d->s32 = s->u32;
                                break;
                            case K_B32:
                                d->s32 = s->b32;
                                break;
                            case K_S32:
                                d->s32 = s->s32;
                                break;
                            default:
                                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected LD instruction");
                        }
                        break;
                    case K_S64:
                        switch (type)
                        {
                            case K_U8:
                                d->s64 = s->u8;
                                break;
                            case K_B8:
                                d->s64 = s->b8;
                                break;
                            case K_S8:
                                d->s64 = s->s8;
                                break;
                            case K_U16:
                                d->s64 = s->u16;
                                break;
                            case K_B16:
                                d->s64 = s->b16;
                                break;
                            case K_S16:
                                d->s64 = s->s16;
                                break;
                            case K_U32:
                                d->s64 = s->u32;
                                break;
                            case K_B32:
                                d->s64 = s->b32;
                                break;
                            case K_S32:
                                d->s64 = s->s32;
                                break;
                            case K_U64:
                                d->s64 = s->u64;
                                break;
                            case K_B64:
                                d->s64 = s->b64;
                                break;
                            case K_S64:
                                d->s64 = s->s64;
                                break;
                            default:
                                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected LD instruction");
                        }
                        break;
                    case K_B8:
                        switch (type)
                        {
                            case K_U8:
                                d->b8 = s->u8;
                                break;
                            case K_B8:
                                d->b8 = s->b8;
                                break;
                            case K_S8:
                                d->b8 = s->s8;
                                break;
                            default:
                                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected LD instruction");
                        }
                        break;
                    case K_B16:
                        switch (type)
                        {
                            case K_U8:
                                d->b16 = s->u8;
                                break;
                            case K_B8:
                                d->b16 = s->b8;
                                break;
                            case K_S8:
                                d->b16 = s->s8;
                                break;
                            case K_U16:
                                d->b16 = s->u16;
                                break;
                            case K_B16:
                                d->b16 = s->b16;
                                break;
                            case K_S16:
                                d->b16 = s->s16;
                                break;
                            default:
                                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected LD instruction");
                        }
                        break;
                    case K_B32:
                        switch (type)
                        {
                            case K_U8:
                                d->b32 = s->u8;
                                break;
                            case K_B8:
                                d->b32 = s->b8;
                                break;
                            case K_S8:
                                d->b32 = s->s8;
                                break;
                            case K_U16:
                                d->b32 = s->u16;
                                break;
                            case K_B16:
                                d->b32 = s->b16;
                                break;
                            case K_S16:
                                d->b32 = s->s16;
                                break;
                            case K_U32:
                                d->b32 = s->u32;
                                break;
                            case K_B32:
                                d->b32 = s->b32;
                                break;
                            case K_S32:
                                d->b32 = s->s32;
                                break;
                            default:
                                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected LD instruction");
                        }
                        break;
                    case K_B64:
                        switch (type)
                        {
                            case K_U8:
                                d->b64 = s->u8;
                                break;
                            case K_B8:
                                d->b64 = s->b8;
                                break;
                            case K_S8:
                                d->b64 = s->s8;
                                break;
                            case K_U16:
                                d->b64 = s->u16;
                                break;
                            case K_B16:
                                d->b64 = s->b16;
                                break;
                            case K_S16:
                                d->b64 = s->s16;
                                break;
                            case K_U32:
                                d->b64 = s->u32;
                                break;
                            case K_B32:
                                d->b64 = s->b32;
                                break;
                            case K_S32:
                                d->b64 = s->s32;
                                break;
                            case K_U64:
                                d->b64 = s->u64;
                                break;
                            case K_B64:
                                d->b64 = s->b64;
                                break;
                            case K_S64:
                                d->b64 = s->s64;
                                break;
                            default:
                                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected LD instruction");
                        }
                        break;
                    case K_F32:
                        d->f32 = s->f32;
                        break;
                    case K_F64:
                        d->f64 = s->f64;
                        break;
                    default:
                        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected LD instruction");
                }
            }
            else
            {
                switch (type)
                {
                    case K_U8:
                        d->u8 = s->u8;
                        break;
                    case K_U16:
                        d->u16 = s->u16;
                        break;
                    case K_U32:
                        d->u32 = s->u32;
                        break;
                    case K_U64:
                        d->u64 = s->u64;
                        break;
                    case K_S8:
                        d->s8 = s->s8;
                        break;
                    case K_S16:
                        d->s16 = s->s16;
                        break;
                    case K_S32:
                        d->s32 = s->s32;
                        break;
                    case K_S64:
                        d->s64 = s->s64;
                        break;
                    case K_F32:
                        d->f32 = s->f32;
                        break;
                    case K_F64:
                        d->f64 = s->f64;
                        break;
                    default:
                        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected LD instruction");
                }
            }
        }

        addr = addr + this->device->Sizeof(type);
    }
    return 0;
}

int THREAD::DoLdu(TREE * inst)
{
    int start = 0;
    if (inst->GetChild(start)->GetType() == TREE_PRED)
        start++;
    assert(inst->GetChild(start)->GetType() == KI_LDU);
    start++;
    TREE * ttype = 0;
    TREE * odst = 0;
    TREE * osrc = 0;
    for (;; ++start)
    {
        TREE * t = inst->GetChild(start);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == TREE_TYPE)
            ttype = t;
        else if (gt == TREE_OPR)
        {
            if (odst == 0)
            {
                odst = t;
            } else if (osrc == 0)
            {
                osrc = t;
            }
            else
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected LDU instruction");
        }
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected LDU instruction");
    }
    assert(ttype != 0);
    assert(odst != 0);
    assert(osrc != 0);
    int ss = 0;
    int cop = 0;
    int vec = 0;
    for (int i = 0; ; ++i)
    {
        TREE * t = ttype->GetChild(i);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == K_CONST || gt == K_GLOBAL || gt == K_LOCAL || gt == K_PARAM || gt == K_SHARED)
            ss = gt;
        else if (gt == K_U8 || gt == K_U16 || gt == K_U32 || gt == K_U64
                 || gt == K_S8 || gt == K_S16 || gt == K_S32 || gt == K_S64
                 || gt == K_F32 || gt == K_F64
                 || gt == K_B8 || gt == K_B16 || gt == K_B32 || gt == K_B64)
            ttype = t;
        else if (gt == K_CA || gt == K_CG || gt == K_CS || gt == K_LU || gt == K_CV)
            cop = gt;
        else if (gt == K_V2 || gt == K_V4)
            vec = gt;
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected LDU instruction");
    }
    assert(ttype != 0);
    assert(vec == 0);
    assert(cop == 0);
    int type = ttype->GetType();

    // Get two operands, assign source to destination.
    TREE * dst = odst->GetChild( 0);
    TREE * src = osrc->GetChild( 0);
    SYMBOL * sdst = 0;
    SYMBOL * ssrc = 0;
    assert(dst->GetType() == T_WORD);
    sdst = this->symbol_table->FindSymbol(dst->GetText());
    
    assert(src->GetType() == T_WORD);
    ssrc = this->symbol_table->FindSymbol(src->GetText());
    TREE * plus = osrc->GetChild( 1);
    CONSTANT value(0);
    if (plus != 0)
    {
        TREE * const_expr_tree = osrc->GetChild( 2);
        assert(const_expr_tree != 0);
        assert(const_expr_tree->GetType() == TREE_CONSTANT_EXPR);
        TREE * const_expr = const_expr_tree->GetChild(0);
        assert(const_expr != 0);
        value = this->device->Eval(K_S32, const_expr);
    }

    TYPES::Types * d = (TYPES::Types*)sdst->pvalue;
    // Unfortunately, different semantics for different storage classes.
    TYPES::Types * s = 0;
    if (ssrc->storage_class != K_REG)
        s = (TYPES::Types*)ssrc->pvalue;
    else if (plus != 0)
    {
        void * addr = *(void**)ssrc->pvalue;
        switch (value.type)
        {
        case K_U8:
            s = (TYPES::Types*)(((unsigned char *)addr) + value.value.u8);
            break;
        case K_U16:
            s = (TYPES::Types*)(((unsigned char *)addr) + value.value.u16);
            break;
        case K_U32:
            s = (TYPES::Types*)(((unsigned char *)addr) + value.value.u32);
            break;
        case K_U64:
            s = (TYPES::Types*)(((unsigned char *)addr) + value.value.u64);
            break;
        case K_S8:
            s = (TYPES::Types*)(((unsigned char *)addr) + value.value.s8);
            break;
        case K_S16:
            s = (TYPES::Types*)(((unsigned char *)addr) + value.value.s16);
            break;
        case K_S32:
            s = (TYPES::Types*)(((unsigned char *)addr) + value.value.s32);
            break;
        case K_S64:
            s = (TYPES::Types*)(((unsigned char *)addr) + value.value.s64);
            break;
        default:
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected LDU instruction");
        }
    }
    else
        s = (TYPES::Types*)ssrc->pvalue;
    
    switch (type)
    {
        case K_U8:
            d->u8 = s->u8;
            break;
        case K_U16:
            d->u16 = s->u16;
            break;
        case K_U32:
            d->u32 = s->u32;
            break;
        case K_U64:
            d->u64 = s->u64;
            break;
        case K_S8:
            d->s8 = s->s8;
            break;
        case K_S16:
            d->s16 = s->s16;
            break;
        case K_S32:
            d->s32 = s->s32;
            break;
        case K_S64:
            d->s64 = s->s64;
            break;
        case K_F32:
            d->f32 = s->f32;
            break;
        case K_F64:
            d->f64 = s->f64;
            break;
        default:
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected LDU instruction");
    }
    return 0;
}

int THREAD::DoLg2(TREE * inst)
{
    int start = 0;
    if (inst->GetChild(start)->GetType() == TREE_PRED)
        start++;
    assert(inst->GetChild(start)->GetType() == KI_LG2);
    start++;
    TREE * ttype = 0;
    TREE * odst = 0;
    TREE * osrc1 = 0;
    for (;; ++start)
    {
        TREE * t = inst->GetChild(start);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == TREE_TYPE)
            ttype = t;
        else if (gt == TREE_OPR)
        {
            if (odst == 0)
            {
                odst = t;
            } else if (osrc1 == 0)
            {
                osrc1 = t;
            }
            else
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected LG2 instruction");
        }
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected LG2 instruction");
    }
    assert(ttype != 0);
    assert(odst != 0);
    assert(osrc1 != 0);
    bool ftz = false;
    int rnd = 0;
    for (int i = 0; ; ++i)
    {
        TREE * t = ttype->GetChild(i);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == K_F32 || gt == K_F64)
            ttype = t;
        else if (gt == K_FTZ)
            ftz = true;
        else if (gt == K_APPROX)
            ;
        else if (gt == K_RN || gt == K_RZ || gt == K_RM || gt == K_RP)
            rnd = gt;
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected LG2 instruction");
    }
    assert(ttype != 0);
    this->device->unimplemented(ftz, "LG2.ftz not implemented.");

    int type = ttype->GetType();
    TREE * dst = odst->GetChild(0);
    TREE * src1 = osrc1->GetChild(0);

    SYMBOL * sdst = 0;
    if (dst->GetType() == T_WORD)
    {
        sdst = this->symbol_table->FindSymbol(dst->GetText());
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected LG2 instruction");

    TYPES::Types value1;
    char * dummy;
    TYPES::Types * d = (TYPES::Types*)sdst->pvalue;
    TYPES::Types * s1 = &value1;

    if (src1->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src1->GetChild( 0));
        switch (type)
        {
            case K_F32:
                s1->f32 = c.value.f32;
                break;
            case K_F64:
                s1->f64 = c.value.f64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected LG2 instruction");
        }
    } else if (src1->GetType() == T_WORD)
    {
        SYMBOL * ssrc1 = this->symbol_table->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        assert(ssrc1->size == this->device->Sizeof(type));
        TYPES::Types * psrc1_value = (TYPES::Types*)ssrc1->pvalue;
        switch (type)
        {
            case K_F32:
                s1->f32 = psrc1_value->f32;
                break;
            case K_F64:
                s1->f64 = psrc1_value->f64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected LG2 instruction");
        }
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected LG2 instruction");

    switch (type)
    {
        case K_F32:
            d->f32 = log(s1->f32) / log(2.0);
            break;
        case K_F64:
            d->f64 = log(s1->f64) / log(2.0);
            break;
        default:
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected LG2 instruction");
    }
    return 0;
}

int THREAD::DoMad(TREE * inst)
{
    // Multiply+add register and/or constants, and store in a register.
    int start = 0;
    if (inst->GetChild(start)->GetType() == TREE_PRED)
    start++;
    assert(inst->GetChild(start)->GetType() == KI_MAD);
    start++;
    assert(inst->GetChild(start)->GetType() == TREE_TYPE);
    TREE * ttype = inst->GetChild(start);
    start++;
    TREE * odst = 0;
    TREE * osrc1 = 0;
    TREE * osrc2 = 0;
    TREE * osrc3 = 0;
    for (;; ++start)
    {
        TREE * t = inst->GetChild(start);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == TREE_OPR)
        {
            if (odst == 0)
            {
                odst = t;
            } else if (osrc1 == 0)
            {
                osrc1 = t;
            } else if (osrc2 == 0)
            {
                osrc2 = t;
            } else if (osrc3 == 0)
            {
                osrc3 = t;
            } else assert(false);
        } else assert(false);
    }
    assert(ttype != 0);
    assert(odst != 0);
    assert(osrc1 != 0);
    assert(osrc2 != 0);
    assert(osrc3 != 0);
    bool sat = false;
    bool ftz = false;
    TREE * twidth = 0;
    TREE * tfrnd = 0;
    for (int i = 0; ; ++i)
    {
        TREE * t = ttype->GetChild(i);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == K_SAT)
            sat = true;
        else if (gt == K_FTZ)
            ftz = true;
        else if (gt == K_RN || gt == K_RZ || gt == K_RM || gt == K_RP)
            tfrnd = t;
        else if (gt == K_F32 || gt == K_F64)
            ttype = t;
        else if (gt== K_LO || gt == K_HI || gt == K_WIDE)
            twidth = t;
        else if (gt == K_U16 || gt == K_U32 || gt == K_U64
                 || gt == K_S16 || gt == K_S32 || gt == K_S64)
            ttype = t;
        else assert(false);
    }
    assert(ttype != 0);
    assert(sat == 0); // unimplemented
    assert(ftz == 0);  // unimplemented.
    int type = ttype->GetType();
    int width = 0;
    if (twidth != 0)
        width = twidth->GetType();

    TREE * dst = odst->GetChild(0);
    TREE * src1 = osrc1->GetChild(0);
    TREE * src2 = osrc2->GetChild(0);
    TREE * src3 = osrc3->GetChild(0);

    SYMBOL * sdst = 0;
    SYMBOL * ssrc1 = 0;
    SYMBOL * ssrc2 = 0;
    SYMBOL * ssrc3 = 0;
    assert(dst->GetType() == T_WORD);
    sdst = this->symbol_table->FindSymbol(dst->GetText());
    char * dummy;

    TYPES::Types value1; // used if literal
    TYPES::Types value2; // used if literal
    TYPES::Types value3; // used if literal
    TYPES::Types * d = (TYPES::Types*)sdst->pvalue;
    TYPES::Types * s1 = &value1;
    TYPES::Types * s2 = &value2;
    TYPES::Types * s3 = &value2;

    if (src1->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src1->GetChild( 0));
        switch (type)
        {
            case K_U16:
                s1->u16 = c.value.u16;
                break;
            case K_S16:
                s1->s16 = c.value.s16;
                break;
            case K_U32:
                s1->u32 = c.value.u32;
                break;
            case K_S32:
                s1->s32 = c.value.s32;
                break;
            case K_F32:
                s1->f32 = c.value.f32;
                break;
            case K_F64:
                s1->f64 = c.value.f64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MAD instruction");
        }
    } else if (src1->GetType() == T_WORD)
    {
        ssrc1 = this->symbol_table->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        s1 = (TYPES::Types*)ssrc1->pvalue;
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MAD instruction");

    if (src2->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src2->GetChild(0));
        switch (type)
        {
            case K_U16:
                s2->u16 = c.value.u16;
                break;
            case K_S16:
                s2->s16 = c.value.s16;
                break;
            case K_U32:
                s2->u32 = c.value.u32;
                break;
            case K_S32:
                s2->s32 = c.value.s32;
                break;
            case K_F32:
                s2->f32 = c.value.f32;
                break;
            case K_F64:
                s2->f64 = c.value.f64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MAD instruction");
        }
    } else if (src2->GetType() == T_WORD)
    {
        ssrc2 = this->symbol_table->FindSymbol(src2->GetText());
        assert(ssrc2 != 0);
        s2 = (TYPES::Types*)ssrc2->pvalue;
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MAD instruction");

    if (src3->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src3->GetChild(0));
        switch (type)
        {
            case K_U16:
                s3->u16 = c.value.u16;
                break;
            case K_S16:
                s3->s16 = c.value.s16;
                break;
            case K_U32:
                s3->u32 = c.value.u32;
                break;
            case K_S32:
                s3->s32 = c.value.s32;
                break;
            case K_F32:
                s3->f32 = c.value.f32;
                break;
            case K_F64:
                s3->f64 = c.value.f64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MAD instruction");
        }
    } else if (src3->GetType() == T_WORD)
    {
        ssrc3 = this->symbol_table->FindSymbol(src3->GetText());
        assert(ssrc3 != 0);
        s3 = (TYPES::Types*)ssrc3->pvalue;
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MAD instruction");

    switch (type)
    {
        case K_U16:
            if (width == K_LO || width == 0)
            {
                unsigned __int32 vx = s1->u16;
                vx = vx * s2->u16;
                vx = vx + s3->u16;
                d->u16 = vx;
            }
            else if (width == K_HI)
            {
                unsigned __int32 vx = s1->u16;
                vx = vx * s2->u16;
                vx = vx + s3->u16;
                d->u16 = vx >> 16;
            }
            else if (width == K_WIDE)
            {
                unsigned __int32 vx = s1->u16;
                vx = vx * s2->u16;
                vx = vx + s3->u32;
                d->u32 = vx;
            }
            else
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MAD instruction");
            break;
        case K_S16:
            if (width == K_LO || width == 0)
            {
                __int32 vx = s1->s16;
                vx = vx * s2->s16;
                vx = vx + s3->s16;
                d->s16 = vx;
            }
            else if (width == K_HI)
            {
                __int32 vx = s1->s16;
                vx = vx * s2->s16;
                vx = vx + s3->s16;
                d->s16 = vx >> 16;
            }
            else if (width == K_WIDE)
            {
                __int32 vx = s1->s16;
                vx = vx * s2->s16;
                vx = vx + s3->s32;
                d->s32 = vx;
            }
            else
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MAD instruction");
            break;
        case K_U32:
            if (width == K_LO || width == 0)
            {
                unsigned __int64 vx = s1->u32;
                vx = vx * s2->u32;
                vx = vx + s3->u32;
                d->u32 = (unsigned __int32)(0xffffffff & vx);
            }
            else if (width == K_HI)
            {
                unsigned __int64 vx = s1->u32;
                vx = vx * s2->u32;
                vx = vx + s3->u32;
                d->u32 = vx >> 32;
            }
            else if (width == K_WIDE)
            {
                unsigned __int64 vx = s1->u32;
                vx = vx * s2->u32;
                vx = vx + s3->u64;
                d->u64 = vx;
            }
            else
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MAD instruction");
            break;
        case K_S32:
            if (width == K_LO || width == 0)
            {
                __int64 vx = s1->s32;
                vx = vx * s2->s32;
                vx = vx + s3->s32;
                d->s32 = (__int32)(0xffffffff & vx);
            }
            else if (width == K_HI)
            {
                __int64 vx = s1->s32;
                vx = vx * s2->s32;
                vx = vx + s3->s32;
                d->s32 = vx >> 32;
            }
            else if (width == K_WIDE)
            {
                __int64 vx = s1->s32;
                vx = vx * s2->s32;
                vx = vx + s3->s64;
                d->s64 = vx;
            }
            else
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MAD instruction");
            break;
        case K_F32:
            if (width == 0)
            {
                double vx = s1->f32;
                vx = vx * s2->f32;
                vx = vx + s3->f32;
                d->f32 = (float)(vx);
            }
            else if (width == K_WIDE)
            {
                double vx = s1->f32;
                vx = vx * s2->f32;
                vx = vx + s3->f64;
                d->f64 = vx;
            }
            else
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MAD instruction");
            break;
        default:
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MAD instruction");
    }
    return 0;
}

int THREAD::DoMad24(TREE * inst)
{
    throw new EMULATED_DEVICE::EMU_ERROR("MAD24 unimplemented");
}

int THREAD::DoMax(TREE * inst)
{
    int start = 0;
    if (inst->GetChild(start)->GetType() == TREE_PRED)
        start++;
    assert(inst->GetChild(start)->GetType() == KI_MAX);
    start++;
    TREE * ttype = 0;
    TREE * odst = 0;
    TREE * osrc1 = 0;
    TREE * osrc2 = 0;
    for (;; ++start)
    {
        TREE * t = inst->GetChild(start);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == TREE_TYPE)
            ttype = t;
        else if (gt == TREE_OPR)
        {
            if (odst == 0)
            {
                odst = t;
            } else if (osrc1 == 0)
            {
                osrc1 = t;
            } else if (osrc2 == 0)
            {
                osrc2 = t;
            }
            else
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MAX instruction");
        }
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MAX instruction");
    }
    assert(ttype != 0);
    assert(odst != 0);
    assert(osrc1 != 0);
    assert(osrc2 != 0);
    bool ftz = false;
    for (int i = 0; ; ++i)
    {
        TREE * t = ttype->GetChild(i);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == K_U16 || gt == K_U32 || gt == K_U64
                 || gt == K_S16 || gt == K_S32 || gt == K_S64)
            ttype = t;
        else if (gt == K_F32 || gt == K_F64)
            ttype = t;
        else if (gt == K_FTZ)
            ftz = true;
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MAX instruction");
    }
    assert(ttype != 0);
    this->device->unimplemented(ftz, "MAX.ftz not implemented.");
    int type = ttype->GetType();
    TREE * dst = odst->GetChild(0);
    TREE * src1 = osrc1->GetChild(0);
    TREE * src2 = osrc2->GetChild(0);

    SYMBOL * sdst = 0;
    if (dst->GetType() == T_WORD)
    {
        sdst = this->symbol_table->FindSymbol(dst->GetText());
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MAX instruction");

    TYPES::Types value1;
    TYPES::Types value2;
    TYPES::Types value3;
    char * dummy;
    TYPES::Types * d = (TYPES::Types*)sdst->pvalue;
    TYPES::Types * s1 = &value1;
    TYPES::Types * s2 = &value2;
    // used for carry out calculation.
    TYPES::Types * temp = &value3;

    if (src1->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src1->GetChild(0));
        switch (type)
        {
            case K_U16:
                s1->u16 = c.value.u16;
                break;
            case K_S16:
                s1->s16 = c.value.s16;
                break;
            case K_U32:
                s1->u32 = c.value.u32;
                break;
            case K_S32:
                s1->s32 = c.value.s32;
                break;
            case K_U64:
                s1->u64 = c.value.u64;
                break;
            case K_S64:
                s1->s64 = c.value.s64;
                break;
            case K_F32:
                s1->f32 = c.value.f32;
                break;
            case K_F64:
                s1->f64 = c.value.f64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MAX instruction");
        }
    } else if (src1->GetType() == T_WORD)
    {
        SYMBOL * ssrc1 = this->symbol_table->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        assert(ssrc1->size == this->device->Sizeof(type));
        TYPES::Types * psrc1_value = (TYPES::Types*)ssrc1->pvalue;
        switch (type)
        {
            case K_U16:
                s1->u16 = psrc1_value->u16;
                break;
            case K_S16:
                s1->s16 = psrc1_value->s16;
                break;
            case K_U32:
                s1->u32 = psrc1_value->u32;
                break;
            case K_S32:
                s1->s32 = psrc1_value->s32;
                break;
            case K_U64:
                s1->u64 = psrc1_value->u64;
                break;
            case K_S64:
                s1->s64 = psrc1_value->s64;
                break;
            case K_F32:
                s1->f32 = psrc1_value->f32;
                break;
            case K_F64:
                s1->f64 = psrc1_value->f64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MAX instruction");
        }
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MAX instruction");

    if (src2->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src2->GetChild(0));
        switch (type)
        {
            case K_U16:
                s2->u16 = c.value.u16;
                break;
            case K_S16:
                s2->s16 = c.value.s16;
                break;
            case K_U32:
                s2->u32 = c.value.u32;
                break;
            case K_S32:
                s2->s32 = c.value.s32;
                break;
            case K_U64:
                s2->u64 = c.value.u64;
                break;
            case K_S64:
                s2->s64 = c.value.s64;
                break;
            case K_F32:
                s1->f32 = c.value.f32;
                break;
            case K_F64:
                s1->f64 = c.value.f64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MAX instruction");
        }
    } else if (src2->GetType() == T_WORD)
    {
        SYMBOL * ssrc2 = this->symbol_table->FindSymbol(src2->GetText());
        assert(ssrc2 != 0);
        assert(ssrc2->size == this->device->Sizeof(type));
        TYPES::Types * psrc2_value = (TYPES::Types*)ssrc2->pvalue;
        switch (type)
        {
            case K_U16:
                s2->u16 = psrc2_value->u16;
                break;
            case K_S16:
                s2->s16 = psrc2_value->s16;
                break;
            case K_U32:
                s2->u32 = psrc2_value->u32;
                break;
            case K_S32:
                s2->s32 = psrc2_value->s32;
                break;
            case K_U64:
                s2->u64 = psrc2_value->u64;
                break;
            case K_S64:
                s2->s64 = psrc2_value->s64;
                break;
            case K_F32:
                s2->f32 = psrc2_value->f32;
                break;
            case K_F64:
                s2->f64 = psrc2_value->f64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MAX instruction");
        }
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MAX instruction");

    switch (type)
    {
        case K_U16:
            d->u16 = s1->u16 > s2->u16 ? s1->u16 : s2->u16;
            break;
        case K_S16:
            d->s16 = s1->s16 > s2->s16 ? s1->s16 : s2->s16;
            break;
        case K_U32:
            d->u32 = s1->u32 > s2->u32 ? s1->u32 : s2->u32;
            break;
        case K_S32:
            d->s32 = s1->s32 > s2->s32 ? s1->s32 : s2->s32;
            break;
        case K_U64:
            d->u64 = s1->u64 > s2->u64 ? s1->u64 : s2->u64;
            break;
        case K_S64:
            d->s64 = s1->s64 > s2->s64 ? s1->s64 : s2->s64;
            break;
        case K_F32:
            if (_isnan(s1->f32))
                d->f32 = s2->f32;
            else if (_isnan(s2->f32))
                d->f32 = s1->f32;
            else
                d->f32 = s1->f32 > s2->f32 ? s1->f32 : s2->f32;
            break;
        case K_F64:
            if (_isnan(s1->f64))
                d->f64 = s2->f64;
            else if (_isnan(s2->f64))
                d->f64 = s1->f64;
            else
                d->f64 = s1->f64 > s2->f64 ? s1->f64 : s2->f64;
            break;
        default:
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MAX instruction");
    }
    return 0;
}

int THREAD::DoMembar(TREE * inst)
{
    throw new EMULATED_DEVICE::EMU_ERROR("MEMBAR unimplemented");
}

int THREAD::DoMin(TREE * inst)
{
    int start = 0;
    if (inst->GetChild(start)->GetType() == TREE_PRED)
        start++;
    assert(inst->GetChild(start)->GetType() == KI_MIN);
    start++;
    TREE * ttype = 0;
    TREE * odst = 0;
    TREE * osrc1 = 0;
    TREE * osrc2 = 0;
    for (;; ++start)
    {
        TREE * t = inst->GetChild(start);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == TREE_TYPE)
            ttype = t;
        else if (gt == TREE_OPR)
        {
            if (odst == 0)
            {
                odst = t;
            } else if (osrc1 == 0)
            {
                osrc1 = t;
            } else if (osrc2 == 0)
            {
                osrc2 = t;
            }
            else
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MIN instruction");
        }
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MIN instruction");
    }
    assert(ttype != 0);
    assert(odst != 0);
    assert(osrc1 != 0);
    assert(osrc2 != 0);
    bool ftz = false;
    for (int i = 0; ; ++i)
    {
        TREE * t = ttype->GetChild(i);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == K_U16 || gt == K_U32 || gt == K_U64
                 || gt == K_S16 || gt == K_S32 || gt == K_S64)
            ttype = t;
        else if (gt == K_F32 || gt == K_F64)
            ttype = t;
        else if (gt == K_FTZ)
            ftz = true;
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MIN instruction");
    }
    assert(ttype != 0);
    this->device->unimplemented(ftz, "MAX.ftz not implemented.");
    int type = ttype->GetType();
    TREE * dst = odst->GetChild(0);
    TREE * src1 = osrc1->GetChild(0);
    TREE * src2 = osrc2->GetChild(0);

    SYMBOL * sdst = 0;
    if (dst->GetType() == T_WORD)
    {
        sdst = this->symbol_table->FindSymbol(dst->GetText());
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MIN instruction");

    TYPES::Types value1;
    TYPES::Types value2;
    TYPES::Types value3;
    char * dummy;
    TYPES::Types * d = (TYPES::Types*)sdst->pvalue;
    TYPES::Types * s1 = &value1;
    TYPES::Types * s2 = &value2;
    // used for carry out calculation.
    TYPES::Types * temp = &value3;

    if (src1->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src1->GetChild(0));
        switch (type)
        {
            case K_U16:
                s1->u16 = c.value.u16;
                break;
            case K_S16:
                s1->s16 = c.value.s16;
                break;
            case K_U32:
                s1->u32 = c.value.u32;
                break;
            case K_S32:
                s1->s32 = c.value.s32;
                break;
            case K_U64:
                s1->u64 = c.value.u64;
                break;
            case K_S64:
                s1->s64 = c.value.s64;
                break;
            case K_F32:
                s1->f32 = c.value.f32;
                break;
            case K_F64:
                s1->f64 = c.value.f64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MIN instruction");
        }
    } else if (src1->GetType() == T_WORD)
    {
        SYMBOL * ssrc1 = this->symbol_table->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        assert(ssrc1->size == this->device->Sizeof(type));
        TYPES::Types * psrc1_value = (TYPES::Types*)ssrc1->pvalue;
        switch (type)
        {
            case K_U16:
                s1->u16 = psrc1_value->u16;
                break;
            case K_S16:
                s1->s16 = psrc1_value->s16;
                break;
            case K_U32:
                s1->u32 = psrc1_value->u32;
                break;
            case K_S32:
                s1->s32 = psrc1_value->s32;
                break;
            case K_U64:
                s1->u64 = psrc1_value->u64;
                break;
            case K_S64:
                s1->s64 = psrc1_value->s64;
                break;
            case K_F32:
                s1->f32 = psrc1_value->f32;
                break;
            case K_F64:
                s1->f64 = psrc1_value->f64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MIN instruction");
        }
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MIN instruction");

    if (src2->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src2->GetChild(0));
        switch (type)
        {
            case K_U16:
                s2->u16 = c.value.u16;
                break;
            case K_S16:
                s2->s16 = c.value.s16;
                break;
            case K_U32:
                s2->u32 = c.value.u32;
                break;
            case K_S32:
                s2->s32 = c.value.s32;
                break;
            case K_U64:
                s2->u64 = c.value.u64;
                break;
            case K_S64:
                s2->s64 = c.value.s64;
                break;
            case K_F32:
                s1->f32 = c.value.f32;
                break;
            case K_F64:
                s1->f64 = c.value.f64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MIN instruction");
        }
    } else if (src2->GetType() == T_WORD)
    {
        SYMBOL * ssrc2 = this->symbol_table->FindSymbol(src2->GetText());
        assert(ssrc2 != 0);
        assert(ssrc2->size == this->device->Sizeof(type));
        TYPES::Types * psrc2_value = (TYPES::Types*)ssrc2->pvalue;
        switch (type)
        {
            case K_U16:
                s2->u16 = psrc2_value->u16;
                break;
            case K_S16:
                s2->s16 = psrc2_value->s16;
                break;
            case K_U32:
                s2->u32 = psrc2_value->u32;
                break;
            case K_S32:
                s2->s32 = psrc2_value->s32;
                break;
            case K_U64:
                s2->u64 = psrc2_value->u64;
                break;
            case K_S64:
                s2->s64 = psrc2_value->s64;
                break;
            case K_F32:
                s2->f32 = psrc2_value->f32;
                break;
            case K_F64:
                s2->f64 = psrc2_value->f64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MIN instruction");
        }
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MIN instruction");

    switch (type)
    {
        case K_U16:
            d->u16 = s1->u16 < s2->u16 ? s1->u16 : s2->u16;
            break;
        case K_S16:
            d->s16 = s1->s16 < s2->s16 ? s1->s16 : s2->s16;
            break;
        case K_U32:
            d->u32 = s1->u32 < s2->u32 ? s1->u32 : s2->u32;
            break;
        case K_S32:
            d->s32 = s1->s32 < s2->s32 ? s1->s32 : s2->s32;
            break;
        case K_U64:
            d->u64 = s1->u64 < s2->u64 ? s1->u64 : s2->u64;
            break;
        case K_S64:
            d->s64 = s1->s64 < s2->s64 ? s1->s64 : s2->s64;
            break;
        case K_F32:
            if (_isnan(s1->f32))
                d->f32 = s2->f32;
            else if (_isnan(s2->f32))
                d->f32 = s1->f32;
            else
                d->f32 = s1->f32 < s2->f32 ? s1->f32 : s2->f32;
            break;
        case K_F64:
            if (_isnan(s1->f64))
                d->f64 = s2->f64;
            else if (_isnan(s2->f64))
                d->f64 = s1->f64;
            else
                d->f64 = s1->f64 < s2->f64 ? s1->f64 : s2->f64;
            break;
        default:
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MIN instruction");
    }
    return 0;
}

int THREAD::DoMov(TREE * inst)
{
    // Assign source to destination.
    int start = 0;
    if (inst->GetChild(start)->GetType() == TREE_PRED)
        start++;
    assert(inst->GetChild(start)->GetType() == KI_MOV);
    start++;
    TREE * ttype = 0;
    TREE * odst = 0;
    TREE * osrc = 0;
    for (;; ++start)
    {
        TREE * t = inst->GetChild(start);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == TREE_TYPE)
            ttype = t;
        else if (gt == TREE_OPR)
        {
            if (odst == 0)
            {
                odst = t;
            } else if (osrc == 0)
            {
                osrc = t;
            }
            else
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MOV instruction");
        }
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MOV instruction");
    }
    assert(ttype != 0);
    assert(odst != 0);
    assert(osrc != 0);
    for (int i = 0; ; ++i)
    {
        TREE * t = ttype->GetChild(i);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == K_PRED
              || gt == K_B16 || gt == K_B32 || gt == K_B64
              || gt == K_U16 || gt == K_U32 || gt == K_U64
              || gt == K_S16 || gt == K_S32 || gt == K_S64
              || gt == K_F32 || gt == K_F64)
            ttype = t;
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MOV instruction");
    }
    assert(ttype);
    int type = ttype->GetType();
    TREE * dst = odst->GetChild( 0);
    TREE * src = osrc->GetChild(0);
    SYMBOL * sdst = 0;
    SYMBOL * ssrc = 0;
    if (dst->GetType() == T_WORD)
    {
        sdst = this->symbol_table->FindSymbol(dst->GetText());
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MOV instruction");

    TYPES::Types * d;
    TYPES::Types * s;
    TYPES::Types value;
    char * dummy;

    d = (TYPES::Types*)sdst->pvalue;
    s = &value;

    if (src->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src->GetChild(0));
        switch (type)
        {
            case K_B16:
                s->b16 = c.value.b16;
                break;
            case K_U16:
                s->u16 = c.value.u16;
                break;
            case K_S16:
                s->s16 = c.value.s16;
                break;
            case K_B32:
                s->b32 = c.value.b32;
                break;
            case K_U32:
                s->u32 = c.value.u32;
                break;
            case K_S32:
                s->s32 = c.value.s32;
                break;
            case K_B64:
                s->b64 = c.value.b64;
                break;
            case K_U64:
                s->u64 = c.value.u64;
                break;
            case K_S64:
                s->s64 = c.value.s64;
                break;
            case K_F32:
                s->f32 = c.value.f32;
                break;
            case K_F64:
                s->f64 = c.value.f64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MOV instruction");
        }
    } else if (src->GetType() == T_WORD)
    {
        ssrc = this->symbol_table->FindSymbol(src->GetText());
        assert(ssrc != 0);
        // Various types of id's to handle:
        switch (ssrc->storage_class)
        {
            case K_GLOBAL:
            case K_LOCAL:
            case K_PARAM:
            case K_SHARED:
            case K_CONST:
                // names in instructions refer to the address of the
                // variable, not the contents.
                s = (TYPES::Types*)&ssrc->pvalue;
                break;
            case K_REG:
                // names in instructions refer to the contents of the
                // register.
                s = (TYPES::Types*)ssrc->pvalue;
                break;
        }
        if (strcmp(ssrc->typestring, "dim3") == 0)
        {
            // Get qualifier of the structure.
            TREE * tqual = (TREE *)osrc->GetChild(1);
            assert(tqual != 0);
            int qual = tqual->GetType();
            if (qual == K_X)
            {
                s = (TYPES::Types*)& ((dim3*)ssrc->pvalue)->x;
            } else if (qual == K_Y)
            {
                s = (TYPES::Types*)& ((dim3*)ssrc->pvalue)->y;
            } else if (qual == K_Z)
            {
                s = (TYPES::Types*)& ((dim3*)ssrc->pvalue)->z;
            }
            else
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MOV instruction");
        }
    }

    switch (type)
    {
        case K_B16:
            d->b16 = s->b16;
            break;
        case K_B32:
            d->b32 = s->b32;
            break;
        case K_B64:
            d->b64 = s->b64;
            break;
        case K_U16:
            d->u16 = s->u16;
            break;
        case K_U32:
            d->u32 = s->u32;
            break;
        case K_U64:
            d->u64 = s->u64;
            break;
        case K_S16:
            d->s16 = s->s16;
            break;
        case K_S32:
            d->s32 = s->s32;
            break;
        case K_S64:
            d->s64 = s->s64;
            break;
        case K_F32:
            d->f32 = s->f32;
            break;
        case K_F64:
            d->f64 = s->f64;
            break;
        default:
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MOV instruction");
    }
    return 0;
}

int THREAD::DoMul(TREE * inst)
{
    // Multiply register and/or constants, and store in a register.
    int start = 0;
    if (inst->GetChild(start)->GetType() == TREE_PRED)
        start++;
    assert(inst->GetChild(start)->GetType() == KI_MUL);
    start++;
    assert(inst->GetChild(start)->GetType() == TREE_TYPE);
    TREE * ttype = inst->GetChild(start);
    start++;
    TREE * odst = 0;
    TREE * osrc1 = 0;
    TREE * osrc2 = 0;
    for (;; ++start)
    {
        TREE * t = inst->GetChild(start);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == TREE_OPR)
        {
            if (odst == 0)
            {
                odst = t;
            } else if (osrc1 == 0)
            {
                osrc1 = t;
            } else if (osrc2 == 0)
            {
                osrc2 = t;
            } else assert(false);
        } else assert(false);
    }
    assert(ttype != 0);
    assert(odst != 0);
    assert(osrc1 != 0);
    assert(osrc2 != 0);
    bool sat = false;
    bool ftz = false;
    TREE * twidth = 0;
    for (int i = 0; ; ++i)
    {
        TREE * t = ttype->GetChild(i);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == K_SAT)
            sat = true;
        else if (gt == K_FTZ)
            ftz = true;
        else if (gt == K_U16 || gt == K_U32 || gt == K_U64
                 || gt == K_S16 || gt == K_S32 || gt == K_S64)
            ttype = t;
        else if (gt == K_F32 || gt == K_F64)
            ttype = t;
        else if (gt== K_LO || gt == K_HI || gt == K_WIDE)
            twidth = t;
        else assert(false);
    }
    assert(ttype != 0);
    assert(sat == 0); // unimplemented
    assert(ftz == 0);  // unimplemented.
    int type = ttype->GetType();
    int width = 0;
    if (twidth != 0)
        width = twidth->GetType();

    TREE * dst = odst->GetChild(0);
    TREE * src1 = osrc1->GetChild(0);
    TREE * src2 = osrc2->GetChild(0);

    SYMBOL * sdst = 0;
    SYMBOL * ssrc1 = 0;
    SYMBOL * ssrc2 = 0;
    assert(dst->GetType() == T_WORD);
    sdst = this->symbol_table->FindSymbol(dst->GetText());
    char * dummy;

    TYPES::Types value1; // used if literal
    TYPES::Types value2; // used if literal
    TYPES::Types * d = (TYPES::Types*)sdst->pvalue;
    TYPES::Types * s1 = &value1;
    TYPES::Types * s2 = &value2;

    if (src1->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src1->GetChild( 0));
        switch (type)
        {
            case K_U16:
                s1->u16 = c.value.u16;
                break;
            case K_S16:
                s1->s16 = c.value.s16;
                break;
            case K_U32:
                s1->u32 = c.value.u32;
                break;
            case K_S32:
                s1->s32 = c.value.s32;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MUL instruction");
        }
    } else if (src1->GetType() == T_WORD)
    {
        ssrc1 = this->symbol_table->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        s1 = (TYPES::Types*)ssrc1->pvalue;
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MUL instruction");

    if (src2->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src2->GetChild(0));
        switch (type)
        {
            case K_U16:
                s2->u16 = c.value.u16;
                break;
            case K_S16:
                s2->s16 = c.value.s16;
                break;
            case K_U32:
                s2->u32 = c.value.u32;
                break;
            case K_S32:
                s2->s32 = c.value.s32;
                break;
            case K_F32:
                s2->f32 = c.value.f32;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MUL instruction");
        }
    } else if (src2->GetType() == T_WORD)
    {
        ssrc2 = this->symbol_table->FindSymbol(src2->GetText());
        assert(ssrc2 != 0);
        s2 = (TYPES::Types*)ssrc2->pvalue;
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MUL instruction");

    switch (type)
    {
        case K_U16:
            if (width == K_LO)
            {
                unsigned __int32 vx = s1->u16;
                vx = vx * s2->u16;
                d->u16 = vx;
            }
            else if (width == K_HI)
            {
                unsigned __int32 vx = s1->u16;
                vx = vx * s2->u16;
                d->u16 = vx >> 16;
            }
            else if (width == K_WIDE)
            {
                unsigned __int32 vx = s1->u16;
                vx = vx * s2->u16;
                d->u32 = vx;
            }
            else
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MUL instruction");
            break;
        case K_S16:
            if (width == K_LO)
            {
                __int32 vx = s1->s16;
                vx = vx * s2->s16;
                d->s16 = vx;
            }
            else if (width == K_HI)
            {
                __int32 vx = s1->s16;
                vx = vx * s2->s16;
                d->s16 = vx >> 16;
            }
            else if (width == K_WIDE)
            {
                __int32 vx = s1->s16;
                vx = vx * s2->s16;
                d->s32 = vx;
            }
            else
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MUL instruction");
            break;
        case K_U32:
            if (width == K_LO)
            {
                unsigned __int64 vx = s1->u32;
                vx = vx * s2->u32;
                d->u32 = (unsigned __int32)(0xffffffff & vx);
            }
            else if (width == K_HI)
            {
                unsigned __int64 vx = s1->u32;
                vx = vx * s2->u32;
                d->u32 = vx >> 32;
            }
            else if (width == K_WIDE)
            {
                unsigned __int64 vx = s1->u32;
                vx = vx * s2->u32;
                d->u64 = vx;
            }
            else
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MUL instruction");
            break;
        case K_S32:
            if (width == K_LO)
            {
                __int64 vx = s1->s32;
                vx = vx * s2->s32;
                d->s32 = (__int32)(0xffffffff & vx);
            }
            else if (width == K_HI)
            {
                __int64 vx = s1->s32;
                vx = vx * s2->s32;
                d->s32 = vx >> 32;
            }
            else if (width == K_WIDE)
            {
                __int64 vx = s1->s32;
                vx = vx * s2->s32;
                d->s64 = vx;
            }
            else
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MUL instruction");
            break;
        case K_F32:
            assert(width == 0);
            d->f32 = s1->f32 * s2->f32;
            break;
        case K_F64:
            assert(width == 0);
            d->f64 = s1->f64 * s2->f64;
            break;
        default:
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MUL instruction");
    }
    return 0;
}

int THREAD::DoMul24(TREE * inst)
{
    // Multiply 24-bit integer numbers, in register and/or constants,
    // and store in a register.
    int start = 0;
    if (inst->GetChild(start)->GetType() == TREE_PRED)
        start++;
    assert(inst->GetChild(start)->GetType() == KI_MUL24);
    start++;
    assert(inst->GetChild(start)->GetType() == TREE_TYPE);
    TREE * ttype = inst->GetChild(start);
    start++;
    TREE * odst = 0;
    TREE * osrc1 = 0;
    TREE * osrc2 = 0;
    for (;; ++start)
    {
        TREE * t = inst->GetChild(start);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == TREE_OPR)
        {
            if (odst == 0)
            {
                odst = t;
            } else if (osrc1 == 0)
            {
                osrc1 = t;
            } else if (osrc2 == 0)
            {
                osrc2 = t;
            }
            else
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MUL24 instruction");
        }
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MUL24 instruction");
    }
    assert(ttype != 0);
    assert(odst != 0);
    assert(osrc1 != 0);
    assert(osrc2 != 0);
    TREE * twidth = 0;
    for (int i = 0; ; ++i)
    {
        TREE * t = ttype->GetChild(i);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == K_U32 || gt == K_S32)
            ttype = t;
        else if (gt== K_LO || gt == K_HI)
            twidth = t;
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MUL24 instruction");
    }
    assert(ttype != 0);
    int type = ttype->GetType();
    int width = 0;
    if (twidth != 0)
        width = twidth->GetType();

    TREE * dst = odst->GetChild(0);
    TREE * src1 = osrc1->GetChild(0);
    TREE * src2 = osrc2->GetChild(0);

    SYMBOL * sdst = 0;
    SYMBOL * ssrc1 = 0;
    SYMBOL * ssrc2 = 0;
    assert(dst->GetType() == T_WORD);
    sdst = this->symbol_table->FindSymbol(dst->GetText());
    char * dummy;

    TYPES::Types value1; // used if literal
    TYPES::Types value2; // used if literal
    TYPES::Types * d = (TYPES::Types*)sdst->pvalue;
    TYPES::Types * s1 = &value1;
    TYPES::Types * s2 = &value2;

    if (src1->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src1->GetChild( 0));
        switch (type)
        {
            case K_U32:
                s1->u32 = c.value.u32;
                break;
            case K_S32:
                s1->s32 = c.value.s32;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MUL24 instruction");
        }
    } else if (src1->GetType() == T_WORD)
    {
        ssrc1 = this->symbol_table->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        s1 = (TYPES::Types*)ssrc1->pvalue;
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MUL24 instruction");

    if (src2->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src2->GetChild(0));
        switch (type)
        {
            case K_U32:
                s2->u32 = c.value.u32;
                break;
            case K_S32:
                s2->s32 = c.value.s32;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MUL24 instruction");
        }
    } else if (src2->GetType() == T_WORD)
    {
        ssrc2 = this->symbol_table->FindSymbol(src2->GetText());
        assert(ssrc2 != 0);
        s2 = (TYPES::Types*)ssrc2->pvalue;
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MUL24 instruction");

    switch (type)
    {
        case K_U32:
            if (width == K_LO)
                d->u32 = (s1->u32 * s2->u32) & 0xffffffff;
            else if (width == K_HI)
                d->u32 = (s1->u32 * s2->u32 ) >> 16;
            else
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MUL24 instruction");
            break;
        case K_S32:
            if (width == K_LO)
                d->s32 = (s1->s32 * s2->s32) & 0xffffffff;
            else if (width == K_HI)
                d->s32 = (s1->s32 * s2->s32 ) >> 16;
            else
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MUL24 instruction");
            break;
        default:
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected MUL24 instruction");
    }
    return 0;
}

int THREAD::DoNeg(TREE * inst)
{
    int start = 0;
    if (inst->GetChild(start)->GetType() == TREE_PRED)
        start++;
    assert(inst->GetChild(start)->GetType() == KI_NEG);
    start++;
    TREE * ttype = 0;
    TREE * odst = 0;
    TREE * osrc1 = 0;
    for (;; ++start)
    {
        TREE * t = inst->GetChild(start);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == TREE_TYPE)
            ttype = t;
        else if (gt == TREE_OPR)
        {
            if (odst == 0)
            {
                odst = t;
            } else if (osrc1 == 0)
            {
                osrc1 = t;
            }
            else
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected NEG instruction");
        }
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected NEG instruction");
    }
    assert(ttype != 0);
    assert(odst != 0);
    assert(osrc1 != 0);
    bool ftz = false;
    for (int i = 0; ; ++i)
    {
        TREE * t = ttype->GetChild(i);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == K_S16 || gt == K_S32 || gt == K_S64)
            ttype = t;
        else if (gt == K_F32 || gt == K_F64)
            ttype = t;
        else if (gt == K_FTZ)
            ftz = true;
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected NEG instruction");
    }
    assert(ttype != 0);
    this->device->unimplemented(ftz, "NEG.ftz not implemented.");

    int type = ttype->GetType();
    TREE * dst = odst->GetChild(0);
    TREE * src1 = osrc1->GetChild(0);

    SYMBOL * sdst = 0;
    if (dst->GetType() == T_WORD)
    {
        sdst = this->symbol_table->FindSymbol(dst->GetText());
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected NEG instruction");

    TYPES::Types value1;
    char * dummy;
    TYPES::Types * d = (TYPES::Types*)sdst->pvalue;
    TYPES::Types * s1 = &value1;

    if (src1->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src1->GetChild( 0));
        switch (type)
        {
            case K_S16:
                s1->s16 = c.value.s16;
                break;
            case K_S32:
                s1->s32 = c.value.s32;
                break;
            case K_S64:
                s1->s64 = c.value.s64;
                break;
            case K_F32:
                s1->f32 = c.value.f32;
                break;
            case K_F64:
                s1->f64 = c.value.f64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected NEG instruction");
        }
    } else if (src1->GetType() == T_WORD)
    {
        SYMBOL * ssrc1 = this->symbol_table->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        assert(ssrc1->size == this->device->Sizeof(type));
        TYPES::Types * psrc1_value = (TYPES::Types*)ssrc1->pvalue;
        switch (type)
        {
            case K_S16:
                s1->s16 = psrc1_value->s16;
                break;
            case K_S32:
                s1->s32 = psrc1_value->s32;
                break;
            case K_S64:
                s1->s64 = psrc1_value->s64;
                break;
            case K_F32:
                s1->f32 = psrc1_value->f32;
                break;
            case K_F64:
                s1->f64 = psrc1_value->f64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected NEG instruction");
        }
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected NEG instruction");

    switch (type)
    {
        case K_S16:
            d->s16 = - s1->s16;
            break;
        case K_S32:
            d->s32 = - s1->s32;
            break;
        case K_S64:
            d->s64 = - s1->s64;
            break;
        case K_F32:
            d->f32 = - s1->f32;
            break;
        case K_F64:
            d->f64 = - s1->f64;
            break;
        default:
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected NEG instruction");
    }
    return 0;
}

int THREAD::DoNot(TREE * inst)
{
    int start = 0;
    if (inst->GetChild(start)->GetType() == TREE_PRED)
        start++;
    assert(inst->GetChild(start)->GetType() == KI_NOT);
    start++;
    TREE * ttype = 0;
    TREE * odst = 0;
    TREE * osrc1 = 0;
    for (;; ++start)
    {
        TREE * t = inst->GetChild(start);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == TREE_TYPE)
            ttype = t;
        else if (gt == TREE_OPR)
        {
            if (odst == 0)
            {
                odst = t;
            } else if (osrc1 == 0)
            {
                osrc1 = t;
            }
            else
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected NOT instruction");
        }
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected NOT instruction");
    }
    assert(ttype != 0);
    assert(odst != 0);
    assert(osrc1 != 0);
    int type = 0;
    for (int i = 0; ; ++i)
    {
        TREE * t = ttype->GetChild(i);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == K_PRED || gt == K_B16 || gt == K_B32 || gt == K_B64)
            type = gt;
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected NOT instruction");
    }
    assert(type != 0);
    TREE * dst = odst->GetChild(0);
    TREE * src1 = osrc1->GetChild(0);

    SYMBOL * sdst = 0;
    if (dst->GetType() == T_WORD)
    {
        sdst = this->symbol_table->FindSymbol(dst->GetText());
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected NOT instruction");

    TYPES::Types value1;
    char * dummy;
    TYPES::Types * d = (TYPES::Types*)sdst->pvalue;
    TYPES::Types * s1 = &value1;

    if (src1->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src1->GetChild( 0));
        switch (type)
        {
            case K_B16:
                s1->b16 = c.value.b16;
                break;
            case K_B32:
                s1->b32 = c.value.b32;
                break;
            case K_B64:
                s1->b64 = c.value.b64;
                break;
            case K_PRED:
                s1->pred = c.value.pred;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected NOT instruction");
        }
    } else if (src1->GetType() == T_WORD)
    {
        SYMBOL * ssrc1 = this->symbol_table->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        assert(ssrc1->size == this->device->Sizeof(type));
        TYPES::Types * psrc1_value = (TYPES::Types*)ssrc1->pvalue;
        switch (type)
        {
            case K_B16:
                s1->b16 = psrc1_value->b16;
                break;
            case K_B32:
                s1->b32 = psrc1_value->b32;
                break;
            case K_B64:
                s1->b64 = psrc1_value->b64;
                break;
            case K_PRED:
                s1->pred = psrc1_value->pred;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected NOT instruction");
        }
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected NOT instruction");

    switch (type)
    {
        case K_B16:
            d->b16 = ~s1->b16;
            break;
        case K_B32:
            d->b32 = ~s1->b32;
            break;
        case K_S64:
            d->b64 = ~s1->b64;
            break;
        case K_PRED:
            d->pred = ! s1->pred; // THIS WORKS, BUT THE FOLLOWING DOES NOT WORK!!! ~s1->pred;
            break;
        default:
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected NOT instruction");
    }
    return 0;
}

int THREAD::DoOr(TREE * inst)
{
    int start = 0;
    if (inst->GetChild(start)->GetType() == TREE_PRED)
        start++;
    assert(inst->GetChild(start)->GetType() == KI_OR);
    start++;
    TREE * ttype = 0;
    TREE * odst = 0;
    TREE * osrc1 = 0;
    TREE * osrc2 = 0;
    for (;; ++start)
    {
        TREE * t = inst->GetChild(start);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == TREE_TYPE)
            ttype = t;
        else if (gt == TREE_OPR)
        {
            if (odst == 0)
            {
                odst = t;
            } else if (osrc1 == 0)
            {
                osrc1 = t;
            } else if (osrc2 == 0)
            {
                osrc2 = t;
            }
            else
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected OR instruction");
        }
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected OR instruction");
    }
    assert(ttype != 0);
    assert(odst != 0);
    assert(osrc1 != 0);
    assert(osrc2 != 0);
    int type = 0;
    for (int i = 0; ; ++i)
    {
        TREE * t = ttype->GetChild(i);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == K_PRED || gt == K_B16 || gt == K_B32 || gt == K_B64)
            type = gt;
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected OR instruction");
    }
    assert(type != 0);
    TREE * dst = odst->GetChild(0);
    TREE * src1 = osrc1->GetChild(0);
    TREE * src2 = osrc2->GetChild(0);

    SYMBOL * sdst = 0;
    if (dst->GetType() == T_WORD)
    {
        sdst = this->symbol_table->FindSymbol(dst->GetText());
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected OR instruction");

    TYPES::Types value1;
    TYPES::Types value2;
    char * dummy;
    TYPES::Types * d = (TYPES::Types*)sdst->pvalue;
    TYPES::Types * s1 = &value1;
    TYPES::Types * s2 = &value2;

    if (src1->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src1->GetChild( 0));
        switch (type)
        {
            case K_B16:
                s1->b16 = c.value.b16;
                break;
            case K_B32:
                s1->b32 = c.value.b32;
                break;
            case K_B64:
                s1->b64 = c.value.b64;
                break;
            case K_PRED:
                s1->pred = c.value.pred;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected OR instruction");
        }
    } else if (src1->GetType() == T_WORD)
    {
        SYMBOL * ssrc1 = this->symbol_table->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        assert(ssrc1->size == this->device->Sizeof(type));
        TYPES::Types * psrc1_value = (TYPES::Types*)ssrc1->pvalue;
        switch (type)
        {
            case K_B16:
                s1->b16 = psrc1_value->b16;
                break;
            case K_B32:
                s1->b32 = psrc1_value->b32;
                break;
            case K_B64:
                s1->b64 = psrc1_value->b64;
                break;
            case K_PRED:
                s1->pred = psrc1_value->pred;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected OR instruction");
        }
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected OR instruction");

    if (src2->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src2->GetChild(0));
        switch (type)
        {
            case K_B16:
                s2->b16 = c.value.b16;
                break;
            case K_B32:
                s2->b32 = c.value.b32;
                break;
            case K_B64:
                s2->b64 = c.value.b64;
                break;
            case K_PRED:
                s2->pred = c.value.pred;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected OR instruction");
        }
    } else if (src2->GetType() == T_WORD)
    {
        SYMBOL * ssrc2 = this->symbol_table->FindSymbol(src2->GetText());
        assert(ssrc2 != 0);
        assert(ssrc2->size == this->device->Sizeof(type));
        TYPES::Types * psrc2_value = (TYPES::Types*)ssrc2->pvalue;
        switch (type)
        {
            case K_B16:
                s2->b16 = psrc2_value->b16;
                break;
            case K_B32:
                s2->b32 = psrc2_value->b32;
                break;
            case K_B64:
                s2->b64 = psrc2_value->b64;
                break;
            case K_PRED:
                s2->pred = psrc2_value->pred;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected OR instruction");
        }
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected OR instruction");

    switch (type)
    {
        case K_B16:
            d->b16 = s1->b16 | s2->b16;
            break;
        case K_B32:
            d->b32 = s1->b32 | s2->b32;
            break;
        case K_S64:
            d->b64 = s1->b64 | s2->b64;
            break;
        case K_PRED:
            d->pred = s1->pred | s2->pred;
            break;
        default:
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected OR instruction");
    }
    return 0;
}

int THREAD::DoPmevent(TREE * inst)
{
    throw new EMULATED_DEVICE::EMU_ERROR("PMEVENT unimplemented");
}

int THREAD::DoPopc(TREE * inst)
{
    throw new EMULATED_DEVICE::EMU_ERROR("POPC unimplemented");
}

int THREAD::DoPrefetch(TREE * inst)
{
    throw new EMULATED_DEVICE::EMU_ERROR("PREFETCH unimplemented");
}

int THREAD::DoPrefetchu(TREE * inst)
{
    throw new EMULATED_DEVICE::EMU_ERROR("PREFETCHU unimplemented");
}

int THREAD::DoPrmt(TREE * inst)
{
    throw new EMULATED_DEVICE::EMU_ERROR("PRMT unimplemented");
}

int THREAD::DoRcp(TREE * inst)
{
    int start = 0;
    if (inst->GetChild(start)->GetType() == TREE_PRED)
        start++;
    assert(inst->GetChild(start)->GetType() == KI_RCP);
    start++;
    TREE * ttype = 0;
    TREE * odst = 0;
    TREE * osrc1 = 0;
    for (;; ++start)
    {
        TREE * t = inst->GetChild(start);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == TREE_TYPE)
            ttype = t;
        else if (gt == TREE_OPR)
        {
            if (odst == 0)
            {
                odst = t;
            } else if (osrc1 == 0)
            {
                osrc1 = t;
            }
            else
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected RCP instruction");
        }
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected RCP instruction");
    }
    assert(ttype != 0);
    assert(odst != 0);
    assert(osrc1 != 0);
    bool ftz = false;
    int rnd = 0;
    for (int i = 0; ; ++i)
    {
        TREE * t = ttype->GetChild(i);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == K_F32 || gt == K_F64)
            ttype = t;
        else if (gt == K_FTZ)
            ftz = true;
        else if (gt == K_APPROX)
            ;
        else if (gt == K_RN || gt == K_RZ || gt == K_RM || gt == K_RP)
            rnd = gt;
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected RCP instruction");
    }
    assert(ttype != 0);
    //this->device->unimplemented(ftz, "RCP.ftz not implemented.");

    int type = ttype->GetType();
    TREE * dst = odst->GetChild(0);
    TREE * src1 = osrc1->GetChild(0);

    SYMBOL * sdst = 0;
    if (dst->GetType() == T_WORD)
    {
        sdst = this->symbol_table->FindSymbol(dst->GetText());
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected RCP instruction");

    TYPES::Types value1;
    char * dummy;
    TYPES::Types * d = (TYPES::Types*)sdst->pvalue;
    TYPES::Types * s1 = &value1;

    if (src1->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src1->GetChild( 0));
        switch (type)
        {
            case K_F32:
                s1->f32 = c.value.f32;
                break;
            case K_F64:
                s1->f64 = c.value.f64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected RCP instruction");
        }
    } else if (src1->GetType() == T_WORD)
    {
        SYMBOL * ssrc1 = this->symbol_table->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        assert(ssrc1->size == this->device->Sizeof(type));
        TYPES::Types * psrc1_value = (TYPES::Types*)ssrc1->pvalue;
        switch (type)
        {
            case K_F32:
                s1->f32 = psrc1_value->f32;
                break;
            case K_F64:
                s1->f64 = psrc1_value->f64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected RCP instruction");
        }
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected RCP instruction");

    switch (type)
    {
        case K_F32:
            d->f32 = 1 / s1->f32;
            break;
        case K_F64:
            d->f64 = 1 / s1->f64;
            break;
        default:
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected RCP instruction");
    }
    return 0;
}

int THREAD::DoRed(TREE * inst)
{
    throw new EMULATED_DEVICE::EMU_ERROR("RED unimplemented");
}

int THREAD::DoRem(TREE * inst)
{
    int start = 0;
    if (inst->GetChild(start)->GetType() == TREE_PRED)
        start++;
    assert(inst->GetChild(start)->GetType() == KI_REM);
    start++;
    assert(inst->GetChild(start)->GetType() == TREE_TYPE);
    TREE * ttype = inst->GetChild(start);
    start++;
    TREE * odst = 0;
    TREE * osrc1 = 0;
    TREE * osrc2 = 0;
    for (;; ++start)
    {
        TREE * t = inst->GetChild(start);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == TREE_OPR)
        {
            if (odst == 0)
            {
                odst = t;
            } else if (osrc1 == 0)
            {
                osrc1 = t;
            } else if (osrc2 == 0)
            {
                osrc2 = t;
            }
            else
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected REM instruction");
        }
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected REM instruction");
    }
    assert(ttype != 0);
    assert(odst != 0);
    assert(osrc1 != 0);
    assert(osrc2 != 0);
    int type = 0;
    for (int i = 0; ; ++i)
    {
        TREE * t = ttype->GetChild(i);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == K_U16 || gt == K_U32 || gt == K_U64
                 || gt == K_S16 || gt == K_S32 || gt == K_S64)
            type = gt;
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected REM instruction");
    }
    assert(type != 0);
    TREE * dst = odst->GetChild(0);
    TREE * src1 = osrc1->GetChild(0);
    TREE * src2 = osrc2->GetChild(0);

    TYPES::Types * pdst_value;
    TYPES::Types * psrc1_value;
    TYPES::Types * psrc2_value;
    TYPES::Types src1_value;// used if literal
    TYPES::Types src2_value;// used if literal

    SYMBOL * sdst = 0;
    SYMBOL * ssrc1 = 0;
    SYMBOL * ssrc2 = 0;
    assert(dst->GetType() == T_WORD);
    sdst = this->symbol_table->FindSymbol(dst->GetText());
    char * dummy;

    TYPES::Types value1; // used if literal
    TYPES::Types value2; // used if literal
    TYPES::Types * d = (TYPES::Types*)sdst->pvalue;
    TYPES::Types * s1 = &value1;
    TYPES::Types * s2 = &value2;

    if (src1->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src1->GetChild( 0));
        switch (type)
        {
            case K_U16:
                s1->u16 = c.value.u16;
                break;
            case K_S16:
                s1->s16 = c.value.s16;
                break;
            case K_U32:
                s1->u32 = c.value.u32;
                break;
            case K_S32:
                s1->s32 = c.value.s32;
                break;
            case K_U64:
                s1->u64 = c.value.u64;
                break;
            case K_S64:
                s1->s64 = c.value.s64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected REM instruction");
        }
    } else if (src1->GetType() == T_WORD)
    {
        ssrc1 = this->symbol_table->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        s1 = (TYPES::Types*)ssrc1->pvalue;
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected REM instruction");

    if (src2->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src2->GetChild(0));
        switch (type)
        {
            case K_U16:
                s2->u16 = c.value.u16;
                break;
            case K_S16:
                s2->s16 = c.value.s16;
                break;
            case K_U32:
                s2->u32 = c.value.u32;
                break;
            case K_S32:
                s2->s32 = c.value.s32;
                break;
            case K_U64:
                s2->u64 = c.value.u64;
                break;
            case K_S64:
                s2->s64 = c.value.s64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected REM instruction");
        }
    } else if (src2->GetType() == T_WORD)
    {
        ssrc2 = this->symbol_table->FindSymbol(src2->GetText());
        assert(ssrc2 != 0);
        s2 = (TYPES::Types*)ssrc2->pvalue;
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected REM instruction");

    switch (type)
    {
        case K_U16:
            d->u16 = s1->u16 % s2->u16;
            break;
        case K_S16:
            d->s16 = s1->s16 % s2->s16;
            break;
        case K_U32:
            d->u32 = s1->u32 % s2->u32;
            break;
        case K_S32:
            d->s32 = s1->s32 % s2->s32;
            break;
        case K_U64:
            d->u64 = s1->u64 % s2->u64;
            break;
        case K_S64:
            d->s64 = s1->s64 % s2->s64;
            break;
        default:
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected REM instruction");
    }
    return 0;
}

int THREAD::DoRet(TREE * inst)
{
    return -KI_RET;
}

int THREAD::DoRsqrt(TREE * inst)
{
    throw new EMULATED_DEVICE::EMU_ERROR("RSQRT unimplemented");
}

int THREAD::DoSad(TREE * inst)
{
    throw new EMULATED_DEVICE::EMU_ERROR("SAD unimplemented");
}

int THREAD::DoSelp(TREE * inst)
{
    int start = 0;
    if (inst->GetChild(start)->GetType() == TREE_PRED)
        start++;
    assert(inst->GetChild(start)->GetType() == KI_SELP);
    start++;
    TREE * ttype = 0;
    TREE * odst = 0;
    TREE * osrc1 = 0;
    TREE * osrc2 = 0;
    TREE * osrc3 = 0;
    for (;; ++start)
    {
        TREE * t = inst->GetChild(start);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == TREE_TYPE)
            ttype = t;
        else if (gt == TREE_OPR)
        {
            if (odst == 0)
            {
                odst = t;
            } else if (osrc1 == 0)
            {
                osrc1 = t;
            } else if (osrc2 == 0)
            {
                osrc2 = t;
            } else if (osrc3 == 0)
            {
                osrc3 = t;
            }
            else
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SELP instruction");
        }
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SELP instruction");
    }
    assert(ttype != 0);
    assert(odst != 0);
    assert(osrc1 != 0);
    assert(osrc2 != 0);
    assert(osrc3 != 0);
    bool ftz = false;
    for (int i = 0; ; ++i)
    {
        TREE * t = ttype->GetChild(i);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == K_U16 || gt == K_U32 || gt == K_U64
                 || gt == K_S16 || gt == K_S32 || gt == K_S64
                 || gt == K_B16 || gt == K_B32 || gt == K_B64
                 || gt == K_F32 || gt == K_F64)
            ttype = t;
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SELP instruction");
    }
    assert(ttype != 0);
    this->device->unimplemented(ftz, "SELP.ftz not implemented.");

    int type = ttype->GetType();
    TREE * dst = odst->GetChild(0);
    TREE * src1 = osrc1->GetChild(0);
    TREE * src2 = osrc2->GetChild(0);
    TREE * src3 = osrc3->GetChild(0);

    SYMBOL * sdst = 0;
    if (dst->GetType() == T_WORD)
    {
        sdst = this->symbol_table->FindSymbol(dst->GetText());
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SELP instruction");

    TYPES::Types * d = (TYPES::Types*)sdst->pvalue;
    TYPES::Types value1;
    TYPES::Types value2;
    TYPES::Types * s1 = &value1;
    TYPES::Types * s2 = &value2;
    TYPES::Types * s3 = 0;
    char * dummy;

    if (src1->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src1->GetChild( 0));
        switch (type)
        {
            case K_S16:
                s1->s16 = c.value.s16;
                break;
            case K_S32:
                s1->s32 = c.value.s32;
                break;
            case K_S64:
                s1->s64 = c.value.s64;
                break;
            case K_U16:
                s1->u16 = c.value.u16;
                break;
            case K_U32:
                s1->u32 = c.value.u32;
                break;
            case K_U64:
                s1->u64 = c.value.u64;
                break;
            case K_B16:
                s1->b16 = c.value.b16;
                break;
            case K_B32:
                s1->b32 = c.value.b32;
                break;
            case K_B64:
                s1->b64 = c.value.b64;
                break;
            case K_F32:
                s1->f32 = c.value.f32;
                break;
            case K_F64:
                s1->f64 = c.value.f64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SELP instruction");
        }
    } else if (src1->GetType() == T_WORD)
    {
        SYMBOL * ssrc1 = this->symbol_table->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        assert(ssrc1->size == this->device->Sizeof(type));
        TYPES::Types * psrc1_value = (TYPES::Types*)ssrc1->pvalue;
        switch (type)
        {
            case K_S16:
                s1->s16 = psrc1_value->s16;
                break;
            case K_S32:
                s1->s32 = psrc1_value->s32;
                break;
            case K_S64:
                s1->s64 = psrc1_value->s64;
                break;
            case K_U16:
                s1->u16 = psrc1_value->u16;
                break;
            case K_U32:
                s1->u32 = psrc1_value->u32;
                break;
            case K_U64:
                s1->u64 = psrc1_value->u64;
                break;
            case K_B16:
                s1->b16 = psrc1_value->b16;
                break;
            case K_B32:
                s1->b32 = psrc1_value->b32;
                break;
            case K_B64:
                s1->b64 = psrc1_value->b64;
                break;
            case K_F32:
                s1->f32 = psrc1_value->f32;
                break;
            case K_F64:
                s1->f64 = psrc1_value->f64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SELP instruction");
        }
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SELP instruction");

    if (src2->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src2->GetChild(0));
        switch (type)
        {
            case K_S16:
                s2->s16 = c.value.s16;
                break;
            case K_S32:
                s2->s32 = c.value.s32;
                break;
            case K_S64:
                s2->s64 = c.value.s64;
                break;
            case K_U16:
                s2->u16 = c.value.u16;
                break;
            case K_U32:
                s2->u32 = c.value.u32;
                break;
            case K_U64:
                s2->u64 = c.value.u64;
                break;
            case K_B16:
                s2->b16 = c.value.b16;
                break;
            case K_B32:
                s2->b32 = c.value.b32;
                break;
            case K_B64:
                s2->b64 = c.value.b64;
                break;
            case K_F32:
                s2->f32 = c.value.f32;
                break;
            case K_F64:
                s2->f64 = c.value.f64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SELP instruction");
        }
    } else if (src2->GetType() == T_WORD)
    {
        SYMBOL * ssrc2 = this->symbol_table->FindSymbol(src2->GetText());
        assert(ssrc2 != 0);
        assert(ssrc2->size == this->device->Sizeof(type));
        TYPES::Types * psrc2_value = (TYPES::Types*)ssrc2->pvalue;
        switch (type)
        {
            case K_S16:
                s2->s16 = psrc2_value->s16;
                break;
            case K_S32:
                s2->s32 = psrc2_value->s32;
                break;
            case K_S64:
                s2->s64 = psrc2_value->s64;
                break;
            case K_U16:
                s2->u16 = psrc2_value->u16;
                break;
            case K_U32:
                s2->u32 = psrc2_value->u32;
                break;
            case K_U64:
                s2->u64 = psrc2_value->u64;
                break;
            case K_B16:
                s2->b16 = psrc2_value->b16;
                break;
            case K_B32:
                s2->b32 = psrc2_value->b32;
                break;
            case K_B64:
                s2->b64 = psrc2_value->b64;
                break;
            case K_F32:
                s2->f32 = psrc2_value->f32;
                break;
            case K_F64:
                s2->f64 = psrc2_value->f64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SELP instruction");
        }
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SELP instruction");

    if (src3->GetType() == T_WORD)
    {
        SYMBOL * ssrc3 = this->symbol_table->FindSymbol(src3->GetText());
        assert(ssrc3 != 0);
// FIX      assert(strcmp(ssrc3->type, ".pred") == 0);
        s3 = (TYPES::Types*)ssrc3->pvalue;
        assert(s3 != 0);
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SELP instruction");

    switch (type)
    {
        case K_S16:
            d->s16 = s3->pred == 1 ? s1->s16 : s2->s16;
            break;
        case K_S32:
            d->s32 = s3->pred == 1 ? s1->s32 : s2->s32;
            break;
        case K_S64:
            d->s64 = s3->pred == 1 ? s1->s64 : s2->s64;
            break;
        case K_U16:
            d->u16 = s3->pred == 1 ? s1->u16 : s2->u16;
            break;
        case K_U32:
            d->s32 = s3->pred == 1 ? s1->u32 : s2->u32;
            break;
        case K_U64:
            d->u64 = s3->pred == 1 ? s1->u64 : s2->u64;
            break;
        case K_B16:
            d->b16 = s3->pred == 1 ? s1->b16 : s2->b16;
            break;
        case K_B32:
            d->b32 = s3->pred == 1 ? s1->b32 : s2->b32;
            break;
        case K_B64:
            d->b64 = s3->pred == 1 ? s1->b64 : s2->b64;
            break;
        case K_F32:
            d->f32 = s3->pred == 1 ? s1->f32 : s2->f32;
            break;
        case K_F64:
            d->f64 = s3->pred == 1 ? s1->f64 : s2->f64;
            break;
        default:
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SELP instruction");
    }
    return 0;
}

int THREAD::DoSet(TREE * inst)
{
    int start = 0;
    if (inst->GetChild(start)->GetType() == TREE_PRED)
        start++;
    assert(inst->GetChild(start)->GetType() == KI_SET);
    start++;
    TREE * ttype = 0;
    TREE * odst1 = 0;
    TREE * odst2 = 0;
    TREE * osrc1 = 0;
    TREE * osrc2 = 0;
    TREE * osrc3 = 0;
    for (;; ++start)
    {
        TREE * t = inst->GetChild(start);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == TREE_TYPE)
            ttype = t;
        else if (gt == TREE_OPR)
        {
            if (odst1 == 0)
            {
                odst1 = t;
            } else if (osrc1 == 0)
            {
                osrc1 = t;
            } else if (osrc2 == 0)
            {
                osrc2 = t;
            } else if (osrc3 == 0)
            {
                osrc3 = t;
            }
            else
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SET instruction");
        }
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SET instruction");
    }
    assert(ttype != 0);
    assert(odst1 != 0);
    assert(osrc1 != 0);
    assert(osrc2 != 0);

    // third source not supported yet.
    assert(osrc3 == 0);

    // split destination, eg, if "p|q" syntax was used, is not
    // supported yet.
    assert(odst1->GetChild(1) == 0);
    bool ftz = false;
    int op = 0;
    int stype = 0;
    int dtype = 0;
    for (int i = 0; ; ++i)
    {
        TREE * t = ttype->GetChild(i);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == K_FTZ)
            ftz = true;
        else if (gt == K_U16 || gt == K_U32 || gt == K_U64
                 || gt == K_S16 || gt == K_S32 || gt == K_S64
                 || gt == K_B16 || gt == K_B32 || gt == K_B64
                 || gt == K_F32 || gt == K_F64)
        {
            if (dtype == 0)
                dtype = gt;
            else if (stype == 0)
                stype = gt;
            else
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SET instruction");
        }
        else if (gt == K_EQ || gt == K_NE || gt == K_LT || gt == K_LE || gt == K_GT || gt == K_GE
                 || gt == K_LO || gt == K_LS || gt == K_HI || gt == K_HS
                 || gt == K_EQU || gt == K_NEU || gt == K_LTU || gt == K_LEU
                 || gt == K_GTU || gt == K_GEU || gt == K_NUM || gt == K_NAN)
            op = gt;
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SET instruction");
    }
    assert(dtype != 0);
    assert(stype != 0);
    assert(op != 0);
    assert(ftz == false);

    TREE * dst1 = odst1->GetChild(0);
    TREE * src1 = osrc1->GetChild(0);
    TREE * src2 = osrc2->GetChild(0);

    SYMBOL * sdst1 = 0;
    SYMBOL * ssrc1 = 0;
    SYMBOL * ssrc2 = 0;
    TYPES::Types value1;
    TYPES::Types value2;
    TYPES::Types * s1 = &value1;
    TYPES::Types * s2 = &value2;

    if (dst1->GetType() == T_WORD)
    {
        sdst1 = this->symbol_table->FindSymbol(dst1->GetText());
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SET instruction");

    TYPES::Types * d = (TYPES::Types*)sdst1->pvalue;

    if (src1->GetType() == T_WORD)
    {
        ssrc1 = this->symbol_table->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        s1 = (TYPES::Types*)ssrc1->pvalue;
        assert(s1 != 0);
    }
    else if (src1->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(stype, src1->GetChild( 0));
        switch (stype)
        {
            case K_B16:
                s1->b16 = c.value.b16;
                break;
            case K_B32:
                s1->b32 = c.value.b32;
                break;
            case K_B64:
                s1->b64 = c.value.b64;
                break;
            case K_U16:
                s1->u16 = c.value.u16;
                break;
            case K_U32:
                s1->u32 = c.value.u32;
                break;
            case K_U64:
                s1->u64 = c.value.u64;
                break;
            case K_S16:
                s1->s16 = c.value.s16;
                break;
            case K_S32:
                s1->s32 = c.value.s32;
                break;
            case K_S64:
                s1->s64 = c.value.s64;
                break;
            case K_F32:
                s1->f32 = c.value.f32;
                break;
            case K_F64:
                s1->f64 = c.value.f64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SET instruction");
        }
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SET instruction");

    if (src2->GetType() == T_WORD)
    {
        ssrc2 = this->symbol_table->FindSymbol(src2->GetText());
        assert(ssrc2 != 0);
        s2 = (TYPES::Types*)ssrc2->pvalue;
        assert(s2 != 0);
    }
    else if (src2->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(stype, src2->GetChild(0));
        switch (stype)
        {
            case K_B16:
                s2->b16 = c.value.b16;
                break;
            case K_B32:
                s2->b32 = c.value.b32;
                break;
            case K_B64:
                s2->b64 = c.value.b64;
                break;
            case K_U16:
                s2->u16 = c.value.u16;
                break;
            case K_U32:
                s2->u32 = c.value.u32;
                break;
            case K_U64:
                s2->u64 = c.value.u64;
                break;
            case K_S16:
                s2->s16 = c.value.s16;
                break;
            case K_S32:
                s2->s32 = c.value.s32;
                break;
            case K_S64:
                s2->s64 = c.value.s64;
                break;
            case K_F32:
                s2->f32 = c.value.f32;
                break;
            case K_F64:
                s2->f64 = c.value.f64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SET instruction");
        }
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SET instruction");

    bool compare = false;
    
    switch (op)
    {
        case K_EQ:
            switch (stype)
            {
                case K_B16:
                    compare = s1->b16 == s2->b16;
                    break;
                case K_B32:
                    compare = s1->b32 == s2->b32;
                    break;
                case K_B64:
                    compare = s1->b64 == s2->b64;
                    break;
                case K_U16:
                    compare = s1->u16 == s2->u16;
                    break;
                case K_U32:
                    compare = s1->u32 == s2->u32;
                    break;
                case K_U64:
                    compare = s1->u64 == s2->u64;
                    break;
                case K_S16:
                    compare = s1->s16 == s2->s16;
                    break;
                case K_S32:
                    compare = s1->s32 == s2->s32;
                    break;
                case K_S64:
                    compare = s1->s64 == s2->s64;
                    break;
                case K_F32:
                    compare = s1->f32 == s2->f32;
                    break;
                case K_F64:
                    compare = s1->f64 == s2->f64;
                    break;
                default:
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SET instruction");
            }
            break;
        case K_NE:
            switch (stype)
            {
                case K_B16:
                    compare = s1->b16 != s2->b16;
                    break;
                case K_B32:
                    compare = s1->b32 != s2->b32;
                    break;
                case K_B64:
                    compare = s1->b64 != s2->b64;
                    break;
                case K_U16:
                    compare = s1->u16 != s2->u16;
                    break;
                case K_U32:
                    compare = s1->u32 != s2->u32;
                    break;
                case K_U64:
                    compare = s1->u64 != s2->u64;
                    break;
                case K_S16:
                    compare = s1->s16 != s2->s16;
                    break;
                case K_S32:
                    compare = s1->s32 != s2->s32;
                    break;
                case K_S64:
                    compare = s1->s64 != s2->s64;
                    break;
                case K_F32:
                    compare = s1->f32 != s2->f32;
                    break;
                case K_F64:
                    compare = s1->f64 != s2->f64;
                    break;
                default:
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SET instruction");
            }
            break;
        case K_LT:
            switch (stype)
            {
                case K_B16:
                    compare = s1->b16 < s2->b16;
                    break;
                case K_B32:
                    compare = s1->b32 < s2->b32;
                    break;
                case K_B64:
                    compare = s1->b64 < s2->b64;
                    break;
                case K_U16:
                    compare = s1->u16 < s2->u16;
                    break;
                case K_U32:
                    compare = s1->u32 < s2->u32;
                    break;
                case K_U64:
                    compare = s1->u64 < s2->u64;
                    break;
                case K_S16:
                    compare = s1->s16 < s2->s16;
                    break;
                case K_S32:
                    compare = s1->s32 < s2->s32;
                    break;
                case K_S64:
                    compare = s1->s64 < s2->s64;
                    break;
                case K_F32:
                    compare = s1->f32 < s2->f32;
                    break;
                case K_F64:
                    compare = s1->f64 < s2->f64;
                    break;
                default:
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SET instruction");
            }
            break;
        case K_LE:
            switch (stype)
            {
                case K_B16:
                    compare = s1->b16 <= s2->b16;
                    break;
                case K_B32:
                    compare = s1->b32 <= s2->b32;
                    break;
                case K_B64:
                    compare = s1->b64 <= s2->b64;
                    break;
                case K_U16:
                    compare = s1->u16 <= s2->u16;
                    break;
                case K_U32:
                    compare = s1->u32 <= s2->u32;
                    break;
                case K_U64:
                    compare = s1->u64 <= s2->u64;
                    break;
                case K_S16:
                    compare = s1->s16 <= s2->s16;
                    break;
                case K_S32:
                    compare = s1->s32 <= s2->s32;
                    break;
                case K_S64:
                    compare = s1->s64 <= s2->s64;
                    break;
                case K_F32:
                    compare = s1->f32 <= s2->f32;
                    break;
                case K_F64:
                    compare = s1->f64 <= s2->f64;
                    break;
                default:
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SET instruction");
            }
            break;
        case K_GT:
            switch (stype)
            {
                case K_B16:
                    compare = s1->b16 > s2->b16;
                    break;
                case K_B32:
                    compare = s1->b32 > s2->b32;
                    break;
                case K_B64:
                    compare = s1->b64 > s2->b64;
                    break;
                case K_U16:
                    compare = s1->u16 > s2->u16;
                    break;
                case K_U32:
                    compare = s1->u32 > s2->u32;
                    break;
                case K_U64:
                    compare = s1->u64 > s2->u64;
                    break;
                case K_S16:
                    compare = s1->s16 > s2->s16;
                    break;
                case K_S32:
                    compare = s1->s32 > s2->s32;
                    break;
                case K_S64:
                    compare = s1->s64 > s2->s64;
                    break;
                case K_F32:
                    compare = s1->f32 > s2->f32;
                    break;
                case K_F64:
                    compare = s1->f64 > s2->f64;
                    break;
                default:
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SET instruction");
            }
            break;
        case K_GE:
            switch (stype)
            {
                case K_B16:
                    compare = s1->b16 >= s2->b16;
                    break;
                case K_B32:
                    compare = s1->b32 >= s2->b32;
                    break;
                case K_B64:
                    compare = s1->b64 >= s2->b64;
                    break;
                case K_U16:
                    compare = s1->u16 >= s2->u16;
                    break;
                case K_U32:
                    compare = s1->u32 >= s2->u32;
                    break;
                case K_U64:
                    compare = s1->u64 >= s2->u64;
                    break;
                case K_S16:
                    compare = s1->s16 >= s2->s16;
                    break;
                case K_S32:
                    compare = s1->s32 >= s2->s32;
                    break;
                case K_S64:
                    compare = s1->s64 >= s2->s64;
                    break;
                case K_F32:
                    compare = s1->f32 >= s2->f32;
                    break;
                case K_F64:
                    compare = s1->f64 >= s2->f64;
                    break;
                default:
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SET instruction");
            }
            break;
        case K_LO:
            switch (stype)
            {
                case K_B16:
                    compare = s1->b16 < s2->b16;
                    break;
                case K_B32:
                    compare = s1->b32 < s2->b32;
                    break;
                case K_B64:
                    compare = s1->b64 < s2->b64;
                    break;
                case K_U16:
                    compare = s1->u16 < s2->u16;
                    break;
                case K_U32:
                    compare = s1->u32 < s2->u32;
                    break;
                case K_U64:
                    compare = s1->u64 < s2->u64;
                    break;
                case K_S16:
                    compare = s1->s16 < s2->s16;
                    break;
                case K_S32:
                    compare = s1->s32 < s2->s32;
                    break;
                case K_S64:
                    compare = s1->s64 < s2->s64;
                    break;
                case K_F32:
                    compare = s1->f32 < s2->f32;
                    break;
                case K_F64:
                    compare = s1->f64 < s2->f64;
                    break;
                default:
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SET instruction");
            }
            break;
        case K_LS:
            switch (stype)
            {
                case K_B16:
                    compare = s1->b16 <= s2->b16;
                    break;
                case K_B32:
                    compare = s1->b32 <= s2->b32;
                    break;
                case K_B64:
                    compare = s1->b64 <= s2->b64;
                    break;
                case K_U16:
                    compare = s1->u16 <= s2->u16;
                    break;
                case K_U32:
                    compare = s1->u32 <= s2->u32;
                    break;
                case K_U64:
                    compare = s1->u64 <= s2->u64;
                    break;
                case K_S16:
                    compare = s1->s16 <= s2->s16;
                    break;
                case K_S32:
                    compare = s1->s32 <= s2->s32;
                    break;
                case K_S64:
                    compare = s1->s64 <= s2->s64;
                    break;
                case K_F32:
                    compare = s1->f32 <= s2->f32;
                    break;
                case K_F64:
                    compare = s1->f64 <= s2->f64;
                    break;
                default:
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SET instruction");
            }
            break;
        case K_HI:
            switch (stype)
            {
                case K_B16:
                    compare = s1->b16 > s2->b16;
                    break;
                case K_B32:
                    compare = s1->b32 > s2->b32;
                    break;
                case K_B64:
                    compare = s1->b64 > s2->b64;
                    break;
                case K_U16:
                    compare = s1->u16 > s2->u16;
                    break;
                case K_U32:
                    compare = s1->u32 > s2->u32;
                    break;
                case K_U64:
                    compare = s1->u64 > s2->u64;
                    break;
                case K_S16:
                    compare = s1->s16 > s2->s16;
                    break;
                case K_S32:
                    compare = s1->s32 > s2->s32;
                    break;
                case K_S64:
                    compare = s1->s64 > s2->s64;
                    break;
                case K_F32:
                    compare = s1->f32 > s2->f32;
                    break;
                case K_F64:
                    compare = s1->f64 > s2->f64;
                    break;
                default:
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SET instruction");
            }
            break;
        case K_HS:
            switch (stype)
            {
                case K_B16:
                    compare = s1->b16 >= s2->b16;
                    break;
                case K_B32:
                    compare = s1->b32 >= s2->b32;
                    break;
                case K_B64:
                    compare = s1->b64 >= s2->b64;
                    break;
                case K_U16:
                    compare = s1->u16 >= s2->u16;
                    break;
                case K_U32:
                    compare = s1->u32 >= s2->u32;
                    break;
                case K_U64:
                    compare = s1->u64 >= s2->u64;
                    break;
                case K_S16:
                    compare = s1->s16 >= s2->s16;
                    break;
                case K_S32:
                    compare = s1->s32 >= s2->s32;
                    break;
                case K_S64:
                    compare = s1->s64 >= s2->s64;
                    break;
                case K_F32:
                    compare = s1->f32 >= s2->f32;
                    break;
                case K_F64:
                    compare = s1->f64 >= s2->f64;
                    break;
                default:
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SET instruction");
            }
            break;
        default:
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SET instruction");
            break;
    }

    // Store the result.
    switch (dtype)
    {
        case K_U32:
            d->u32 = (compare)? -1 : 0;
            break;
        case K_S32:
            d->s32 = (compare)? -1 : 0;
            break;
        case K_F32:
            d->f32 = (compare)? 1.0 : 0;
            break;
        default:
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SET instruction");
    }
    return 0;
}

int THREAD::DoSetp(TREE * inst)
{
    int start = 0;
    if (inst->GetChild(start)->GetType() == TREE_PRED)
        start++;
    assert(inst->GetChild(start)->GetType() == KI_SETP);
    start++;
    TREE * ttype = 0;
    TREE * odst1 = 0;
    TREE * odst2 = 0;
    TREE * osrc1 = 0;
    TREE * osrc2 = 0;
    TREE * osrc3 = 0;
    for (;; ++start)
    {
        TREE * t = inst->GetChild(start);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == TREE_TYPE)
            ttype = t;
        else if (gt == TREE_OPR)
        {
            if (odst1 == 0)
            {
                odst1 = t;
            } else if (osrc1 == 0)
            {
                osrc1 = t;
            } else if (osrc2 == 0)
            {
                osrc2 = t;
            } else if (osrc3 == 0)
            {
                osrc3 = t;
            }
            else
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SETP instruction");
        }
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SETP instruction");
    }
    assert(ttype != 0);
    assert(odst1 != 0);
    assert(osrc1 != 0);
    assert(osrc2 != 0);

    // third source not supported yet.
    assert(osrc3 == 0);

    // split destination, eg, if "p|q" syntax was used, is not
    // supported yet.
    assert(odst1->GetChild(1) == 0);
    bool ftz = false;
    int op = 0;
    for (int i = 0; ; ++i)
    {
        TREE * t = ttype->GetChild(i);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == K_FTZ)
            ftz = true;
        else if (gt == K_U16 || gt == K_U32 || gt == K_U64
                 || gt == K_S16 || gt == K_S32 || gt == K_S64
                 || gt == K_B16 || gt == K_B32 || gt == K_B64
                 || gt == K_F32 || gt == K_F64
                )
            ttype = t;
        else if (gt == K_EQ || gt == K_NE || gt == K_LT || gt == K_LE || gt == K_GT || gt == K_GE
                 || gt == K_LO || gt == K_LS || gt == K_HI || gt == K_HS
                 || gt == K_EQU || gt == K_NEU || gt == K_LTU || gt == K_LEU
                 || gt == K_GTU || gt == K_GEU || gt == K_NUM || gt == K_NAN)
            op = gt;
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SETP instruction");
    }
    assert(ttype != 0);
    assert(op != 0);
    assert(ftz == false);

    int type = ttype->GetType();
    TREE * dst1 = odst1->GetChild(0);
    TREE * src1 = osrc1->GetChild(0);
    TREE * src2 = osrc2->GetChild(0);

    SYMBOL * sdst1 = 0;
    SYMBOL * ssrc1 = 0;
    SYMBOL * ssrc2 = 0;
    TYPES::Types value1;
    TYPES::Types value2;
    TYPES::Types * s1 = &value1;
    TYPES::Types * s2 = &value2;

    if (dst1->GetType() == T_WORD)
    {
        sdst1 = this->symbol_table->FindSymbol(dst1->GetText());
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SETP instruction");

    TYPES::Types * d = (TYPES::Types*)sdst1->pvalue;

    if (src1->GetType() == T_WORD)
    {
        ssrc1 = this->symbol_table->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        s1 = (TYPES::Types*)ssrc1->pvalue;
        assert(s1 != 0);
    }
    else if (src1->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src1->GetChild( 0));
        switch (type)
        {
            case K_B16:
                s1->b16 = c.value.b16;
                break;
            case K_B32:
                s1->b32 = c.value.b32;
                break;
            case K_B64:
                s1->b64 = c.value.b64;
                break;
            case K_U16:
                s1->u16 = c.value.u16;
                break;
            case K_U32:
                s1->u32 = c.value.u32;
                break;
            case K_U64:
                s1->u64 = c.value.u64;
                break;
            case K_S16:
                s1->s16 = c.value.s16;
                break;
            case K_S32:
                s1->s32 = c.value.s32;
                break;
            case K_S64:
                s1->s64 = c.value.s64;
                break;
            case K_F32:
                s1->f32 = c.value.f32;
                break;
            case K_F64:
                s1->f64 = c.value.f64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SETP instruction");
        }
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SETP instruction");

    if (src2->GetType() == T_WORD)
    {
        ssrc2 = this->symbol_table->FindSymbol(src2->GetText());
        assert(ssrc2 != 0);
        s2 = (TYPES::Types*)ssrc2->pvalue;
        assert(s2 != 0);
    }
    else if (src2->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src2->GetChild(0));
        switch (type)
        {
            case K_B16:
                s2->b16 = c.value.b16;
                break;
            case K_B32:
                s2->b32 = c.value.b32;
                break;
            case K_B64:
                s2->b64 = c.value.b64;
                break;
            case K_U16:
                s2->u16 = c.value.u16;
                break;
            case K_U32:
                s2->u32 = c.value.u32;
                break;
            case K_U64:
                s2->u64 = c.value.u64;
                break;
            case K_S16:
                s2->s16 = c.value.s16;
                break;
            case K_S32:
                s2->s32 = c.value.s32;
                break;
            case K_S64:
                s2->s64 = c.value.s64;
                break;
            case K_F32:
                s2->f32 = c.value.f32;
                break;
            case K_F64:
                s2->f64 = c.value.f64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SETP instruction");
        }
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SETP instruction");

    switch (op)
    {
        case K_EQ:
            switch (type)
            {
                case K_B16:
                    d->pred = s1->b16 == s2->b16;
                    break;
                case K_B32:
                    d->pred = s1->b32 == s2->b32;
                    break;
                case K_B64:
                    d->pred = s1->b64 == s2->b64;
                    break;
                case K_U16:
                    d->pred = s1->u16 == s2->u16;
                    break;
                case K_U32:
                    d->pred = s1->u32 == s2->u32;
                    break;
                case K_U64:
                    d->pred = s1->u64 == s2->u64;
                    break;
                case K_S16:
                    d->pred = s1->s16 == s2->s16;
                    break;
                case K_S32:
                    d->pred = s1->s32 == s2->s32;
                    break;
                case K_S64:
                    d->pred = s1->s64 == s2->s64;
                    break;
                case K_F32:
                    d->pred = s1->f32 == s2->f32;
                    break;
                case K_F64:
                    d->pred = s1->f64 == s2->f64;
                    break;
                default:
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SETP instruction");
            }
            break;
        case K_NE:
            switch (type)
            {
                case K_B16:
                    d->pred = s1->b16 != s2->b16;
                    break;
                case K_B32:
                    d->pred = s1->b32 != s2->b32;
                    break;
                case K_B64:
                    d->pred = s1->b64 != s2->b64;
                    break;
                case K_U16:
                    d->pred = s1->u16 != s2->u16;
                    break;
                case K_U32:
                    d->pred = s1->u32 != s2->u32;
                    break;
                case K_U64:
                    d->pred = s1->u64 != s2->u64;
                    break;
                case K_S16:
                    d->pred = s1->s16 != s2->s16;
                    break;
                case K_S32:
                    d->pred = s1->s32 != s2->s32;
                    break;
                case K_S64:
                    d->pred = s1->s64 != s2->s64;
                    break;
                case K_F32:
                    d->pred = s1->f32 != s2->f32;
                    break;
                case K_F64:
                    d->pred = s1->f64 != s2->f64;
                    break;
                default:
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SETP instruction");
            }
            break;
        case K_LT:
            switch (type)
            {
                case K_U16:
                    d->pred = s1->u16 < s2->u16;
                    break;
                case K_U32:
                    d->pred = s1->u32 < s2->u32;
                    break;
                case K_U64:
                    d->pred = s1->u64 < s2->u64;
                    break;
                case K_S16:
                    d->pred = s1->s16 < s2->s16;
                    break;
                case K_S32:
                    d->pred = s1->s32 < s2->s32;
                    break;
                case K_S64:
                    d->pred = s1->s64 < s2->s64;
                    break;
                case K_F32:
                    d->pred = s1->f32 < s2->f32;
                    break;
                case K_F64:
                    d->pred = s1->f64 < s2->f64;
                    break;
                default:
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SETP instruction");
            }
            break;
        case K_LE:
            switch (type)
            {
                case K_U16:
                    d->pred = s1->u16 <= s2->u16;
                    break;
                case K_U32:
                    d->pred = s1->u32 <= s2->u32;
                    break;
                case K_U64:
                    d->pred = s1->u64 <= s2->u64;
                    break;
                case K_S16:
                    d->pred = s1->s16 <= s2->s16;
                    break;
                case K_S32:
                    d->pred = s1->s32 <= s2->s32;
                    break;
                case K_S64:
                    d->pred = s1->s64 <= s2->s64;
                    break;
                case K_F32:
                    d->pred = s1->f32 <= s2->f32;
                    break;
                case K_F64:
                    d->pred = s1->f64 <= s2->f64;
                    break;
                default:
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SETP instruction");
            }
            break;
        case K_GT:
            switch (type)
            {
                case K_U16:
                    d->pred = s1->u16 > s2->u16;
                    break;
                case K_U32:
                    d->pred = s1->u32 > s2->u32;
                    break;
                case K_U64:
                    d->pred = s1->u64 > s2->u64;
                    break;
                case K_S16:
                    d->pred = s1->s16 > s2->s16;
                    break;
                case K_S32:
                    d->pred = s1->s32 > s2->s32;
                    break;
                case K_S64:
                    d->pred = s1->s64 > s2->s64;
                    break;
                case K_F32:
                    d->pred = s1->f32 > s2->f32;
                    break;
                case K_F64:
                    d->pred = s1->f64 > s2->f64;
                    break;
                default:
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SETP instruction");
            }
            break;
        case K_GE:
            switch (type)
            {
                case K_U16:
                    d->pred = s1->u16 >= s2->u16;
                    break;
                case K_U32:
                    d->pred = s1->u32 >= s2->u32;
                    break;
                case K_U64:
                    d->pred = s1->u64 >= s2->u64;
                    break;
                case K_S16:
                    d->pred = s1->s16 >= s2->s16;
                    break;
                case K_S32:
                    d->pred = s1->s32 >= s2->s32;
                    break;
                case K_S64:
                    d->pred = s1->s64 >= s2->s64;
                    break;
                case K_F32:
                    d->pred = s1->f32 >= s2->f32;
                    break;
                case K_F64:
                    d->pred = s1->f64 >= s2->f64;
                    break;
                default:
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SETP instruction");
            }
            break;
        case K_LO:
            switch (type)
            {
                case K_U16:
                    d->pred = s1->u16 < s2->u16;
                    break;
                case K_U32:
                    d->pred = s1->u32 < s2->u32;
                    break;
                case K_U64:
                    d->pred = s1->u64 < s2->u64;
                    break;
                default:
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SETP instruction");
            }
            break;
        case K_LS:
            switch (type)
            {
                case K_U16:
                    d->pred = s1->u16 <= s2->u16;
                    break;
                case K_U32:
                    d->pred = s1->u32 <= s2->u32;
                    break;
                case K_U64:
                    d->pred = s1->u64 <= s2->u64;
                    break;
                default:
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SETP instruction");
            }
            break;
        case K_HI:
            switch (type)
            {
                case K_U16:
                    d->pred = s1->u16 > s2->u16;
                    break;
                case K_U32:
                    d->pred = s1->u32 > s2->u32;
                    break;
                case K_U64:
                    d->pred = s1->u64 > s2->u64;
                    break;
                default:
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SETP instruction");
            }
            break;
        case K_HS:
            switch (type)
            {
                case K_U16:
                    d->pred = s1->u16 >= s2->u16;
                    break;
                case K_U32:
                    d->pred = s1->u32 >= s2->u32;
                    break;
                case K_U64:
                    d->pred = s1->u64 >= s2->u64;
                    break;
                default:
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SETP instruction");
            }
            break;
        default:
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SETP instruction");
            break;
    }
    return 0;
}

int THREAD::DoShl(TREE * inst)
{
    int start = 0;
    if (inst->GetChild(start)->GetType() == TREE_PRED)
        start++;
    assert(inst->GetChild(start)->GetType() == KI_SHL);
    start++;
    TREE * ttype = 0;
    TREE * odst = 0;
    TREE * osrc1 = 0;
    TREE * osrc2 = 0;
    for (;; ++start)
    {
        TREE * t = inst->GetChild(start);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == TREE_TYPE)
            ttype = t;
        else if (gt == TREE_OPR)
        {
            if (odst == 0)
            {
                odst = t;
            } else if (osrc1 == 0)
            {
                osrc1 = t;
            } else if (osrc2 == 0)
            {
                osrc2 = t;
            }
            else
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SHL instruction");
        }
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SHL instruction");
    }
    assert(ttype != 0);
    assert(odst != 0);
    assert(osrc1 != 0);
    assert(osrc2 != 0);
    int type = 0;
    for (int i = 0; ; ++i)
    {
        TREE * t = ttype->GetChild(i);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == K_B16 || gt == K_B32 || gt == K_B64)
            type = gt;
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SHL instruction");
    }
    assert(type != 0);
    TREE * dst = odst->GetChild(0);
    TREE * src1 = osrc1->GetChild(0);
    TREE * src2 = osrc2->GetChild(0);

    SYMBOL * sdst = 0;
    if (dst->GetType() == T_WORD)
    {
        sdst = this->symbol_table->FindSymbol(dst->GetText());
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SHL instruction");

    TYPES::Types value1;
    TYPES::Types value2;
    char * dummy;
    TYPES::Types * d = (TYPES::Types*)sdst->pvalue;
    TYPES::Types * s1 = &value1;
    TYPES::Types * s2 = &value2;

    if (src1->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src1->GetChild( 0));
        switch (type)
        {
            case K_B16:
                s1->b16 = c.value.b16;
                break;
            case K_B32:
                s1->b32 = c.value.b32;
                break;
            case K_B64:
                s1->b64 = c.value.b64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SHL instruction");
        }
    } else if (src1->GetType() == T_WORD)
    {
        SYMBOL * ssrc1 = this->symbol_table->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        assert(ssrc1->size == this->device->Sizeof(type));
        TYPES::Types * psrc1_value = (TYPES::Types*)ssrc1->pvalue;
        switch (type)
        {
            case K_B16:
                s1->b16 = psrc1_value->b16;
                break;
            case K_B32:
                s1->b32 = psrc1_value->b32;
                break;
            case K_B64:
                s1->b64 = psrc1_value->b64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SHL instruction");
        }
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SHL instruction");

    if (src2->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src2->GetChild(0));
        switch (type)
        {
            case K_B16:
                s2->b16 = c.value.b16;
                break;
            case K_B32:
                s2->b32 = c.value.b32;
                break;
            case K_B64:
                s2->b64 = c.value.b64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SHL instruction");
        }
    } else if (src2->GetType() == T_WORD)
    {
        SYMBOL * ssrc2 = this->symbol_table->FindSymbol(src2->GetText());
        assert(ssrc2 != 0);
        assert(ssrc2->size == this->device->Sizeof(type));
        TYPES::Types * psrc2_value = (TYPES::Types*)ssrc2->pvalue;
        switch (type)
        {
            case K_B16:
                s2->b16 = psrc2_value->b16;
                break;
            case K_B32:
                s2->b32 = psrc2_value->b32;
                break;
            case K_B64:
                s2->b64 = psrc2_value->b64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SHL instruction");
        }
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SHL instruction");

    switch (type)
    {
        case K_B16:
            d->b16 = s1->b16 << s2->b16;
            break;
        case K_B32:
            d->b32 = s1->b32 << s2->b32;
            break;
        case K_B64:
            d->b64 = s1->b64 << s2->b64;
            break;
        default:
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SHL instruction");
    }
    return 0;
}

int THREAD::DoShr(TREE * inst)
{
    int start = 0;
    if (inst->GetChild(start)->GetType() == TREE_PRED)
        start++;
    assert(inst->GetChild(start)->GetType() == KI_SHR);
    start++;
    TREE * ttype = 0;
    TREE * odst = 0;
    TREE * osrc1 = 0;
    TREE * osrc2 = 0;
    for (;; ++start)
    {
        TREE * t = inst->GetChild(start);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == TREE_TYPE)
            ttype = t;
        else if (gt == TREE_OPR)
        {
            if (odst == 0)
            {
                odst = t;
            } else if (osrc1 == 0)
            {
                osrc1 = t;
            } else if (osrc2 == 0)
            {
                osrc2 = t;
            }
            else
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SHR instruction");
        }
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SHR instruction");
    }
    assert(ttype != 0);
    assert(odst != 0);
    assert(osrc1 != 0);
    assert(osrc2 != 0);
    int type = 0;
    for (int i = 0; ; ++i)
    {
        TREE * t = ttype->GetChild(i);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == K_B16 || gt == K_B32 || gt == K_B64
              || gt == K_U16 || gt == K_U32 || gt == K_U64
              || gt == K_S16 || gt == K_S32 || gt == K_S64)
            type = gt;
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SHR instruction");
    }
    assert(type != 0);
    TREE * dst = odst->GetChild(0);
    TREE * src1 = osrc1->GetChild(0);
    TREE * src2 = osrc2->GetChild(0);

    SYMBOL * sdst = 0;
    if (dst->GetType() == T_WORD)
    {
        sdst = this->symbol_table->FindSymbol(dst->GetText());
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SHR instruction");

    TYPES::Types value1;
    TYPES::Types value2;
    char * dummy;
    TYPES::Types * d = (TYPES::Types*)sdst->pvalue;
    TYPES::Types * s1 = &value1;
    TYPES::Types * s2 = &value2;

    if (src1->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src1->GetChild( 0));
        switch (type)
        {
            case K_B16:
                s1->b16 = c.value.b16;
                break;
            case K_B32:
                s1->b32 = c.value.b32;
                break;
            case K_B64:
                s1->b64 = c.value.b64;
                break;
            case K_U16:
                s1->u16 = c.value.u16;
                break;
            case K_U32:
                s1->u32 = c.value.u32;
                break;
            case K_U64:
                s1->u64 = c.value.u64;
                break;
            case K_S16:
                s1->s16 = c.value.s16;
                break;
            case K_S32:
                s1->s32 = c.value.s32;
                break;
            case K_S64:
                s1->s64 = c.value.s64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SHR instruction");
        }
    } else if (src1->GetType() == T_WORD)
    {
        SYMBOL * ssrc1 = this->symbol_table->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        assert(ssrc1->size == this->device->Sizeof(type));
        TYPES::Types * psrc1_value = (TYPES::Types*)ssrc1->pvalue;
        switch (type)
        {
            case K_B16:
                s1->b16 = psrc1_value->b16;
                break;
            case K_B32:
                s1->b32 = psrc1_value->b32;
                break;
            case K_B64:
                s1->b64 = psrc1_value->b64;
                break;
            case K_U16:
                s1->u16 = psrc1_value->u16;
                break;
            case K_U32:
                s1->u32 = psrc1_value->u32;
                break;
            case K_U64:
                s1->u64 = psrc1_value->u64;
                break;
            case K_S16:
                s1->s16 = psrc1_value->s16;
                break;
            case K_S32:
                s1->s32 = psrc1_value->s32;
                break;
            case K_S64:
                s1->s64 = psrc1_value->s64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SHR instruction");
        }
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SHR instruction");

    if (src2->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src2->GetChild(0));
        switch (type)
        {
            case K_B16:
                s2->b16 = c.value.b16;
                break;
            case K_B32:
                s2->b32 = c.value.b32;
                break;
            case K_B64:
                s2->b64 = c.value.b64;
                break;
            case K_U16:
                s2->u16 = c.value.u16;
                break;
            case K_U32:
                s2->u32 = c.value.u32;
                break;
            case K_U64:
                s2->u64 = c.value.u64;
                break;
            case K_S16:
                s2->s16 = c.value.s16;
                break;
            case K_S32:
                s2->s32 = c.value.s32;
                break;
            case K_S64:
                s2->s64 = c.value.s64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SHR instruction");
        }
    } else if (src2->GetType() == T_WORD)
    {
        SYMBOL * ssrc2 = this->symbol_table->FindSymbol(src2->GetText());
        assert(ssrc2 != 0);
        assert(ssrc2->size == this->device->Sizeof(type));
        TYPES::Types * psrc2_value = (TYPES::Types*)ssrc2->pvalue;
        switch (type)
        {
            case K_B16:
                s2->b16 = psrc2_value->b16;
                break;
            case K_B32:
                s2->b32 = psrc2_value->b32;
                break;
            case K_B64:
                s2->b64 = psrc2_value->b64;
                break;
            case K_U16:
                s2->u16 = psrc2_value->u16;
                break;
            case K_U32:
                s2->u32 = psrc2_value->u32;
                break;
            case K_U64:
                s2->u64 = psrc2_value->u64;
                break;
            case K_S16:
                s2->s16 = psrc2_value->s16;
                break;
            case K_S32:
                s2->s32 = psrc2_value->s32;
                break;
            case K_S64:
                s2->s64 = psrc2_value->s64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SHR instruction");
        }
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SHR instruction");

    switch (type)
    {
        case K_B16:
            d->b16 = s1->b16 >> s2->b16;
            break;
        case K_B32:
            d->b32 = s1->b32 >> s2->b32;
            break;
        case K_B64:
            d->b64 = s1->b64 >> s2->b64;
            break;
        case K_U16:
            d->u16 = s1->u16 >> s2->u16;
            break;
        case K_U32:
            d->u32 = s1->u32 >> s2->u32;
            break;
        case K_U64:
            d->u64 = s1->u64 >> s2->u64;
            break;
        case K_S16:
            d->s16 = s1->s16 >> s2->s16;
            break;
        case K_S32:
            d->s32 = s1->s32 >> s2->s32;
            break;
        case K_S64:
            d->s64 = s1->s64 >> s2->s64;
            break;
        default:
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SHR instruction");
    }
    return 0;
}

int THREAD::DoSin(TREE * inst)
{
    throw new EMULATED_DEVICE::EMU_ERROR("SIN unimplemented");
}

int THREAD::DoSlct(TREE * inst)
{
    throw new EMULATED_DEVICE::EMU_ERROR("SLCT unimplemented");
}

int THREAD::DoSqrt(TREE * inst)
{
    int start = 0;
    if (inst->GetChild(start)->GetType() == TREE_PRED)
        start++;
    assert(inst->GetChild(start)->GetType() == KI_SQRT);
    start++;
    assert(inst->GetChild(start)->GetType() == TREE_TYPE);
    TREE * ttype = inst->GetChild(start);
    start++;
    TREE * odst = 0;
    TREE * osrc1 = 0;
    for (;; ++start)
    {
        TREE * t = inst->GetChild(start);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == TREE_OPR)
        {
            if (odst == 0)
            {
                odst = t;
            } else if (osrc1 == 0)
            {
                osrc1 = t;
            }
            else
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SQRT instruction");
        }
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SQRT instruction");
    }
    assert(ttype != 0);
    assert(odst != 0);
    assert(osrc1 != 0);
    bool ftz = false;
    TREE * tfrnd = 0;
    for (int i = 0; ; ++i)
    {
        TREE * t = ttype->GetChild(i);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == K_FTZ)
            ftz = true;
        else if (gt == K_F32 || gt == K_F64)
            ttype = t;
        else if (gt == K_RN || gt == K_RZ || gt == K_RM || gt == K_RP)
            tfrnd = t;
        else if (gt == K_FULL || gt == K_APPROX)
            ;
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SQRT instruction");
    }
    assert(ttype != 0);
    assert(ftz == 0);  // unimplemented.
    int type = ttype->GetType();

    TREE * dst = odst->GetChild(0);
    TREE * src1 = osrc1->GetChild(0);

    TYPES::Types * pdst_value;
    TYPES::Types * psrc1_value;
    TYPES::Types src1_value;// used if literal

    SYMBOL * sdst = 0;
    SYMBOL * ssrc1 = 0;
    assert(dst->GetType() == T_WORD);
    sdst = this->symbol_table->FindSymbol(dst->GetText());
    char * dummy;

    TYPES::Types value1; // used if literal
    TYPES::Types * d = (TYPES::Types*)sdst->pvalue;
    TYPES::Types * s1 = &value1;

    if (src1->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src1->GetChild( 0));
        switch (type)
        {
            case K_F32:
                s1->f32 = c.value.f32;
                break;
            case K_F64:
                s1->f64 = c.value.f64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SQRT instruction");
        }
    } else if (src1->GetType() == T_WORD)
    {
        ssrc1 = this->symbol_table->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        s1 = (TYPES::Types*)ssrc1->pvalue;
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SQRT instruction");

    switch (type)
    {
        case K_F32:
            d->f32 = sqrt(s1->f32);
            break;
        case K_F64:
            d->f64 = sqrt(s1->f64);
            break;
        default:
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SQRT instruction");
    }
    return 0;
}

int THREAD::DoSt(TREE * inst)
{
    int start = 0;
    if (inst->GetChild(start)->GetType() == TREE_PRED)
        start++;
    assert(inst->GetChild(start)->GetType() == KI_ST);
    start++;
    TREE * ttype = 0;
    TREE * odst = 0;
    TREE * osrc = 0;
    for (;; ++start)
    {
        TREE * t = inst->GetChild(start);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == TREE_TYPE)
            ttype = t;
        else if (gt == TREE_OPR)
        {
            if (odst == 0)
            {
                odst = t;
            } else if (osrc == 0)
            {
                osrc = t;
            }
            else
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ST instruction");
        }
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ST instruction");
    }
    assert(ttype != 0);
    assert(odst != 0);
    assert(osrc != 0);

    int ss = 0;
    int cop = 0;
    int vec = 0;
    int vol = 0;
    for (int i = 0; ; ++i)
    {
        TREE * t = ttype->GetChild(i);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == K_CONST || gt == K_GLOBAL || gt == K_LOCAL || gt == K_PARAM || gt == K_SHARED)
            ss = gt;
        else if (gt == K_U8 || gt == K_U16 || gt == K_U32 || gt == K_U64
                 || gt == K_S8 || gt == K_S16 || gt == K_S32 || gt == K_S64
                 || gt == K_F32 || gt == K_F64
                 || gt == K_B8 || gt == K_B16 || gt == K_B32 || gt == K_B64)
            ttype = t;
        else if (gt == K_CA || gt == K_CG || gt == K_CS || gt == K_LU || gt == K_CV)
            cop = gt;
        else if (gt == K_V2 || gt == K_V4)
            vec = gt;
        else if (gt == K_VOLATILE)
            vol = gt;
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ST instruction");
    }
    assert(ttype != 0);
    assert(cop == 0);
    int type = ttype->GetType();

    TREE * dst = odst->GetChild( 0);
    assert(dst->GetType() == T_WORD);
    SYMBOL * sdst = this->symbol_table->FindSymbol(dst->GetText());
    assert(sdst != 0);
    TREE * plus = odst->GetChild( 1);
    CONSTANT value(0);
    if (plus != 0)
    {
        TREE * const_expr_tree = odst->GetChild( 2);
        assert(const_expr_tree != 0);
        assert(const_expr_tree->GetType() == TREE_CONSTANT_EXPR);
        TREE * const_expr = const_expr_tree->GetChild(0);
        assert(const_expr != 0);
        value = this->device->Eval(K_S32, const_expr);
    }

    TYPES::Types * d = 0;
    unsigned char * addr = 0;
    switch (sdst->storage_class)
    {
        case K_GLOBAL:
        case K_LOCAL:
        case K_PARAM:
        case K_SHARED:
        case K_CONST:
            {
                addr = (unsigned char *)sdst->pvalue;
            }
            break;
        case K_REG:
            {
                addr = *(unsigned char **)sdst->pvalue;
            }
            break;
        default:
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ST instruction");
    }

    switch (value.type)
    {
        case K_U8:
            addr = addr + value.value.s32;
            break;
        case K_U16:
            addr = addr + value.value.s32;
            break;
        case K_U32:
            addr = addr + value.value.s32;
            break;
        case K_U64:
            addr = addr + value.value.s32;
            break;
        case K_S8:
            addr = addr + value.value.s32;
            break;
        case K_S16:
            addr = addr + value.value.s32;
            break;
        case K_S32:
            addr = addr + value.value.s32;
            break;
        case K_S64:
            addr = addr + value.value.s32;
            break;
        case K_B8:
            addr = addr + value.value.s32;
            break;
        case K_B16:
            addr = addr + value.value.s32;
            break;
        case K_B32:
            addr = addr + value.value.s32;
            break;
        case K_B64:
            addr = addr + value.value.s32;
            break;
        case K_F32:
            addr = addr + value.value.s32;
            break;
        case K_F64:
            addr = addr + value.value.s32;
            break;
        default:
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ST instruction");
    }

    int times = 1;
    if (vec == K_V2)
        times = 2;
    else if (vec == K_V4)
        times = 4;

    for (int i = 0; i < times; ++i)
    {
        TYPES::Types * d = (TYPES::Types*)addr;
        
        TREE * src = osrc->GetChild( i);
        TYPES::Types * s = 0;
        SYMBOL * ssrc = 0;
        if (src->GetType() == TREE_CONSTANT_EXPR)
        {
            // cannot do both store from constant and to vector.
            assert(i == 0);
            CONSTANT c = this->device->Eval(type, src->GetChild(0));
            s = &c.value;
        } else if (src->GetType() == T_WORD)
        {
            ssrc = this->symbol_table->FindSymbol(src->GetText());
            assert(ssrc != 0);
            // Various types of id's to handle:
            switch (ssrc->storage_class)
            {
                case K_GLOBAL:
                case K_LOCAL:
                case K_PARAM:
                case K_SHARED:
                case K_CONST:
                // names in instructions refer to the address of the
                // variable, not the contents.
                    s = (TYPES::Types*)&ssrc->pvalue;
                    break;
                case K_REG:
                // names in instructions refer to the contents of the
                // register.
                    s = (TYPES::Types*)ssrc->pvalue;
                    break;
            }
            if (strcmp(ssrc->typestring, "dim3") == 0)
            {
            // Get qualifier of the structure.
                TREE * tqual = (TREE *)osrc->GetChild(1);
                assert(tqual != 0);
                int qual = tqual->GetType();
                if (qual == K_X)
                {
                    s = (TYPES::Types*)& ((dim3*)ssrc->pvalue)->x;
                } else if (qual == K_Y)
                {
                    s = (TYPES::Types*)& ((dim3*)ssrc->pvalue)->y;
                } else if (qual == K_Z)
                {
                    s = (TYPES::Types*)& ((dim3*)ssrc->pvalue)->z;
                }
                else
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ST instruction");
            }
        }
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ST instruction");

        // Determine if there is a write in or out of bounds.
        switch (ss)
        {
            case K_GLOBAL:
            case K_LOCAL:
            case K_PARAM:
            case K_CONST:
            case K_REG:
                break;
            case K_SHARED:
                {
                    // verify addr is within a pvalue with the given storage class.
                    SYMBOL * ok = this->symbol_table->FindAddr(ss, addr);
                    if (! ok)
                    {
                        throw new EMULATED_DEVICE::EMU_ERROR("Out of bounds ST in kernel caught.");
                    }
                }
                break;
        }

        switch (type)
        {
            case K_U8:
                d->u8 = s->u8;
                break;
            case K_U16:
                d->u16 = s->u16;
                break;
            case K_U32:
                d->u32 = s->u32;
                break;
            case K_U64:
                d->u64 = s->u64;
                break;
            case K_S8:
                d->s8 = s->s8;
                break;
            case K_S16:
                d->s16 = s->s16;
                break;
            case K_S32:
                d->s32 = s->s32;
                break;
            case K_S64:
                d->s64 = s->s64;
                break;
            case K_B8:
                d->b8 = s->b8;
                break;
            case K_B16:
                d->b16 = s->b16;
                break;
            case K_B32:
                d->b32 = s->b32;
                break;
            case K_B64:
                d->b64 = s->b64;
                break;
            case K_F32:
                d->f32 = s->f32;
                break;
            case K_F64:
                d->f64 = s->f64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected ST instruction");
        }

        addr = addr + this->device->Sizeof(type);
    }

    return 0;
}

int THREAD::DoSub(TREE * inst)
{
    int start = 0;
    if (inst->GetChild(start)->GetType() == TREE_PRED)
        start++;
    assert(inst->GetChild(start)->GetType() == KI_SUB);
    start++;
    TREE * ttype = 0;
    TREE * odst = 0;
    TREE * osrc1 = 0;
    TREE * osrc2 = 0;
    for (;; ++start)
    {
        TREE * t = inst->GetChild(start);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == TREE_TYPE)
            ttype = t;
        else if (gt == TREE_OPR)
        {
            if (odst == 0)
            {
                odst = t;
            } else if (osrc1 == 0)
            {
                osrc1 = t;
            } else if (osrc2 == 0)
            {
                osrc2 = t;
            }
            else
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SUB instruction");
        }
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SUB instruction");
    }
    assert(ttype != 0);
    assert(odst != 0);
    assert(osrc1 != 0);
    assert(osrc2 != 0);
    bool sat = false;
    bool cc = false;
    int rnd = 0;
    bool ftz = false;
    for (int i = 0; ; ++i)
    {
        TREE * t = ttype->GetChild(i);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == K_SAT)
            sat = true;
        else if (gt == K_U16 || gt == K_U32 || gt == K_U64
                 || gt == K_S16 || gt == K_S32 || gt == K_S64)
            ttype = t;
        else if (gt == K_F32 || gt == K_F64)
            ttype = t;
        else if (gt == K_CC)
            cc = true;
        else if (gt == K_RN || gt == K_RZ || gt == K_RM || gt == K_RP)
            rnd = gt;
        else if (gt == K_FTZ)
            ftz = true;
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SUB instruction");
    }
    assert(ttype != 0);
    int type = ttype->GetType();
    TREE * dst = odst->GetChild(0);
    TREE * src1 = osrc1->GetChild(0);
    TREE * src2 = osrc2->GetChild(0);

    SYMBOL * sdst = 0;
    if (dst->GetType() == T_WORD)
    {
        sdst = this->symbol_table->FindSymbol(dst->GetText());
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SUB instruction");

    TYPES::Types value1;
    TYPES::Types value2;
    TYPES::Types value3;
    char * dummy;
    TYPES::Types * d = (TYPES::Types*)sdst->pvalue;
    TYPES::Types * s1 = &value1;
    TYPES::Types * s2 = &value2;
    // used for carry out calculation.
    TYPES::Types * temp = &value3;

    if (src1->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src1->GetChild( 0));
        switch (type)
        {
            case K_U16:
                s1->u16 = c.value.u16;
                break;
            case K_S16:
                s1->s16 = c.value.s16;
                break;
            case K_U32:
                s1->u32 = c.value.u32;
                break;
            case K_S32:
                s1->s32 = c.value.s32;
                break;
            case K_U64:
                s1->u64 = c.value.u64;
                break;
            case K_S64:
                s1->s64 = c.value.s64;
                break;
            case K_F32:
                s1->f32 = c.value.f32;
                break;
            case K_F64:
                s1->f64 = c.value.f64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SUB instruction");
        }
    } else if (src1->GetType() == T_WORD)
    {
        SYMBOL * ssrc1 = this->symbol_table->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        assert(ssrc1->size == this->device->Sizeof(type));
        TYPES::Types * psrc1_value = (TYPES::Types*)ssrc1->pvalue;
        switch (type)
        {
            case K_U16:
                s1->u16 = psrc1_value->u16;
                break;
            case K_S16:
                s1->s16 = psrc1_value->s16;
                break;
            case K_U32:
                s1->u32 = psrc1_value->u32;
                break;
            case K_S32:
                s1->s32 = psrc1_value->s32;
                break;
            case K_U64:
                s1->u64 = psrc1_value->u64;
                break;
            case K_S64:
                s1->s64 = psrc1_value->s64;
                break;
            case K_F32:
                s1->f32 = psrc1_value->f32;
                break;
            case K_F64:
                s1->f64 = psrc1_value->f64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SUB instruction");
        }
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SUB instruction");

    if (src2->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src2->GetChild(0));
        switch (type)
        {
            case K_U16:
                s2->u16 = c.value.u16;
                break;
            case K_S16:
                s2->s16 = c.value.s16;
                break;
            case K_U32:
                s2->u32 = c.value.u32;
                break;
            case K_S32:
                s2->s32 = c.value.s32;
                break;
            case K_U64:
                s2->u64 = c.value.u64;
                break;
            case K_S64:
                s2->s64 = c.value.s64;
                break;
            case K_F32:
                s1->f32 = c.value.f32;
                break;
            case K_F64:
                s1->f64 = c.value.f64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SUB instruction");
        }
    } else if (src2->GetType() == T_WORD)
    {
        SYMBOL * ssrc2 = this->symbol_table->FindSymbol(src2->GetText());
        assert(ssrc2 != 0);
        assert(ssrc2->size == this->device->Sizeof(type));
        TYPES::Types * psrc2_value = (TYPES::Types*)ssrc2->pvalue;
        switch (type)
        {
            case K_U16:
                s2->u16 = psrc2_value->u16;
                break;
            case K_S16:
                s2->s16 = psrc2_value->s16;
                break;
            case K_U32:
                s2->u32 = psrc2_value->u32;
                break;
            case K_S32:
                s2->s32 = psrc2_value->s32;
                break;
            case K_U64:
                s2->u64 = psrc2_value->u64;
                break;
            case K_S64:
                s2->s64 = psrc2_value->s64;
                break;
            case K_F32:
                s2->f32 = psrc2_value->f32;
                break;
            case K_F64:
                s2->f64 = psrc2_value->f64;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SUB instruction");
        }
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SUB instruction");

    switch (type)
    {
        case K_U16:
            temp->u32 = s1->u16 - (unsigned __int32)s2->u16;
            if (temp->u32 >> 16)
                this->carry = 1;
            else
                this->carry = 0;
            d->u16 = temp->u32;
            break;
        case K_S16:
            temp->s32 = s1->s16 - (signed __int32)s2->s16;
            if ((temp->s32 >> 16) && 0xffff)
                this->carry = 1;
            else
                this->carry = 0;
            d->s16 = temp->s32;
            break;
        case K_U32:
            temp->u64 = s1->u32 - (unsigned __int64)s2->u32;
            if ((temp->u64 >> 32) && 0xffffffff)
                this->carry = 1;
            else
                this->carry = 0;
            d->u32 = temp->u64;
            break;
        case K_S32:
            temp->s64 = s1->s32 - (signed __int64)s2->s32;
            if ( temp->s64 > (__int64)0x7fffffff
                 || temp->s64 < (__int64)0xffffffff80000000)
            {
                if (sat && temp->s64 > (__int64)0x7fffffff)
                {
                    temp->s64 = (__int64)0x7fffffff;
                }
                else if (sat && temp->s64 < (__int64)0xffffffff80000000)
                {
                    temp->s64 = (__int64)0xffffffff80000000;
                }
                this->carry = 1;
            }
            else
            {
                this->carry = 0;
            }
            d->s32 = temp->s64;
            break;
        case K_S64:
            d->s64 = s1->s64 - s2->s64;
            break;
        case K_U64:
            d->u64 = s1->u64 - s2->u64;
            break;
        case K_F32:
            temp->f64 = s1->f32 - (double)s2->f32;
            // Round.
            switch (rnd)
            {
                case K_RN:
                case K_RZ:
                case K_RM:
                    d->f32 = temp->f64;
                    break;
                case K_RP:
                    // test low bits of mantissa, round up.
                    if (temp->b64 & 0x00000000ffffffff)
                        temp->b64 |= 0x0000000100000000;
                    d->f32 = temp->f64;
                    break;
                default:
                    d->f32 = temp->f64;
                    break;
            }
            if (sat)
            {
                if (d->f32 > 1.0)
                    d->f32 = 1.0;
                else if (d->f32 < 0.0)
                    d->f32 = 0.0;
            }
            break;
        case K_F64:
            d->f64 = s1->f64 - s2->f64;
            if (sat)
            {
                if (d->f64 > 1.0)
                    d->f64 = 1.0;
                else if (d->f64 < 0.0)
                    d->f64 = 0.0;
            }
            break;
        default:
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected SUB instruction");
    }
    return 0;
}

int THREAD::DoSubc(TREE * inst)
{
    throw new EMULATED_DEVICE::EMU_ERROR("SUBC unimplemented");
}

int THREAD::DoSuld(TREE * inst)
{
    throw new EMULATED_DEVICE::EMU_ERROR("SULD unimplemented");
}

int THREAD::DoSuq(TREE * inst)
{
    throw new EMULATED_DEVICE::EMU_ERROR("SUQ unimplemented");
}

int THREAD::DoSured(TREE * inst)
{
    throw new EMULATED_DEVICE::EMU_ERROR("SURED unimplemented");
}

int THREAD::DoSust(TREE * inst)
{
    throw new EMULATED_DEVICE::EMU_ERROR("SUST unimplemented");
}

int THREAD::DoTestp(TREE * inst)
{
    throw new EMULATED_DEVICE::EMU_ERROR("TESTP unimplemented");
}

// Compute the sum of the number of bits for each of x, y, z, and w,
// depending on "component".  If component is zero, then return just x.
// If component is 1, then return x + y. If component is 2, then return
// x + y + z. etc.  If component is 4 or larger, or negative, throw an
// error.
int prefix_sum_channel_size(struct cudaChannelFormatDesc * desc, int component)
{
    if (component == 0)
        return 0;
    else if (component == 1)
        return desc->x;
    else if (component == 2)
        return desc->x + desc->y;
    else if (component == 3)
        return desc->x + desc->y + desc->z;
    return 0;
}

int total_sum_channel_size(struct cudaChannelFormatDesc * desc)
{
    return desc->x + desc->y + desc->z + desc->w;
}

int THREAD::DoTex(TREE * inst)
{
    int start = 0;
    if (inst->GetChild(start)->GetType() == TREE_PRED)
        start++;
    assert(inst->GetChild(start)->GetType() == KI_TEX);
    start++;
    TREE * ttype = 0;
    TREE * odst1 = 0;
    TREE * odst2 = 0;
    TREE * osrc1 = 0;
    TREE * osrc2 = 0;
    TREE * osrc3 = 0;
    for (;; ++start)
    {
        TREE * t = inst->GetChild(start);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == TREE_TYPE)
            ttype = t;
        else if (gt == TREE_OPR)
        {
            if (odst1 == 0)
            {
                odst1 = t;
            } else if (osrc1 == 0)
            {
                osrc1 = t;
            } else if (osrc2 == 0)
            {
                osrc2 = t;
            } else if (osrc3 == 0)
            {
                osrc3 = t;
            }
            else
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected TEX instruction");
        } else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected TEX instruction");
    }
    assert(ttype != 0);
    assert(odst1 != 0);
    assert(osrc1 != 0);
    assert(osrc2 != 0);

    // third source not supported yet.
    assert(osrc3 == 0);

    bool _1d = false;
    bool _2d = false;
    bool _3d = false;
    int vec = 0;
    int stype = 0;
    int dtype = 0;
    for (int i = 0; ; ++i)
    {
        TREE * t = ttype->GetChild(i);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == K_1D)
            _1d = true;
        else if (gt == K_2D)
            _2d = true;
        else if (gt == K_3D)
            _3d = true;
        else if (gt == K_V4)
            vec = gt;
        else if (dtype == 0)
        {
            if (gt == K_U32 || gt == K_S32 || gt == K_F32)
                dtype = gt;
            else
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected TEX instruction");
        }
        else if (stype == 0)
        {
            if (gt == K_U32 || gt == K_S32 || gt == K_F32)
                stype = gt;
            else
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected TEX instruction");
        }
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected TEX instruction");
    }
    assert(stype != 0);
    assert(dtype != 0);

    int type = ttype->GetType();
    TREE * src1 = osrc1->GetChild(0);
    TREE * src2 = osrc2->GetChild(0);

    // Look up symbol dimensions.
    SYMBOL * tex = 0;
    if (src1->GetType() == T_WORD)
    {
        tex = this->symbol_table->FindSymbol(src1->GetText());
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected TEX instruction");
    // Get details of the texture.
    assert(tex != 0);
    

    // Find texture from name.
    std::map<char*, TEXTURE*, ltstr>::iterator i = this->device->texturename_to_texture.find(tex->name);
    assert(i != this->device->texturename_to_texture.end());
    TEXTURE * texture = i->second;

    int times = 1;
    if (vec == K_V4)
        times = 4;

    // Find texture binding from texture.
    // There are going to be only one of two possible ways:
    // (1) linear device memory
    // (2) CUDA Array.
    std::map<void*, TEXREF*>::iterator j = this->device->texture_to_device_memory_binding.find(texture->hostVar);
    if (j != this->device->texture_to_device_memory_binding.end())
    {
        // Linear device memory.
        TEXREF * texture_binding = j->second;

        TYPES::Types * s;

        if (_1d)
        {
            // Compute actual source.
            TYPES::Types value;
            SYMBOL * ssrc2 = this->symbol_table->FindSymbol(src2->GetText());
            assert(ssrc2 != 0);
            switch (ssrc2->storage_class)
            {
                case K_GLOBAL:
                case K_LOCAL:
                case K_PARAM:
                case K_SHARED:
                case K_CONST:
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected TEX instruction");
                    break;
                case K_REG:
                    // names in instructions refer to the contents of the
                    // register.
                    s = (TYPES::Types*)ssrc2->pvalue;
                    break;
            }

            // Get value of register.
            float findex = 0;
            switch (stype)
            {
                case K_S32:
                    findex = s->s32;
                    break;
                case K_F32:
                    findex = s->f32;
                    break;
                default:
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected TEX instruction");
            }

            for (int i = 0; i < times; ++i)
            {
                s = (TYPES::Types*)(texture_binding->devPtr);
                unsigned char * addr = 0;
                addr = ((unsigned char*)s);
                int index = (int) findex;

                s = (TYPES::Types*)(addr
                                    + prefix_sum_channel_size(&texture_binding->texref->channelDesc, i) / 8
                                    + index * (total_sum_channel_size(&texture_binding->texref->channelDesc) / 8));

                TREE * dst1 = odst1->GetChild(i);
                SYMBOL * sdst = 0;
                if (dst1->GetType() == T_WORD)
                {
                    sdst = this->symbol_table->FindSymbol(dst1->GetText());
                }
                else
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected TEX instruction");
                TYPES::Types * d = (TYPES::Types*)sdst->pvalue;

                switch (dtype)
                {
                    case K_U32:
                        d->u32 = s->u32;
                        break;
                    case K_S32:
                        d->s32 = s->s32;
                        break;
                    case K_F32:
                        d->f32 = s->f32;
                        break;
                    default:
                        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected TEX instruction");
                }
            }
        } else  if (_2d)
        {
            float findex_x;
            float findex_y;
            TYPES::Types value;
            SYMBOL * ssrca2 = this->symbol_table->FindSymbol(osrc2->GetChild(0)->GetText());
            SYMBOL * ssrcb2 = this->symbol_table->FindSymbol(osrc2->GetChild(1)->GetText());
            assert(ssrca2 != 0);
            assert(ssrcb2 != 0);
            switch (ssrca2->storage_class)
            {
                case K_GLOBAL:
                case K_LOCAL:
                case K_PARAM:
                case K_SHARED:
                case K_CONST:
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected TEX instruction");
                    break;
                case K_REG:
                    // names in instructions refer to the contents of the
                    // register.
                    s = (TYPES::Types*)ssrca2->pvalue;
                    break;
            }

            // Get value of register.
            float findex = 0;
            switch (stype)
            {
                case K_S32:
                    findex_x = s->s32;
                    break;
                case K_F32:
                    findex_x = s->f32;
                    break;
                default:
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected TEX instruction");
            }

            switch (ssrcb2->storage_class)
            {
                case K_GLOBAL:
                case K_LOCAL:
                case K_PARAM:
                case K_SHARED:
                case K_CONST:
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected TEX instruction");
                    break;
                case K_REG:
                    // names in instructions refer to the contents of the
                    // register.
                    s = (TYPES::Types*)ssrcb2->pvalue;
                    break;
            }

            switch (stype)
            {
                case K_S32:
                    findex_y = s->s32;
                    break;
                case K_F32:
                    findex_y = s->f32;
                    break;
                default:
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected TEX instruction");
            }

            float increment = 0;

            int ix1 = (int)floor(findex_x - increment);
            int ix2 = ix1 + 1;
            int iy1 = (int)floor(findex_y - increment);
            int iy2 = iy1 + 1;
            if (ix1 < 0)
            {
                ix1 = 0;
            }
            if (ix2 >= texture_binding->width)
            {
                ix2 = texture_binding->width - 1;
                ix1 = ix2;
            }
            if (iy1 < 0)
            {
                iy1 = 0;
            }
            if (iy2 >= texture_binding->height)
            {
                iy2 = texture_binding->height - 1;
                iy1 = iy2;
            }

            for (int i = 0; i < times; ++i)
            {
                s = (TYPES::Types*)(texture_binding->devPtr);
                unsigned char * addr = 0;
                addr = ((unsigned char*)s);

                // Linear memory, so do not interpolate. So, we only need one value to fetch.
                int index =  ix1 + iy1 * texture_binding->width;

                s = (TYPES::Types*)(addr
                                    + prefix_sum_channel_size(&texture_binding->texref->channelDesc, i) / 8
                                    + index * (total_sum_channel_size(&texture_binding->texref->channelDesc) / 8));

                TREE * dst1 = odst1->GetChild(i);
                SYMBOL * sdst = 0;
                if (dst1->GetType() == T_WORD)
                {
                    sdst = this->symbol_table->FindSymbol(dst1->GetText());
                }
                else
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected TEX instruction");
                TYPES::Types * d = (TYPES::Types*)sdst->pvalue;

                switch (dtype)
                {
                    case K_U32:
                        d->u32 = s->u32;
                        break;
                    case K_S32:
                        d->s32 = s->s32;
                        break;
                    case K_F32:
                        d->f32 = s->f32;
                        break;
                    default:
                        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected TEX instruction");
                }
            }
        }
    }
    else {
        // Array bound texture.  Perform filtering.
        std::map<void*, TEXARR*>::iterator k = this->device->texture_to_array_binding.find(texture->hostVar);
        assert(k != this->device->texture_to_array_binding.end());

        // Texture bound to array.
        TEXARR * texture_binding = k->second;
        cudaArray * cudaArr = texture_binding->array;
        ARRAY * arr = (ARRAY*)cudaArr;
        CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();

        TYPES::Types * s;
        TYPES::Types value;

        if (_1d)
        {
            // Compute actual source.
            TYPES::Types value;
            SYMBOL * ssrc2 = this->symbol_table->FindSymbol(src2->GetText());
            assert(ssrc2 != 0);
            switch (ssrc2->storage_class)
            {
                case K_GLOBAL:
                case K_LOCAL:
                case K_PARAM:
                case K_SHARED:
                case K_CONST:
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected TEX instruction");
                    break;
                case K_REG:
                    // names in instructions refer to the contents of the
                    // register.
                    s = (TYPES::Types*)ssrc2->pvalue;
                    break;
            }

            // Get value of register.
            float findex = 0;
            switch (stype)
            {
                case K_S32:
                    findex = s->s32;
                    break;
                case K_F32:
                    findex = s->f32;
                    break;
                default:
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected TEX instruction");
            }

            for (int i = 0; i < times; ++i)
            {
                s = (TYPES::Types*)(arr->Memory() + cu->padding_size);
                unsigned char * addr = 0;
                addr = ((unsigned char*)s);

                TREE * dst1 = odst1->GetChild(i);

                // Add value to texture binding address and dereference to get and assign value.
                SYMBOL * sdst = 0;
                if (dst1->GetType() == T_WORD)
                {
                    sdst = this->symbol_table->FindSymbol(dst1->GetText());
                }
                else
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected TEX instruction");
                TYPES::Types * d = (TYPES::Types*)sdst->pvalue;

                if (texture->hostVar->filterMode & cudaFilterModeLinear)
                {
                    // Perform interpolation.
                    // findex will be between floor(findex) and ceil(findex).  However, before grabbing the values
                    // at these two indices, perform index range checking.
                    int i1 = (int)floor(findex - 0.5);
                    int i2 = i1 + 1;
                    if (i1 < 0)
                    {
                        i1 = 0;
                        i2 = 0;
                    }
                    if (i1+1 >= arr->Width())
                    {
                        i1 = arr->Width() - 1;
                        i2 = i1;
                    }

                    s = (TYPES::Types*)(addr
                        + prefix_sum_channel_size(&texture_binding->texref->channelDesc, i) / 8
                                        + i1 * (total_sum_channel_size(&texture_binding->texref->channelDesc) / 8));
                    
                    TYPES::Types* s1 = s;
                    TYPES::Types* s2 = (TYPES::Types*)(addr
                        + prefix_sum_channel_size(&texture_binding->texref->channelDesc, i) / 8
                        + i2 * (total_sum_channel_size(&texture_binding->texref->channelDesc) / 8));

                    switch (dtype)
                    {
                        case K_U32:
                            d->u32 = s->s32;
                            break;
                        case K_S32:
                            d->s32 = s->s32;
                            break;
                        case K_F32:
                            d->f32 = s1->f32 + (s2->f32 - s1->f32) * (findex - 0.5 - i1);
                            break;
                        default:
                            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected TEX instruction");
                    }

                }
                else {
                    int index = (int) findex;
                    s = (TYPES::Types*)(addr + index * (this->device->Sizeof(dtype)));

                    switch (dtype)
                    {
                        case K_U32:
                            d->u32 = s->u32;
                            break;
                        case K_S32:
                            d->s32 = s->s32;
                            break;
                        case K_F32:
                            d->f32 = s->f32;
                            break;
                        default:
                            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected TEX instruction");
                    }
                }
            }
        } else  if (_2d)
        {
            float findex_x;
            float findex_y;
            // Compute actual source.
            TYPES::Types value;
            SYMBOL * ssrca2 = this->symbol_table->FindSymbol(osrc2->GetChild(0)->GetText());
            SYMBOL * ssrcb2 = this->symbol_table->FindSymbol(osrc2->GetChild(1)->GetText());
            assert(ssrca2 != 0);
            assert(ssrcb2 != 0);
            switch (ssrca2->storage_class)
            {
                case K_GLOBAL:
                case K_LOCAL:
                case K_PARAM:
                case K_SHARED:
                case K_CONST:
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected TEX instruction");
                    break;
                case K_REG:
                    // names in instructions refer to the contents of the
                    // register.
                    s = (TYPES::Types*)ssrca2->pvalue;
                    break;
            }

            // Get value of register.
            float findex = 0;
            switch (stype)
            {
                case K_S32:
                    findex_x = s->s32;
                    break;
                case K_F32:
                    findex_x = s->f32;
                    break;
                default:
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected TEX instruction");
            }

            switch (ssrcb2->storage_class)
            {
                case K_GLOBAL:
                case K_LOCAL:
                case K_PARAM:
                case K_SHARED:
                case K_CONST:
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected TEX instruction");
                    break;
                case K_REG:
                    // names in instructions refer to the contents of the
                    // register.
                    s = (TYPES::Types*)ssrcb2->pvalue;
                    break;
            }

            switch (stype)
            {
                case K_S32:
                    findex_y = s->s32;
                    break;
                case K_F32:
                    findex_y = s->f32;
                    break;
                default:
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected TEX instruction");
            }

            float increment = (texture->hostVar->filterMode & cudaFilterModeLinear && dtype == K_F32) ? 0.5 : 0;
            float r_findex_x = findex_x - increment;
            float r_findex_y = findex_y - increment;
            if (r_findex_x < 0)
                r_findex_x = 0;
            if (r_findex_y < 0)
                r_findex_y = 0;

            int ix1 = (int)floor(r_findex_x);
            int ix2 = ix1 + 1;
            int iy1 = (int)floor(r_findex_y);
            int iy2 = iy1 + 1;
            if (ix1 < 0)
            {
                ix1 = 0;
            }
            if (ix2 >= arr->Width())
            {
                ix2 = arr->Width() - 1;
                ix1 = ix2;
            }
            if (iy1 < 0)
            {
                iy1 = 0;
            }
            if (iy2 >= arr->Height())
            {
                iy2 = arr->Height() - 1;
                iy1 = iy2;
            }

            for (int i = 0; i < times; ++i)
            {
                s = (TYPES::Types*)(arr->Memory() + cu->padding_size);
                unsigned char * addr = 0;
                addr = ((unsigned char*)s);

                TREE * dst1 = odst1->GetChild(i);

                SYMBOL * sdst = 0;
                if (dst1->GetType() == T_WORD)
                {
                    sdst = this->symbol_table->FindSymbol(dst1->GetText());
                }
                else
                    throw new EMULATED_DEVICE::EMU_ERROR("Unexpected TEX instruction");
                TYPES::Types * d = (TYPES::Types*)sdst->pvalue;

                if (texture->hostVar->filterMode & cudaFilterModeLinear && dtype == K_F32)
                {
                    int index_q11 = ix1 + arr->Width() * iy1;
                    int index_q12 = ix1 + arr->Width() * iy2;
                    int index_q21 = ix2 + arr->Width() * iy1;
                    int index_q22 = ix2 + arr->Width() * iy2;

                    TYPES::Types* q11 = (TYPES::Types*)(addr
                        + prefix_sum_channel_size(&texture_binding->texref->channelDesc, i) / 8
                        + index_q11 * (total_sum_channel_size(&texture_binding->texref->channelDesc) / 8));

                    TYPES::Types* q12 = (TYPES::Types*)(addr
                        + prefix_sum_channel_size(&texture_binding->texref->channelDesc, i) / 8
                        + index_q12 * (total_sum_channel_size(&texture_binding->texref->channelDesc) / 8));

                    TYPES::Types* q21 = (TYPES::Types*)(addr
                        + prefix_sum_channel_size(&texture_binding->texref->channelDesc, i) / 8
                        + index_q21 * (total_sum_channel_size(&texture_binding->texref->channelDesc) / 8));

                    TYPES::Types* q22 = (TYPES::Types*)(addr
                        + prefix_sum_channel_size(&texture_binding->texref->channelDesc, i) / 8
                        + index_q22 * (total_sum_channel_size(&texture_binding->texref->channelDesc) / 8));

                    // ix1 - ix2, as well as iy1 - iy2, can never be zero.  Basica assumption in binear
                    // interpolation.  In this case, create adjusted x and y coordinates so that it looks
                    // like a flat function.

                    if (ix1 - ix2 == 0)
                    {
                        ix2 = ix1 + 1;
                    }
                    if (iy1 - iy2 == 0)
                    {
                        iy2 = iy1 + 1;
                    }

                    float p1 = q11->f32 * (ix2 - r_findex_x) * (iy2 - r_findex_y);
                    float p2 = q21->f32 * (r_findex_x - ix1) * (iy2 - r_findex_y);
                    float p3 = q12->f32 * (ix2 - r_findex_x) * (r_findex_y - iy1);
                    float p4 = q22->f32 * (r_findex_x - ix1) * (r_findex_y - iy1);

                    d->f32 = p1 + p2 + p3 + p4;
                }
                else
                {
                    // Linear memory, so do not interpolate. So, we only need one value to fetch.
                    int index = ix1 + arr->Width() * iy1;

                    s = (TYPES::Types*)(addr
                        + prefix_sum_channel_size(&texture_binding->texref->channelDesc, i) / 8
                        + index * (total_sum_channel_size(&texture_binding->texref->channelDesc) / 8));

                    switch (dtype)
                    {
                        case K_U32:
                            d->u32 = s->u32;
                            break;
                        case K_S32:
                            d->s32 = s->s32;
                            break;
                        case K_F32:
                            d->f32 = s->f32;
                            break;
                        default:
                            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected TEX instruction");
                    }
                }
            }
        }
    }
    return 0;
}

int THREAD::DoTrap(TREE * inst)
{
    throw new EMULATED_DEVICE::EMU_ERROR("TRAP unimplemented");
}

int THREAD::DoTxq(TREE * inst)
{
    throw new EMULATED_DEVICE::EMU_ERROR("TXQ unimplemented");
}

int THREAD::DoVabsdiff(TREE * inst)
{
    throw new EMULATED_DEVICE::EMU_ERROR("VABSDIFF unimplemented");
}

int THREAD::DoVadd(TREE * inst)
{
    throw new EMULATED_DEVICE::EMU_ERROR("Vadd unimplemented");
}

int THREAD::DoVmad(TREE * inst)
{
    throw new EMULATED_DEVICE::EMU_ERROR("VMAD unimplemented");
}

int THREAD::DoVmax(TREE * inst)
{
    throw new EMULATED_DEVICE::EMU_ERROR("VMAX unimplemented");
}

int THREAD::DoVmin(TREE * inst)
{
    throw new EMULATED_DEVICE::EMU_ERROR("VMIN unimplemented");
}

int THREAD::DoVote(TREE * inst)
{
    throw new EMULATED_DEVICE::EMU_ERROR("VOTE unimplemented");
}

int THREAD::DoVset(TREE * inst)
{
    throw new EMULATED_DEVICE::EMU_ERROR("VSET unimplemented");
}

int THREAD::DoVshl(TREE * inst)
{
    throw new EMULATED_DEVICE::EMU_ERROR("VSHL unimplemented");
}

int THREAD::DoVshr(TREE * inst)
{
    throw new EMULATED_DEVICE::EMU_ERROR("VSHR unimplemented");
}

int THREAD::DoVsub(TREE * inst)
{
    throw new EMULATED_DEVICE::EMU_ERROR("VSUB unimplemented");
}

int THREAD::DoXor(TREE * inst)
{
    int start = 0;
    if (inst->GetChild(start)->GetType() == TREE_PRED)
        start++;
    assert(inst->GetChild(start)->GetType() == KI_XOR);
    start++;
    TREE * ttype = 0;
    TREE * odst = 0;
    TREE * osrc1 = 0;
    TREE * osrc2 = 0;
    for (;; ++start)
    {
        TREE * t = inst->GetChild(start);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == TREE_TYPE)
            ttype = t;
        else if (gt == TREE_OPR)
        {
            if (odst == 0)
            {
                odst = t;
            } else if (osrc1 == 0)
            {
                osrc1 = t;
            } else if (osrc2 == 0)
            {
                osrc2 = t;
            }
            else
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected XOR instruction");
        }
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected XOR instruction");
    }
    assert(ttype != 0);
    assert(odst != 0);
    assert(osrc1 != 0);
    assert(osrc2 != 0);
    int type = 0;
    for (int i = 0; ; ++i)
    {
        TREE * t = ttype->GetChild(i);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == K_PRED || gt == K_B16 || gt == K_B32 || gt == K_B64)
            type = gt;
        else
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected XOR instruction");
    }
    assert(type != 0);
    TREE * dst = odst->GetChild(0);
    TREE * src1 = osrc1->GetChild(0);
    TREE * src2 = osrc2->GetChild(0);

    SYMBOL * sdst = 0;
    if (dst->GetType() == T_WORD)
    {
        sdst = this->symbol_table->FindSymbol(dst->GetText());
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected XOR instruction");

    TYPES::Types value1;
    TYPES::Types value2;
    char * dummy;
    TYPES::Types * d = (TYPES::Types*)sdst->pvalue;
    TYPES::Types * s1 = &value1;
    TYPES::Types * s2 = &value2;

    if (src1->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src1->GetChild( 0));
        switch (type)
        {
            case K_B16:
                s1->b16 = c.value.b16;
                break;
            case K_B32:
                s1->b32 = c.value.b32;
                break;
            case K_B64:
                s1->b64 = c.value.b64;
                break;
            case K_PRED:
                s1->pred = c.value.pred;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected XOR instruction");
        }
    } else if (src1->GetType() == T_WORD)
    {
        SYMBOL * ssrc1 = this->symbol_table->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        assert(ssrc1->size == this->device->Sizeof(type));
        TYPES::Types * psrc1_value = (TYPES::Types*)ssrc1->pvalue;
        switch (type)
        {
            case K_B16:
                s1->b16 = psrc1_value->b16;
                break;
            case K_B32:
                s1->b32 = psrc1_value->b32;
                break;
            case K_B64:
                s1->b64 = psrc1_value->b64;
                break;
            case K_PRED:
                s1->pred = psrc1_value->pred;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected XOR instruction");
        }
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected XOR instruction");

    if (src2->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->device->Eval(type, src2->GetChild(0));
        switch (type)
        {
            case K_B16:
                s2->b16 = c.value.b16;
                break;
            case K_B32:
                s2->b32 = c.value.b32;
                break;
            case K_B64:
                s2->b64 = c.value.b64;
                break;
            case K_PRED:
                s2->pred = c.value.pred;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected XOR instruction");
        }
    } else if (src2->GetType() == T_WORD)
    {
        SYMBOL * ssrc2 = this->symbol_table->FindSymbol(src2->GetText());
        assert(ssrc2 != 0);
        assert(ssrc2->size == this->device->Sizeof(type));
        TYPES::Types * psrc2_value = (TYPES::Types*)ssrc2->pvalue;
        switch (type)
        {
            case K_B16:
                s2->b16 = psrc2_value->b16;
                break;
            case K_B32:
                s2->b32 = psrc2_value->b32;
                break;
            case K_B64:
                s2->b64 = psrc2_value->b64;
                break;
            case K_PRED:
                s2->pred = psrc2_value->pred;
                break;
            default:
                throw new EMULATED_DEVICE::EMU_ERROR("Unexpected XOR instruction");
        }
    }
    else
        throw new EMULATED_DEVICE::EMU_ERROR("Unexpected XOR instruction");

    switch (type)
    {
        case K_B16:
            d->b16 = s1->b16 ^ s2->b16;
            break;
        case K_B32:
            d->b32 = s1->b32 ^ s2->b32;
            break;
        case K_S64:
            d->b64 = s1->b64 ^ s2->b64;
            break;
        case K_PRED:
            d->pred = s1->pred ^ s2->pred;
            break;
        default:
            throw new EMULATED_DEVICE::EMU_ERROR("Unexpected XOR instruction");
    }
    return 0;
}
