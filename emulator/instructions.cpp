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
#include "emulator.h"
#include "thread.h"
#include "symbol.h"
#include "symbol-table.h"
#include "constant.h"
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
            } else assert(false);
        } else assert(false);
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
        else assert(false);
    }
    assert(ttype != 0);
    this->emulator->unimplemented(ftz, "ABS.ftz not implemented.");

    int type = ttype->GetType();
    TREE * dst = odst->GetChild(0);
    TREE * src1 = osrc1->GetChild(0);

    SYMBOL * sdst = 0;
    if (dst->GetType() == T_WORD)
    {
        sdst = this->root->FindSymbol(dst->GetText());
    } else assert(false);

    TYPES::Types value1;
    char * dummy;
    TYPES::Types * d = (TYPES::Types*)sdst->pvalue;
    TYPES::Types * s1 = &value1;

    if (src1->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->emulator->Eval(type, src1->GetChild(0));
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
                assert(false);
        }
    } else if (src1->GetType() == T_WORD)
    {
        SYMBOL * ssrc1 = this->root->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        assert(ssrc1->size == this->emulator->Sizeof(type));
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
                assert(false);
        }
    } else assert(false);

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
            assert(false);
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
            } else assert(false);
        } else assert(false);
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
        else assert(false);
    }
    assert(ttype != 0);
    this->emulator->unimplemented(ftz, "ADD.ftz not implemented.");
    int type = ttype->GetType();
    TREE * dst = odst->GetChild(0);
    TREE * src1 = osrc1->GetChild(0);
    TREE * src2 = osrc2->GetChild(0);

    SYMBOL * sdst = 0;
    if (dst->GetType() == T_WORD)
    {
        sdst = this->root->FindSymbol(dst->GetText());
    } else assert(false);

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
        CONSTANT c = this->emulator->Eval(type, src1->GetChild(0));
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
                assert(false);
        }
    } else if (src1->GetType() == T_WORD)
    {
        SYMBOL * ssrc1 = this->root->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        assert(ssrc1->size == this->emulator->Sizeof(type));
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
                assert(false);
        }
    } else assert(false);

    if (src2->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->emulator->Eval(type, src2->GetChild(0));
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
                assert(false);
        }
    } else if (src2->GetType() == T_WORD)
    {
        SYMBOL * ssrc2 = this->root->FindSymbol(src2->GetText());
        assert(ssrc2 != 0);
        assert(ssrc2->size == this->emulator->Sizeof(type));
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
                assert(false);
        }
    } else assert(false);

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
            assert(false);
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
            } else assert(false);
        } else assert(false);
    }
    assert(ttype != 0);
    assert(odst != 0);
    assert(osrc1 != 0);
    assert(osrc2 != 0);
    bool cc = false;
    for (int i = 0; ; ++i)
    {
        TREE * t = ttype->GetChild(i);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == K_U32 || gt == K_S32)
            ttype = t;
        else if (gt == K_CC)
            cc = true;
        else assert(false);
    }
    assert(ttype != 0);
    int type = ttype->GetType();
    TREE * dst = odst->GetChild(0);
    TREE * src1 = osrc1->GetChild(0);
    TREE * src2 = osrc2->GetChild(0);

    SYMBOL * sdst = 0;
    if (dst->GetType() == T_WORD)
    {
        sdst = this->root->FindSymbol(dst->GetText());
    } else assert(false);

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
        CONSTANT c = this->emulator->Eval(type, src1->GetChild(0));
        switch (type)
        {
            case K_U32:
                s1->u32 = c.value.u32;
                break;
            case K_S32:
                s1->s32 = c.value.s32;
                break;
            default:
                assert(false);
        }
    } else if (src1->GetType() == T_WORD)
    {
        SYMBOL * ssrc1 = this->root->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        assert(ssrc1->size == this->emulator->Sizeof(type));
        TYPES::Types * psrc1_value = (TYPES::Types*)ssrc1->pvalue;
        switch (type)
        {
            case K_U32:
                s1->u32 = psrc1_value->u32;
                break;
            case K_S32:
                s1->s32 = psrc1_value->s32;
                break;
            default:
                assert(false);
        }
    } else assert(false);

    if (src2->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->emulator->Eval(type, src2->GetChild(0));
        switch (type)
        {
            case K_U32:
                s2->u32 = c.value.u32;
                break;
            case K_S32:
                s2->s32 = c.value.s32;
                break;
            default:
                assert(false);
        }
    } else if (src2->GetType() == T_WORD)
    {
        SYMBOL * ssrc2 = this->root->FindSymbol(src2->GetText());
        assert(ssrc2 != 0);
        assert(ssrc2->size == this->emulator->Sizeof(type));
        TYPES::Types * psrc2_value = (TYPES::Types*)ssrc2->pvalue;
        switch (type)
        {
            case K_U32:
                s2->u32 = psrc2_value->u32;
                break;
            case K_S32:
                s2->s32 = psrc2_value->s32;
                break;
            default:
                assert(false);
        }
    } else assert(false);

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
        case K_S32:
            temp->s64 = s1->s32 + (signed __int64)s2->s32 + this->carry;
            if ((temp->s64 >> 32) && 0xffffffff)
                this->carry = 1;
            else
                this->carry = 0;
            d->s32 = temp->s32;
            break;
        default:
            assert(false);
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
            } else assert(false);
        } else assert(false);
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
        else assert(false);
    }
    assert(type != 0);
    TREE * dst = odst->GetChild(0);
    TREE * src1 = osrc1->GetChild(0);
    TREE * src2 = osrc2->GetChild(0);

    SYMBOL * sdst = 0;
    if (dst->GetType() == T_WORD)
    {
        sdst = this->root->FindSymbol(dst->GetText());
    } else assert(false);

    TYPES::Types value1;
    TYPES::Types value2;
    char * dummy;
    TYPES::Types * d = (TYPES::Types*)sdst->pvalue;
    TYPES::Types * s1 = &value1;
    TYPES::Types * s2 = &value2;

    if (src1->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->emulator->Eval(type, src1->GetChild(0));
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
                assert(false);
        }
    } else if (src1->GetType() == T_WORD)
    {
        SYMBOL * ssrc1 = this->root->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        assert(ssrc1->size == this->emulator->Sizeof(type));
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
                assert(false);
        }
    } else assert(false);

    if (src2->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->emulator->Eval(type, src2->GetChild(0));
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
                assert(false);
        }
    } else if (src2->GetType() == T_WORD)
    {
        SYMBOL * ssrc2 = this->root->FindSymbol(src2->GetText());
        assert(ssrc2 != 0);
        assert(ssrc2->size == this->emulator->Sizeof(type));
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
                assert(false);
        }
    } else assert(false);

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
            assert(false);
    }
    return 0;
}

int THREAD::DoAtom(TREE * inst)
{
    throw new EMULATOR::Unimplemented("ATOM unimplemented");
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
            } else this->emulator->unimplemented(true, "BAR with two or more operands unimplemented.");
        } else assert(false);
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
    this->emulator->unimplemented(arrive, "BAR.arrive not implemented.");
    this->emulator->unimplemented(red, "BAR.red not implemented.");
    
    int type = K_U32;
    TREE * src1 = osrc1->GetChild(0);

    TYPES::Types value1;
    char * dummy;
    TYPES::Types * s1 = &value1;

    if (src1->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->emulator->Eval(type, src1->GetChild(0));
        switch (type)
        {
            case K_U32:
                s1->u32 = c.value.u32;
                break;
            default:
                assert(false);
        }
        this->emulator->unimplemented(c.value.u32 != 0, "BAR with non-zero argument not implemented.");
    } else if (src1->GetType() == T_WORD)
    {
        SYMBOL * ssrc1 = this->root->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        assert(ssrc1->size == this->emulator->Sizeof(type));
        TYPES::Types * psrc1_value = (TYPES::Types*)ssrc1->pvalue;
        switch (type)
        {
            case K_U32:
                s1->u32 = psrc1_value->u32;
                break;
            default:
                assert(false);
        }
    } else assert(false);

    this->emulator->unimplemented(s1->u32 != 0, "BAR with non-zero argument not implemented.");

    // pack up the THREAD context.  Synchronize all assumed.

    return -KI_BAR;
}

int THREAD::DoBfe(TREE * inst)
{
    throw new EMULATOR::Unimplemented("BFE unimplemented");
}

int THREAD::DoBfi(TREE * inst)
{
    throw new EMULATOR::Unimplemented("BFI unimplemented");
}

int THREAD::DoBfind(TREE * inst)
{
    throw new EMULATOR::Unimplemented("BFIND unimplemented");
}

int THREAD::DoBrev(TREE * inst)
{
    throw new EMULATOR::Unimplemented("BREV unimplemented");
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
    SYMBOL * sdst = this->root->FindSymbol(dst->GetText());
    assert (sdst != 0);
    return (int)sdst->pvalue;
}

int THREAD::DoBrkpt(TREE * inst)
{
    throw new EMULATOR::Unimplemented("BRKPT unimplemented");
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
        else assert(false);
    }

    // Call of vprintf only thing supported...
    if (strcmp(tfunc->GetText(), "vprintf") != 0)
    {
        throw new EMULATOR::Unimplemented("Only CALL of vprintf implemented.\n");
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
        else assert(false);
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
            if (u->GetType() == T_UNDERSCORE)
                once++;
            else
                throw new EMULATOR::Unimplemented("CALL non-underscore return unimplemented.");
        } else
            break;
    }
    if (once != 1)
        throw new EMULATOR::Unimplemented("CALL with multiple returns unimplemented.");

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
                SYMBOL * param = this->root->FindSymbol(u->GetText());
                assert(param != 0);
                this->emulator->unimplemented(param->storage_class != K_PARAM, "CALL indirect form unimplemented.");
                stack.push_back(param);
            }
            else
                this->emulator->unimplemented("CALL unimplemented parameter type.");
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
//            throw new EMULATOR::Unimplemented("EMULATOR::Unimplemented type for vprintf\n");
//            break;
//        }
//    }

    vprintf( (char*)(((TYPES::Types*)stack[0]->pvalue)->u32), (va_list) ((TYPES::Types*)stack[1]->pvalue)->u32);
//    va_end(out);

    return 0;
}

int THREAD::DoClz(TREE * inst)
{
    throw new EMULATOR::Unimplemented("CLZ unimplemented");
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
            } else assert(false);
        } else assert(false);
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
        else assert(false);
    }
    assert(type != 0);
    TREE * dst = odst->GetChild(0);
    TREE * src1 = osrc1->GetChild(0);

    SYMBOL * sdst = 0;
    if (dst->GetType() == T_WORD)
    {
        sdst = this->root->FindSymbol(dst->GetText());
    } else assert(false);

    TYPES::Types value1;
    char * dummy;
    TYPES::Types * d = (TYPES::Types*)sdst->pvalue;
    TYPES::Types * s1 = &value1;

    if (src1->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->emulator->Eval(type, src1->GetChild( 0));
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
                assert(false);
        }
    } else if (src1->GetType() == T_WORD)
    {
        SYMBOL * ssrc1 = this->root->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        assert(ssrc1->size == this->emulator->Sizeof(type));
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
                assert(false);
        }
    } else assert(false);

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
            assert(false);
    }
    return 0;
}

int THREAD::DoCopysign(TREE * inst)
{
    throw new EMULATOR::Unimplemented("COPYSIGN unimplemented");
}

int THREAD::DoCos(TREE * inst)
{
    throw new EMULATOR::Unimplemented("COS unimplemented");
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
            else assert(false);
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
            else assert(false);
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

    SYMBOL * s1 = this->root->FindSymbol(o1->GetChild(0)->GetText());
    assert(s1 != 0);
    SYMBOL * s2 = this->root->FindSymbol(o2->GetChild(0)->GetText());
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
        } else assert(false);
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
                    assert(false);
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
                    assert(false);
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
                    assert(false);
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
                    assert(false);
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
                    assert(false);
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
                    assert(false);
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
                    assert(false);
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
                    assert(false);
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
                    assert(false);
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
                    assert(false);
            }
            break;

        default:
            assert(false);
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
            } else assert(false);
        } else assert(false);
    }
    assert(ttype != 0);
    assert(odst != 0);
    assert(osrc != 0);
    int storage_class = 0;
    for (int i = 0; ; ++i)
    {
        TREE * t = ttype->GetChild(i);
        if (t == 0)
            break;
        int gt = t->GetType();
        if (gt == K_U32 || gt == K_U64)
            ttype = t;
        else if (gt == K_SHARED || gt == K_GLOBAL || gt == K_LOCAL)
            storage_class = gt;
        else assert(false);
    }
    assert(ttype);
    int type = ttype->GetType();
    TREE * dst = odst->GetChild( 0);
    TREE * src = osrc->GetChild(0);
    SYMBOL * sdst = 0;
    SYMBOL * ssrc = 0;
    if (dst->GetType() == T_WORD)
    {
        sdst = this->root->FindSymbol(dst->GetText());
    } else assert(false);

    TYPES::Types * d;
    TYPES::Types * s;
    TYPES::Types value;
    char * dummy;

    d = (TYPES::Types*)sdst->pvalue;

    // different semantics depending on if register, global, etc.
    if (src->GetType() == T_WORD)
    {
        ssrc = this->root->FindSymbol(src->GetText());
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
            assert(false);
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
    sdst = this->root->FindSymbol(dst->GetText());
    char * dummy;

    TYPES::Types value1; // used if literal
    TYPES::Types value2; // used if literal
    TYPES::Types * d = (TYPES::Types*)sdst->pvalue;
    TYPES::Types * s1 = &value1;
    TYPES::Types * s2 = &value2;

    if (src1->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->emulator->Eval(type, src1->GetChild( 0));
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
                assert(false);
        }
    } else if (src1->GetType() == T_WORD)
    {
        ssrc1 = this->root->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        s1 = (TYPES::Types*)ssrc1->pvalue;
    } else assert(false);

    if (src2->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->emulator->Eval(type, src2->GetChild(0));
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
                assert(false);
        }
    } else if (src2->GetType() == T_WORD)
    {
        ssrc2 = this->root->FindSymbol(src2->GetText());
        assert(ssrc2 != 0);
        s2 = (TYPES::Types*)ssrc2->pvalue;
    } else assert(false);

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
            assert(false);
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
            } else assert(false);
        } else assert(false);
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
        else assert(false);
    }
    assert(ttype != 0);
    this->emulator->unimplemented(ftz, "EX2.ftz not implemented.");

    int type = ttype->GetType();
    TREE * dst = odst->GetChild(0);
    TREE * src1 = osrc1->GetChild(0);

    SYMBOL * sdst = 0;
    if (dst->GetType() == T_WORD)
    {
        sdst = this->root->FindSymbol(dst->GetText());
    } else assert(false);

    TYPES::Types value1;
    char * dummy;
    TYPES::Types * d = (TYPES::Types*)sdst->pvalue;
    TYPES::Types * s1 = &value1;

    if (src1->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->emulator->Eval(type, src1->GetChild( 0));
        switch (type)
        {
            case K_F32:
                s1->f32 = c.value.f32;
                break;
            case K_F64:
                s1->f64 = c.value.f64;
                break;
            default:
                assert(false);
        }
    } else if (src1->GetType() == T_WORD)
    {
        SYMBOL * ssrc1 = this->root->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        assert(ssrc1->size == this->emulator->Sizeof(type));
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
                assert(false);
        }
    } else assert(false);

    switch (type)
    {
        case K_F32:
            d->f32 = pow(2, s1->f32);
            break;
        case K_F64:
            d->f64 = pow(2, s1->f64);
            break;
        default:
            assert(false);
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
    sdst = this->root->FindSymbol(dst->GetText());
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
        CONSTANT c = this->emulator->Eval(type, src1->GetChild( 0));
        switch (type)
        {
            case K_F32:
                s1->f32 = c.value.f32;
                break;
            case K_F64:
                s1->f64 = c.value.f64;
                break;
            default:
                assert(false);
        }
    } else if (src1->GetType() == T_WORD)
    {
        ssrc1 = this->root->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        s1 = (TYPES::Types*)ssrc1->pvalue;
    } else assert(false);

    if (src2->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->emulator->Eval(type, src2->GetChild(0));
        switch (type)
        {
            case K_F32:
                s2->f32 = c.value.f32;
                break;
            case K_F64:
                s2->f64 = c.value.f64;
                break;
            default:
                assert(false);
        }
    } else if (src2->GetType() == T_WORD)
    {
        ssrc2 = this->root->FindSymbol(src2->GetText());
        assert(ssrc2 != 0);
        s2 = (TYPES::Types*)ssrc2->pvalue;
    } else assert(false);

    if (src3->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->emulator->Eval(type, src3->GetChild(0));
        switch (type)
        {
            case K_F32:
                s3->f32 = c.value.f32;
                break;
            case K_F64:
                s3->f64 = c.value.f64;
                break;
            default:
                assert(false);
        }
    } else if (src3->GetType() == T_WORD)
    {
        ssrc3 = this->root->FindSymbol(src3->GetText());
        assert(ssrc3 != 0);
        s3 = (TYPES::Types*)ssrc3->pvalue;
    } else assert(false);

    switch (type)
    {
        case K_F32:
            d->f32 = s1->f32 * s2->f32 + s3->f32;
            break;
        case K_F64:
            d->f64 = s1->f64 * s2->f64 + s3->f64;
            break;
        default:
            assert(false);
    }
    return 0;
}

int THREAD::DoIsspacep(TREE * inst)
{
    throw new EMULATOR::Unimplemented("ISSPACEP unimplemented");
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
            } else assert(false);
        } else assert(false);
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
        else assert(false);
    }
    assert(ttype != 0);
    assert(cop == 0);
    int type = ttype->GetType();

    // Get two operands, assign source to destination.
    TREE * src = osrc->GetChild( 0);
    SYMBOL * ssrc = 0;
    
    assert(src->GetType() == T_WORD);
    ssrc = this->root->FindSymbol(src->GetText());
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
        value = this->emulator->Eval(K_S32, const_expr);
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
            assert(false);
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
            assert(false);
    }

    for (int i = 0; i < times; ++i)
    {
        TREE * dst = odst->GetChild( i);
        SYMBOL * sdst = 0;
        assert(dst->GetType() == T_WORD);
        sdst = this->root->FindSymbol(dst->GetText());
        assert(sdst != 0);
        TYPES::Types * d = (TYPES::Types*)sdst->pvalue;

        TYPES::Types * s = (TYPES::Types*)addr;

        // For targets that are register, we may widen to the size of the register.
        if (sdst->storage_class == K_REG)
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
                            assert(false);
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
                            assert(false);
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
                            assert(false);
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
                            assert(false);
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
                            assert(false);
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
                            assert(false);
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
                            assert(false);
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
                            assert(false);
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
                            assert(false);
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
                            assert(false);
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
                            assert(false);
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
                            assert(false);
                    }
                    break;
                case K_F32:
                    d->f32 = s->f32;
                    break;
                case K_F64:
                    d->f64 = s->f64;
                    break;
                default:
                    assert(false);
            }
        else
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
                    assert(false);
            }

        addr = addr + this->emulator->Sizeof(type);
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
            } else assert(false);
        } else assert(false);
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
        else assert(false);
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
    sdst = this->root->FindSymbol(dst->GetText());
    
    assert(src->GetType() == T_WORD);
    ssrc = this->root->FindSymbol(src->GetText());
    TREE * plus = osrc->GetChild( 1);
    CONSTANT value(0);
    if (plus != 0)
    {
        TREE * const_expr_tree = osrc->GetChild( 2);
        assert(const_expr_tree != 0);
        assert(const_expr_tree->GetType() == TREE_CONSTANT_EXPR);
        TREE * const_expr = const_expr_tree->GetChild(0);
        assert(const_expr != 0);
        value = this->emulator->Eval(K_S32, const_expr);
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
            assert(false);
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
            assert(false);
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
            } else assert(false);
        } else assert(false);
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
        else assert(false);
    }
    assert(ttype != 0);
    this->emulator->unimplemented(ftz, "LG2.ftz not implemented.");

    int type = ttype->GetType();
    TREE * dst = odst->GetChild(0);
    TREE * src1 = osrc1->GetChild(0);

    SYMBOL * sdst = 0;
    if (dst->GetType() == T_WORD)
    {
        sdst = this->root->FindSymbol(dst->GetText());
    } else assert(false);

    TYPES::Types value1;
    char * dummy;
    TYPES::Types * d = (TYPES::Types*)sdst->pvalue;
    TYPES::Types * s1 = &value1;

    if (src1->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->emulator->Eval(type, src1->GetChild( 0));
        switch (type)
        {
            case K_F32:
                s1->f32 = c.value.f32;
                break;
            case K_F64:
                s1->f64 = c.value.f64;
                break;
            default:
                assert(false);
        }
    } else if (src1->GetType() == T_WORD)
    {
        SYMBOL * ssrc1 = this->root->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        assert(ssrc1->size == this->emulator->Sizeof(type));
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
                assert(false);
        }
    } else assert(false);

    switch (type)
    {
        case K_F32:
            d->f32 = log(s1->f32) / log(2.0);
            break;
        case K_F64:
            d->f64 = log(s1->f64) / log(2.0);
            break;
        default:
            assert(false);
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
    sdst = this->root->FindSymbol(dst->GetText());
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
        CONSTANT c = this->emulator->Eval(type, src1->GetChild( 0));
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
                assert(false);
        }
    } else if (src1->GetType() == T_WORD)
    {
        ssrc1 = this->root->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        s1 = (TYPES::Types*)ssrc1->pvalue;
    } else assert(false);

    if (src2->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->emulator->Eval(type, src2->GetChild(0));
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
                assert(false);
        }
    } else if (src2->GetType() == T_WORD)
    {
        ssrc2 = this->root->FindSymbol(src2->GetText());
        assert(ssrc2 != 0);
        s2 = (TYPES::Types*)ssrc2->pvalue;
    } else assert(false);

    if (src3->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->emulator->Eval(type, src3->GetChild(0));
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
                assert(false);
        }
    } else if (src3->GetType() == T_WORD)
    {
        ssrc3 = this->root->FindSymbol(src3->GetText());
        assert(ssrc3 != 0);
        s3 = (TYPES::Types*)ssrc3->pvalue;
    } else assert(false);

    switch (type)
    {
        case K_U16:
            if (width == K_LO)
                d->u16 = s1->u16 * s2->u16 + s3->u16;
            else if (width == K_HI)
                d->u16 = (s1->u16 * s2->u16 + s3->u16) >> 16;
            else if (width == K_WIDE)
                d->u32 = s1->u16 * s2->u16 + s3->u32;
            else assert(false);
            break;
        case K_S16:
            if (width == K_LO)
                d->s16 = s1->s16 * s2->s16 + s3->s16;
            else if (width == K_HI)
                d->s16 = (s1->s16 * s2->s16 + s3->s16) >> 16;
            else if (width == K_WIDE)
                d->s32 = s1->s16 * s2->s16 + s3->s32;
            else assert(false);
            break;
        case K_U32:
            if (width == K_LO)
                d->u32 = s1->u32 * s2->u32 + s3->u32;
            else if (width == K_HI)
                d->u32 = (s1->u32 * s2->u32 + s3->u32) >> 32;
            else if (width == K_WIDE)
                d->u64 = s1->u32 * s2->u32 + s3->u64;
            else assert(false);
            break;
        case K_S32:
            if (width == K_LO)
                d->s32 = s1->s32 * s2->s32 + s3->s32;
            else if (width == K_HI)
                d->s32 = (s1->s32 * s2->s32 + s3->s32) >> 32;
            else if (width == K_WIDE)
                d->s64 = s1->s32 * s2->s32 + s3->s64;
            else assert(false);
            break;
        case K_F32:
            d->f32 = s1->f32 * s2->f32 + s3->f32;
            break;
        case K_F64:
            d->f64 = s1->f64 * s2->f64 + s3->f64;
            break;
        default:
            assert(false);
    }
    return 0;
}

int THREAD::DoMad24(TREE * inst)
{
    throw new EMULATOR::Unimplemented("MAD24 unimplemented");
}

int THREAD::DoMax(TREE * inst)
{
    throw new EMULATOR::Unimplemented("MAX unimplemented");
}

int THREAD::DoMembar(TREE * inst)
{
    throw new EMULATOR::Unimplemented("MEMBAR unimplemented");
}

int THREAD::DoMin(TREE * inst)
{
    throw new EMULATOR::Unimplemented("MIN unimplemented");
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
            } else assert(false);
        } else assert(false);
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
              || gt == K_U16 || gt == K_U32 || gt == K_U64
              || gt == K_S16 || gt == K_S32 || gt == K_S64
              || gt == K_F32 || gt == K_F64)
            ttype = t;
        else assert(false);
    }
    assert(ttype);
    int type = ttype->GetType();
    TREE * dst = odst->GetChild( 0);
    TREE * src = osrc->GetChild(0);
    SYMBOL * sdst = 0;
    SYMBOL * ssrc = 0;
    if (dst->GetType() == T_WORD)
    {
        sdst = this->root->FindSymbol(dst->GetText());
    } else assert(false);

    TYPES::Types * d;
    TYPES::Types * s;
    TYPES::Types value;
    char * dummy;

    d = (TYPES::Types*)sdst->pvalue;
    s = &value;

    if (src->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->emulator->Eval(type, src->GetChild(0));
        switch (type)
        {
            case K_U16:
                s->u16 = c.value.u16;
                break;
            case K_S16:
                s->s16 = c.value.s16;
                break;
            case K_U32:
                s->u32 = c.value.u32;
                break;
            case K_S32:
                s->s32 = c.value.s32;
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
                assert(false);
        }
    } else if (src->GetType() == T_WORD)
    {
        ssrc = this->root->FindSymbol(src->GetText());
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
            else assert(false);
        }
    }

    switch (type)
    {
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
            assert(false);
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
    sdst = this->root->FindSymbol(dst->GetText());
    char * dummy;

    TYPES::Types value1; // used if literal
    TYPES::Types value2; // used if literal
    TYPES::Types * d = (TYPES::Types*)sdst->pvalue;
    TYPES::Types * s1 = &value1;
    TYPES::Types * s2 = &value2;

    if (src1->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->emulator->Eval(type, src1->GetChild( 0));
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
                assert(false);
        }
    } else if (src1->GetType() == T_WORD)
    {
        ssrc1 = this->root->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        s1 = (TYPES::Types*)ssrc1->pvalue;
    } else assert(false);

    if (src2->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->emulator->Eval(type, src2->GetChild(0));
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
            default:
                assert(false);
        }
    } else if (src2->GetType() == T_WORD)
    {
        ssrc2 = this->root->FindSymbol(src2->GetText());
        assert(ssrc2 != 0);
        s2 = (TYPES::Types*)ssrc2->pvalue;
    } else assert(false);

    switch (type)
    {
        case K_U16:
            if (width == K_LO)
                d->u16 = s1->u16 * s2->u16;
            else if (width == K_HI)
                d->u16 = (s1->u16 * s2->u16 ) >> 16;
            else if (width == K_WIDE)
                d->u32 = s1->u16 * s2->u16;
            else assert(false);
            break;
        case K_S16:
            if (width == K_LO)
                d->s16 = s1->s16 * s2->s16;
            else if (width == K_HI)
                d->s16 = (s1->s16 * s2->s16 ) >> 16;
            else if (width == K_WIDE)
                d->s32 = s1->s16 * s2->s16;
            else assert(false);
            break;
        case K_U32:
            if (width == K_LO)
                d->u32 = s1->u32 * s2->u32;
            else if (width == K_HI)
                d->u32 = (s1->u32 * s2->u32 ) >> 16;
            else if (width == K_WIDE)
                d->u64 = s1->u32 * s2->u32;
            else assert(false);
            break;
        case K_S32:
            if (width == K_LO)
                d->s32 = s1->s32 * s2->s32;
            else if (width == K_HI)
                d->s32 = (s1->s32 * s2->s32 ) >> 16;
            else if (width == K_WIDE)
                d->s64 = s1->s32 * s2->s32;
            else assert(false);
            break;
        case K_U64:
            if (width == K_LO)
                d->u64 = s1->u64 * s2->u64;
            else if (width == K_HI)
                d->u64 = (s1->u64 * s2->u64 ) >> 16;
            else assert(false);
            break;
        case K_S64:
            if (width == K_LO)
                d->s64 = s1->s64 * s2->s64;
            else if (width == K_HI)
                d->s64 = (s1->s64 * s2->s64 ) >> 16;
            else assert(false);
            break;
        case K_F32:
            d->f32 = s1->f32 * s2->f32;
            break;
        case K_F64:
            d->f64 = s1->f64 * s2->f64;
            break;
        default:
            assert(false);
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
            } else assert(false);
        } else assert(false);
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
        else assert(false);
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
    sdst = this->root->FindSymbol(dst->GetText());
    char * dummy;

    TYPES::Types value1; // used if literal
    TYPES::Types value2; // used if literal
    TYPES::Types * d = (TYPES::Types*)sdst->pvalue;
    TYPES::Types * s1 = &value1;
    TYPES::Types * s2 = &value2;

    if (src1->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->emulator->Eval(type, src1->GetChild( 0));
        switch (type)
        {
            case K_U32:
                s1->u32 = c.value.u32;
                break;
            case K_S32:
                s1->s32 = c.value.s32;
                break;
            default:
                assert(false);
        }
    } else if (src1->GetType() == T_WORD)
    {
        ssrc1 = this->root->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        s1 = (TYPES::Types*)ssrc1->pvalue;
    } else assert(false);

    if (src2->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->emulator->Eval(type, src2->GetChild(0));
        switch (type)
        {
            case K_U32:
                s2->u32 = c.value.u32;
                break;
            case K_S32:
                s2->s32 = c.value.s32;
                break;
            default:
                assert(false);
        }
    } else if (src2->GetType() == T_WORD)
    {
        ssrc2 = this->root->FindSymbol(src2->GetText());
        assert(ssrc2 != 0);
        s2 = (TYPES::Types*)ssrc2->pvalue;
    } else assert(false);

    switch (type)
    {
        case K_U32:
            if (width == K_LO)
                d->u32 = (s1->u32 * s2->u32) & 0xffffffff;
            else if (width == K_HI)
                d->u32 = (s1->u32 * s2->u32 ) >> 16;
            else assert(false);
            break;
        case K_S32:
            if (width == K_LO)
                d->s32 = (s1->s32 * s2->s32) & 0xffffffff;
            else if (width == K_HI)
                d->s32 = (s1->s32 * s2->s32 ) >> 16;
            else assert(false);
            break;
        default:
            assert(false);
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
            } else assert(false);
        } else assert(false);
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
        else assert(false);
    }
    assert(ttype != 0);
    this->emulator->unimplemented(ftz, "NEG.ftz not implemented.");

    int type = ttype->GetType();
    TREE * dst = odst->GetChild(0);
    TREE * src1 = osrc1->GetChild(0);

    SYMBOL * sdst = 0;
    if (dst->GetType() == T_WORD)
    {
        sdst = this->root->FindSymbol(dst->GetText());
    } else assert(false);

    TYPES::Types value1;
    char * dummy;
    TYPES::Types * d = (TYPES::Types*)sdst->pvalue;
    TYPES::Types * s1 = &value1;

    if (src1->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->emulator->Eval(type, src1->GetChild( 0));
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
                assert(false);
        }
    } else if (src1->GetType() == T_WORD)
    {
        SYMBOL * ssrc1 = this->root->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        assert(ssrc1->size == this->emulator->Sizeof(type));
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
                assert(false);
        }
    } else assert(false);

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
            assert(false);
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
            } else assert(false);
        } else assert(false);
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
        else assert(false);
    }
    assert(type != 0);
    TREE * dst = odst->GetChild(0);
    TREE * src1 = osrc1->GetChild(0);

    SYMBOL * sdst = 0;
    if (dst->GetType() == T_WORD)
    {
        sdst = this->root->FindSymbol(dst->GetText());
    } else assert(false);

    TYPES::Types value1;
    char * dummy;
    TYPES::Types * d = (TYPES::Types*)sdst->pvalue;
    TYPES::Types * s1 = &value1;

    if (src1->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->emulator->Eval(type, src1->GetChild( 0));
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
                assert(false);
        }
    } else if (src1->GetType() == T_WORD)
    {
        SYMBOL * ssrc1 = this->root->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        assert(ssrc1->size == this->emulator->Sizeof(type));
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
                assert(false);
        }
    } else assert(false);

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
            d->pred = ~s1->pred;
            break;
        default:
            assert(false);
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
            } else assert(false);
        } else assert(false);
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
        else assert(false);
    }
    assert(type != 0);
    TREE * dst = odst->GetChild(0);
    TREE * src1 = osrc1->GetChild(0);
    TREE * src2 = osrc2->GetChild(0);

    SYMBOL * sdst = 0;
    if (dst->GetType() == T_WORD)
    {
        sdst = this->root->FindSymbol(dst->GetText());
    } else assert(false);

    TYPES::Types value1;
    TYPES::Types value2;
    char * dummy;
    TYPES::Types * d = (TYPES::Types*)sdst->pvalue;
    TYPES::Types * s1 = &value1;
    TYPES::Types * s2 = &value2;

    if (src1->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->emulator->Eval(type, src1->GetChild( 0));
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
                assert(false);
        }
    } else if (src1->GetType() == T_WORD)
    {
        SYMBOL * ssrc1 = this->root->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        assert(ssrc1->size == this->emulator->Sizeof(type));
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
                assert(false);
        }
    } else assert(false);

    if (src2->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->emulator->Eval(type, src2->GetChild(0));
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
                assert(false);
        }
    } else if (src2->GetType() == T_WORD)
    {
        SYMBOL * ssrc2 = this->root->FindSymbol(src2->GetText());
        assert(ssrc2 != 0);
        assert(ssrc2->size == this->emulator->Sizeof(type));
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
                assert(false);
        }
    } else assert(false);

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
            assert(false);
    }
    return 0;
}

int THREAD::DoPmevent(TREE * inst)
{
    throw new EMULATOR::Unimplemented("PMEVENT unimplemented");
}

int THREAD::DoPopc(TREE * inst)
{
    throw new EMULATOR::Unimplemented("POPC unimplemented");
}

int THREAD::DoPrefetch(TREE * inst)
{
    throw new EMULATOR::Unimplemented("PREFETCH unimplemented");
}

int THREAD::DoPrefetchu(TREE * inst)
{
    throw new EMULATOR::Unimplemented("PREFETCHU unimplemented");
}

int THREAD::DoPrmt(TREE * inst)
{
    throw new EMULATOR::Unimplemented("PRMT unimplemented");
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
            } else assert(false);
        } else assert(false);
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
        else assert(false);
    }
    assert(ttype != 0);
    this->emulator->unimplemented(ftz, "RCP.ftz not implemented.");

    int type = ttype->GetType();
    TREE * dst = odst->GetChild(0);
    TREE * src1 = osrc1->GetChild(0);

    SYMBOL * sdst = 0;
    if (dst->GetType() == T_WORD)
    {
        sdst = this->root->FindSymbol(dst->GetText());
    } else assert(false);

    TYPES::Types value1;
    char * dummy;
    TYPES::Types * d = (TYPES::Types*)sdst->pvalue;
    TYPES::Types * s1 = &value1;

    if (src1->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->emulator->Eval(type, src1->GetChild( 0));
        switch (type)
        {
            case K_F32:
                s1->f32 = c.value.f32;
                break;
            case K_F64:
                s1->f64 = c.value.f64;
                break;
            default:
                assert(false);
        }
    } else if (src1->GetType() == T_WORD)
    {
        SYMBOL * ssrc1 = this->root->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        assert(ssrc1->size == this->emulator->Sizeof(type));
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
                assert(false);
        }
    } else assert(false);

    switch (type)
    {
        case K_F32:
            d->f32 = 1 / s1->f32;
            break;
        case K_F64:
            d->f64 = 1 / s1->f64;
            break;
        default:
            assert(false);
    }
    return 0;
}

int THREAD::DoRed(TREE * inst)
{
    throw new EMULATOR::Unimplemented("RED unimplemented");
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
            } else assert(false);
        } else assert(false);
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
        else assert(false);
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
    sdst = this->root->FindSymbol(dst->GetText());
    char * dummy;

    TYPES::Types value1; // used if literal
    TYPES::Types value2; // used if literal
    TYPES::Types * d = (TYPES::Types*)sdst->pvalue;
    TYPES::Types * s1 = &value1;
    TYPES::Types * s2 = &value2;

    if (src1->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->emulator->Eval(type, src1->GetChild( 0));
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
                assert(false);
        }
    } else if (src1->GetType() == T_WORD)
    {
        ssrc1 = this->root->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        s1 = (TYPES::Types*)ssrc1->pvalue;
    } else assert(false);

    if (src2->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->emulator->Eval(type, src2->GetChild(0));
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
                assert(false);
        }
    } else if (src2->GetType() == T_WORD)
    {
        ssrc2 = this->root->FindSymbol(src2->GetText());
        assert(ssrc2 != 0);
        s2 = (TYPES::Types*)ssrc2->pvalue;
    } else assert(false);

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
            assert(false);
    }
    return 0;
}

int THREAD::DoRet(TREE * inst)
{
    throw new EMULATOR::Unimplemented("RET unimplemented");
}

int THREAD::DoRsqrt(TREE * inst)
{
    throw new EMULATOR::Unimplemented("RSQRT unimplemented");
}

int THREAD::DoSad(TREE * inst)
{
    throw new EMULATOR::Unimplemented("SAD unimplemented");
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
            } else assert(false);
        } else assert(false);
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
        else assert(false);
    }
    assert(ttype != 0);
    this->emulator->unimplemented(ftz, "SELP.ftz not implemented.");

    int type = ttype->GetType();
    TREE * dst = odst->GetChild(0);
    TREE * src1 = osrc1->GetChild(0);
    TREE * src2 = osrc2->GetChild(0);
    TREE * src3 = osrc3->GetChild(0);

    SYMBOL * sdst = 0;
    if (dst->GetType() == T_WORD)
    {
        sdst = this->root->FindSymbol(dst->GetText());
    } else assert(false);

    TYPES::Types * d = (TYPES::Types*)sdst->pvalue;
    TYPES::Types value1;
    TYPES::Types value2;
    TYPES::Types * s1 = &value1;
    TYPES::Types * s2 = &value2;
    TYPES::Types * s3 = 0;
    char * dummy;

    if (src1->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->emulator->Eval(type, src1->GetChild( 0));
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
                assert(false);
        }
    } else if (src1->GetType() == T_WORD)
    {
        SYMBOL * ssrc1 = this->root->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        assert(ssrc1->size == this->emulator->Sizeof(type));
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
                assert(false);
        }
    } else assert(false);

    if (src2->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->emulator->Eval(type, src2->GetChild(0));
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
                assert(false);
        }
    } else if (src2->GetType() == T_WORD)
    {
        SYMBOL * ssrc2 = this->root->FindSymbol(src2->GetText());
        assert(ssrc2 != 0);
        assert(ssrc2->size == this->emulator->Sizeof(type));
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
                assert(false);
        }
    } else assert(false);

    if (src3->GetType() == T_WORD)
    {
        SYMBOL * ssrc3 = this->root->FindSymbol(src3->GetText());
        assert(ssrc3 != 0);
// FIX      assert(strcmp(ssrc3->type, ".pred") == 0);
        s3 = (TYPES::Types*)ssrc3->pvalue;
        assert(s3 != 0);
    } else assert(false);

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
            assert(false);
    }
    return 0;
}

int THREAD::DoSet(TREE * inst)
{
    throw new EMULATOR::Unimplemented("SET unimplemented");
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
            else assert(false);
        } else assert(false);
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
        else assert(false);
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
        sdst1 = this->root->FindSymbol(dst1->GetText());
    } else assert(false);

    TYPES::Types * d = (TYPES::Types*)sdst1->pvalue;

    if (src1->GetType() == T_WORD)
    {
        ssrc1 = this->root->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        s1 = (TYPES::Types*)ssrc1->pvalue;
        assert(s1 != 0);
    }
    else if (src1->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->emulator->Eval(type, src1->GetChild( 0));
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
                assert(false);
        }
    }
    else assert(false);

    if (src2->GetType() == T_WORD)
    {
        ssrc2 = this->root->FindSymbol(src2->GetText());
        assert(ssrc2 != 0);
        s2 = (TYPES::Types*)ssrc2->pvalue;
        assert(s2 != 0);
    }
    else if (src2->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->emulator->Eval(type, src2->GetChild(0));
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
                assert(false);
        }
    }
    else assert(false);

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
                    assert(false);
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
                    assert(false);
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
                    assert(false);
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
                    assert(false);
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
                    assert(false);
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
                    assert(false);
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
                    assert(false);
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
                    assert(false);
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
                    assert(false);
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
                    assert(false);
            }
            break;
        default:
            assert(false);
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
            } else assert(false);
        } else assert(false);
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
        else assert(false);
    }
    assert(type != 0);
    TREE * dst = odst->GetChild(0);
    TREE * src1 = osrc1->GetChild(0);
    TREE * src2 = osrc2->GetChild(0);

    SYMBOL * sdst = 0;
    if (dst->GetType() == T_WORD)
    {
        sdst = this->root->FindSymbol(dst->GetText());
    } else assert(false);

    TYPES::Types value1;
    TYPES::Types value2;
    char * dummy;
    TYPES::Types * d = (TYPES::Types*)sdst->pvalue;
    TYPES::Types * s1 = &value1;
    TYPES::Types * s2 = &value2;

    if (src1->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->emulator->Eval(type, src1->GetChild( 0));
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
                assert(false);
        }
    } else if (src1->GetType() == T_WORD)
    {
        SYMBOL * ssrc1 = this->root->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        assert(ssrc1->size == this->emulator->Sizeof(type));
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
                assert(false);
        }
    } else assert(false);

    if (src2->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->emulator->Eval(type, src2->GetChild(0));
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
                assert(false);
        }
    } else if (src2->GetType() == T_WORD)
    {
        SYMBOL * ssrc2 = this->root->FindSymbol(src2->GetText());
        assert(ssrc2 != 0);
        assert(ssrc2->size == this->emulator->Sizeof(type));
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
                assert(false);
        }
    } else assert(false);

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
            assert(false);
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
            } else assert(false);
        } else assert(false);
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
        else assert(false);
    }
    assert(type != 0);
    TREE * dst = odst->GetChild(0);
    TREE * src1 = osrc1->GetChild(0);
    TREE * src2 = osrc2->GetChild(0);

    SYMBOL * sdst = 0;
    if (dst->GetType() == T_WORD)
    {
        sdst = this->root->FindSymbol(dst->GetText());
    } else assert(false);

    TYPES::Types value1;
    TYPES::Types value2;
    char * dummy;
    TYPES::Types * d = (TYPES::Types*)sdst->pvalue;
    TYPES::Types * s1 = &value1;
    TYPES::Types * s2 = &value2;

    if (src1->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->emulator->Eval(type, src1->GetChild( 0));
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
                assert(false);
        }
    } else if (src1->GetType() == T_WORD)
    {
        SYMBOL * ssrc1 = this->root->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        assert(ssrc1->size == this->emulator->Sizeof(type));
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
                assert(false);
        }
    } else assert(false);

    if (src2->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->emulator->Eval(type, src2->GetChild(0));
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
                assert(false);
        }
    } else if (src2->GetType() == T_WORD)
    {
        SYMBOL * ssrc2 = this->root->FindSymbol(src2->GetText());
        assert(ssrc2 != 0);
        assert(ssrc2->size == this->emulator->Sizeof(type));
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
                assert(false);
        }
    } else assert(false);

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
            assert(false);
    }
    return 0;
}

int THREAD::DoSin(TREE * inst)
{
    throw new EMULATOR::Unimplemented("SIN unimplemented");
}

int THREAD::DoSlct(TREE * inst)
{
    throw new EMULATOR::Unimplemented("SLCT unimplemented");
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
            } else assert(false);
        } else assert(false);
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
        else assert(false);
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
    sdst = this->root->FindSymbol(dst->GetText());
    char * dummy;

    TYPES::Types value1; // used if literal
    TYPES::Types * d = (TYPES::Types*)sdst->pvalue;
    TYPES::Types * s1 = &value1;

    if (src1->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->emulator->Eval(type, src1->GetChild( 0));
        switch (type)
        {
            case K_F32:
                s1->f32 = c.value.f32;
                break;
            case K_F64:
                s1->f64 = c.value.f64;
                break;
            default:
                assert(false);
        }
    } else if (src1->GetType() == T_WORD)
    {
        ssrc1 = this->root->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        s1 = (TYPES::Types*)ssrc1->pvalue;
    } else assert(false);

    switch (type)
    {
        case K_F32:
            d->f32 = sqrt(s1->f32);
            break;
        case K_F64:
            d->f64 = sqrt(s1->f64);
            break;
        default:
            assert(false);
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
            } else assert(false);
        } else assert(false);
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
        else assert(false);
    }
    assert(ttype != 0);
    assert(cop == 0);
    int type = ttype->GetType();

    TREE * dst = odst->GetChild( 0);
    assert(dst->GetType() == T_WORD);
    SYMBOL * sdst = this->root->FindSymbol(dst->GetText());
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
        value = this->emulator->Eval(K_S32, const_expr);
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
            assert(false);
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
            assert(false);
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
            CONSTANT c = this->emulator->Eval(type, src->GetChild(0));
            s = &c.value;
        } else if (src->GetType() == T_WORD)
        {
            ssrc = this->root->FindSymbol(src->GetText());
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
                else assert(false);
            }
        } else assert(false);

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
                assert(false);
        }

        addr = addr + this->emulator->Sizeof(type);
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
            } else assert(false);
        } else assert(false);
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
        else assert(false);
    }
    assert(ttype != 0);
    int type = ttype->GetType();
    TREE * dst = odst->GetChild(0);
    TREE * src1 = osrc1->GetChild(0);
    TREE * src2 = osrc2->GetChild(0);

    SYMBOL * sdst = 0;
    if (dst->GetType() == T_WORD)
    {
        sdst = this->root->FindSymbol(dst->GetText());
    } else assert(false);

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
        CONSTANT c = this->emulator->Eval(type, src1->GetChild( 0));
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
                assert(false);
        }
    } else if (src1->GetType() == T_WORD)
    {
        SYMBOL * ssrc1 = this->root->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        assert(ssrc1->size == this->emulator->Sizeof(type));
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
                assert(false);
        }
    } else assert(false);

    if (src2->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->emulator->Eval(type, src2->GetChild(0));
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
                assert(false);
        }
    } else if (src2->GetType() == T_WORD)
    {
        SYMBOL * ssrc2 = this->root->FindSymbol(src2->GetText());
        assert(ssrc2 != 0);
        assert(ssrc2->size == this->emulator->Sizeof(type));
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
                assert(false);
        }
    } else assert(false);

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
            assert(false);
    }
    return 0;
}

int THREAD::DoSubc(TREE * inst)
{
    throw new EMULATOR::Unimplemented("SUBC unimplemented");
}

int THREAD::DoSuld(TREE * inst)
{
    throw new EMULATOR::Unimplemented("SULD unimplemented");
}

int THREAD::DoSuq(TREE * inst)
{
    throw new EMULATOR::Unimplemented("SUQ unimplemented");
}

int THREAD::DoSured(TREE * inst)
{
    throw new EMULATOR::Unimplemented("SURED unimplemented");
}

int THREAD::DoSust(TREE * inst)
{
    throw new EMULATOR::Unimplemented("SUST unimplemented");
}

int THREAD::DoTestp(TREE * inst)
{
    throw new EMULATOR::Unimplemented("TESTP unimplemented");
}

int THREAD::DoTex(TREE * inst)
{
    throw new EMULATOR::Unimplemented("TEX unimplemented");
}

int THREAD::DoTrap(TREE * inst)
{
    throw new EMULATOR::Unimplemented("TRAP unimplemented");
}

int THREAD::DoTxq(TREE * inst)
{
    throw new EMULATOR::Unimplemented("TXQ unimplemented");
}

int THREAD::DoVabsdiff(TREE * inst)
{
    throw new EMULATOR::Unimplemented("VABSDIFF unimplemented");
}

int THREAD::DoVadd(TREE * inst)
{
    throw new EMULATOR::Unimplemented("Vadd unimplemented");
}

int THREAD::DoVmad(TREE * inst)
{
    throw new EMULATOR::Unimplemented("VMAD unimplemented");
}

int THREAD::DoVmax(TREE * inst)
{
    throw new EMULATOR::Unimplemented("VMAX unimplemented");
}

int THREAD::DoVmin(TREE * inst)
{
    throw new EMULATOR::Unimplemented("VMIN unimplemented");
}

int THREAD::DoVote(TREE * inst)
{
    throw new EMULATOR::Unimplemented("VOTE unimplemented");
}

int THREAD::DoVset(TREE * inst)
{
    throw new EMULATOR::Unimplemented("VSET unimplemented");
}

int THREAD::DoVshl(TREE * inst)
{
    throw new EMULATOR::Unimplemented("VSHL unimplemented");
}

int THREAD::DoVshr(TREE * inst)
{
    throw new EMULATOR::Unimplemented("VSHR unimplemented");
}

int THREAD::DoVsub(TREE * inst)
{
    throw new EMULATOR::Unimplemented("VSUB unimplemented");
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
            } else assert(false);
        } else assert(false);
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
        else assert(false);
    }
    assert(type != 0);
    TREE * dst = odst->GetChild(0);
    TREE * src1 = osrc1->GetChild(0);
    TREE * src2 = osrc2->GetChild(0);

    SYMBOL * sdst = 0;
    if (dst->GetType() == T_WORD)
    {
        sdst = this->root->FindSymbol(dst->GetText());
    } else assert(false);

    TYPES::Types value1;
    TYPES::Types value2;
    char * dummy;
    TYPES::Types * d = (TYPES::Types*)sdst->pvalue;
    TYPES::Types * s1 = &value1;
    TYPES::Types * s2 = &value2;

    if (src1->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->emulator->Eval(type, src1->GetChild( 0));
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
                assert(false);
        }
    } else if (src1->GetType() == T_WORD)
    {
        SYMBOL * ssrc1 = this->root->FindSymbol(src1->GetText());
        assert(ssrc1 != 0);
        assert(ssrc1->size == this->emulator->Sizeof(type));
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
                assert(false);
        }
    } else assert(false);

    if (src2->GetType() == TREE_CONSTANT_EXPR)
    {
        CONSTANT c = this->emulator->Eval(type, src2->GetChild(0));
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
                assert(false);
        }
    } else if (src2->GetType() == T_WORD)
    {
        SYMBOL * ssrc2 = this->root->FindSymbol(src2->GetText());
        assert(ssrc2 != 0);
        assert(ssrc2->size == this->emulator->Sizeof(type));
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
                assert(false);
        }
    } else assert(false);

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
            assert(false);
    }
    return 0;
}
