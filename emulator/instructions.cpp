#include "emulator.h"
#include <assert.h>
#include <iostream>

int CUDA_EMULATOR::DoAdd(pANTLR3_BASE_TREE inst)
{
    int start = 0;
    if (GetType(GetChild(inst, start)) == TREE_PRED)
        start++;
    assert(GetType(GetChild(inst, start)) == KI_ADD);
    start++;
    pANTLR3_BASE_TREE ttype = 0;
    pANTLR3_BASE_TREE odst = 0;
    pANTLR3_BASE_TREE osrc1 = 0;
    pANTLR3_BASE_TREE osrc2 = 0;
    for (;; ++start)
    {
        pANTLR3_BASE_TREE t = GetChild(inst, start);
        if (t == 0)
            break;
        int gt = GetType(t);
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
    for (int i = 0; ; ++i)
    {
        pANTLR3_BASE_TREE t = GetChild(ttype, i);
        if (t == 0)
            break;
        int gt = GetType(t);
        if (gt == K_SAT)
            sat = true;
        else if (gt == K_U16 || gt == K_U32 || gt == K_U64
                 || gt == K_S16 || gt == K_S32 || gt == K_S64)
            ttype = t;
        else assert(false);
    }
    assert(ttype != 0);
    assert(sat == 0);
    int type = GetType(ttype);
    pANTLR3_BASE_TREE dst = GetChild(odst,0);
    pANTLR3_BASE_TREE src1 = GetChild(osrc1,0);
    pANTLR3_BASE_TREE src2 = GetChild(osrc2,0);

    Symbol * sdst = 0;
    if (dst->getType(dst) == T_WORD)
    {
        sdst = FindSymbol(GetText(dst));
    } else assert(false);
    typedef union TYPES {
        __int64 s64;
        __int32 s32;
        __int16 s16;;
        unsigned __int64 u64;
        unsigned __int32 u32;
        unsigned __int16 u16;
    } TYPES;

    TYPES value1;
    TYPES value2;
    char * dummy;
    TYPES * d = (TYPES*)sdst->pvalue;
    TYPES * s1 = &value1;
    TYPES * s2 = &value2;

    if (GetType(src1) == TREE_CONSTANT_EXPR)
    {
        Constant c = Eval(type, GetChild(src1, 0));
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
    } else if (GetType(src1) == T_WORD)
    {
        Symbol * ssrc1 = FindSymbol(GetText(src1));
        assert(ssrc1 != 0);
        assert(ssrc1->size == Sizeof(type));
        TYPES * psrc1_value = (TYPES*)ssrc1->pvalue;
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
            default:
                assert(false);
        }
    } else assert(false);

    if (GetType(src2) == TREE_CONSTANT_EXPR)
    {
        Constant c = Eval(type, GetChild(src2, 0));
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
    } else if (GetType(src2) == T_WORD)
    {
        Symbol * ssrc2 = FindSymbol(GetText(src2));
        assert(ssrc2 != 0);
        assert(ssrc2->size == Sizeof(type));
        TYPES * psrc2_value = (TYPES*)ssrc2->pvalue;
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
            default:
                assert(false);
        }
    } else assert(false);

    switch (type)
    {
        case K_U16:
            d->u16 = s1->u16 + s2->u16;
            break;
        case K_S16:
            d->s16 = s1->s16 + s2->s16;
            break;
        case K_U32:
            d->u32 = s1->u32 + s2->u32;
            break;
        case K_S32:
            d->s32 = s1->s32 + s2->s32;
            break;
        case K_S64:
            d->s64 = s1->s64 + s2->s64;
            break;
        case K_U64:
            d->u64 = s1->u64 + s2->u64;
            break;
        default:
            assert(false);
    }
    return 0;
}

int CUDA_EMULATOR::DoBar(pANTLR3_BASE_TREE inst)
{
    // ONLY VERY SIMPLE SYNCHRONIZATION IMPLEMENTED!!!
    int start = 0;
    if (GetType(GetChild(inst, start)) == TREE_PRED)
        start++;
    assert(GetType(GetChild(inst, start)) == KI_BAR);
    start++;
    pANTLR3_BASE_TREE ttype = 0;
    pANTLR3_BASE_TREE osrc1 = 0;
    for (;; ++start)
    {
        pANTLR3_BASE_TREE t = GetChild(inst, start);
        if (t == 0)
            break;
        int gt = GetType(t);
        if (gt == TREE_TYPE)
            ttype = t;
        else if (gt == TREE_OPR)
        {
            if (osrc1 == 0)
            {
                osrc1 = t;
            } else assert(false);
        } else assert(false);
    }
    assert(ttype != 0);
    assert(osrc1 != 0);
    for (int i = 0; ; ++i)
    {
        pANTLR3_BASE_TREE t = GetChild(ttype, i);
        if (t == 0)
            break;
        int gt = GetType(t);
        if (gt != K_SYNC)
            assert(false);
    }
    int type = K_U32;
    pANTLR3_BASE_TREE src1 = GetChild(osrc1,0);

    TYPES value1;
    char * dummy;
    TYPES * s1 = &value1;

    if (GetType(src1) == TREE_CONSTANT_EXPR)
    {
        Constant c = Eval(type, GetChild(src1, 0));
        switch (type)
        {
            case K_U32:
                s1->u32 = c.value.u32;
                break;
            default:
                assert(false);
        }
    } else if (GetType(src1) == T_WORD)
    {
        Symbol * ssrc1 = FindSymbol(GetText(src1));
        assert(ssrc1 != 0);
        assert(ssrc1->size == Sizeof(type));
        TYPES * psrc1_value = (TYPES*)ssrc1->pvalue;
        switch (type)
        {
            case K_U32:
                s1->u32 = psrc1_value->u32;
                break;
            default:
                assert(false);
        }
    } else assert(false);

    assert(s1->u32 == 0);

    // pack up the thread context.  Synchronize all assumed.

    return -KI_BAR;
}

int CUDA_EMULATOR::DoBra(pANTLR3_BASE_TREE inst)
{
    int start = 0;
    for (;;)
    {
        pANTLR3_BASE_TREE t = GetChild(inst, start);
        assert(t != 0);
        if (GetType(t) == TREE_OPR)
            break;
        start++;
    }
    pANTLR3_BASE_TREE opr = GetChild(inst, start);
    pANTLR3_BASE_TREE dst = GetChild(opr, 0);
    assert(GetType(dst) == T_WORD);
    Symbol * sdst = FindSymbol(GetText(dst));
    assert (sdst != 0);
    return (int)sdst->pvalue;
}

int CUDA_EMULATOR::DoCvt(pANTLR3_BASE_TREE inst)
{
    int start = 0;
    if (GetType(GetChild(inst, start)) == TREE_PRED)
        start++;
    assert(GetType(GetChild(inst, start)) == KI_CVT);
    start++;
    assert(GetType(GetChild(inst, start)) == TREE_TYPE);
    pANTLR3_BASE_TREE type = GetChild(inst, start);
    start++;
    bool ftz = false;
    bool sat = false;
    pANTLR3_BASE_TREE tirnd = 0;
    pANTLR3_BASE_TREE tfrnd = 0;
    int irnd = 0;
    int frnd = 0;
    int src_type = 0;
    int dst_type = 0;
    int i = 0;
    for (;; ++i)
    {
        pANTLR3_BASE_TREE t = GetChild(type, i);
        if (t == 0)
            break;
        int gt = GetType(t);
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
            pANTLR3_BASE_TREE t = GetChild(tirnd, i);
            if (t == 0)
                break;
            int gt = GetType(t);
            if (gt == K_RNI || gt == K_RZI || gt == K_RMI || gt == K_RPI)
                irnd = gt;
            else assert(false);
        }
    }
    if (tfrnd != 0)
    {
        for (int i = 0; ; ++i)
        {
            pANTLR3_BASE_TREE t = GetChild(tfrnd, i);
            if (t == 0)
                break;
            int gt = GetType(t);
            if (gt == K_RN || gt == K_RZ || gt == K_RM || gt == K_RP)
                frnd = gt;
            else assert(false);
        }
    }

    typedef union TYPES {
        __int64 s64;
        __int32 s32;
        __int16 s16;;
        __int8 s8;
        unsigned __int64 u64;
        unsigned __int32 u32;
        unsigned __int16 u16;
        unsigned __int8 u8;
        float f32;
        double f64;
    } TYPES;
    TYPES * dst_value;
    TYPES * src_value;

    pANTLR3_BASE_TREE o1 = GetChild(inst, start++);
    assert(GetType(o1) == TREE_OPR);
    assert(GetType(GetChild(o1, 0)) == T_WORD);
    pANTLR3_BASE_TREE o2 = GetChild(inst, start++);
    assert(GetType(o2) == TREE_OPR);
    assert(GetType(GetChild(o2, 0)) == T_WORD);

    Symbol * s1 = FindSymbol(GetText(GetChild(o1, 0)));
    assert(s1 != 0);
    Symbol * s2 = FindSymbol(GetText(GetChild(o2, 0)));
    assert(s2 != 0);

    dst_value = (TYPES*)s1->pvalue;

    // handle .x, .y, .z stuff.
    if (strcmp(s2->type, "dim3") == 0)
    {
        // Get qualifier of the structure.
        pANTLR3_BASE_TREE tqual = GetChild(o2, 1);
        assert(tqual != 0);
        int qual = GetType(tqual);
        if (qual == K_X)
        {
            src_value = (TYPES*) &(((dim3*)s2->pvalue)->x);
        } else if (qual == K_Y)
        {
            src_value = (TYPES*) &(((dim3*)s2->pvalue)->y);
        } else if (qual == K_Z)
        {
            src_value = (TYPES*) &(((dim3*)s2->pvalue)->z);
        } else assert(false);
    } else
        src_value = (TYPES*)s2->pvalue;

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


int CUDA_EMULATOR::DoCvta(pANTLR3_BASE_TREE inst)
{
    // Assign source to destination.
    int start = 0;
    if (GetType(GetChild(inst, start)) == TREE_PRED)
        start++;
    assert(GetType(GetChild(inst, start)) == KI_CVTA);
    start++;
    pANTLR3_BASE_TREE ttype = 0;
    pANTLR3_BASE_TREE odst = 0;
    pANTLR3_BASE_TREE osrc = 0;
    for (;; ++start)
    {
        pANTLR3_BASE_TREE t = GetChild(inst, start);
        if (t == 0)
            break;
        int gt = GetType(t);
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
        pANTLR3_BASE_TREE t = GetChild(ttype, i);
        if (t == 0)
            break;
        int gt = GetType(t);
        if (gt == K_U32 || gt == K_U64)
            ttype = t;
        else if (gt == K_SHARED || gt == K_GLOBAL || gt == K_LOCAL)
            storage_class = gt;
        else assert(false);
    }
    assert(ttype);
    int type = GetType(ttype);
    pANTLR3_BASE_TREE dst = GetChild(odst, 0);
    pANTLR3_BASE_TREE src = GetChild(osrc,0);
    Symbol * sdst = 0;
    Symbol * ssrc = 0;
    if (dst->getType(dst) == T_WORD)
    {
        sdst = FindSymbol(GetText(dst));
    } else assert(false);

    typedef union TYPES {
        unsigned __int64 u64;
        unsigned __int32 u32;
    } TYPES;
    TYPES * d;
    TYPES * s;
    TYPES value;
    char * dummy;

    d = (TYPES*)sdst->pvalue;
    if (GetType(src) == T_WORD)
    {
        ssrc = FindSymbol(GetText(src));
        // Various types of id's to handle:
        assert(ssrc != 0);
        s = (TYPES*)ssrc->pvalue;
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

int CUDA_EMULATOR::DoDiv(pANTLR3_BASE_TREE inst)
{
    int start = 0;
    if (GetType(GetChild(inst, start)) == TREE_PRED)
        start++;
    assert(GetType(GetChild(inst, start)) == KI_DIV);
    start++;
    assert(GetType(GetChild(inst, start)) == TREE_TYPE);
    pANTLR3_BASE_TREE ttype = GetChild(inst, start);
    start++;
    pANTLR3_BASE_TREE odst = 0;
    pANTLR3_BASE_TREE osrc1 = 0;
    pANTLR3_BASE_TREE osrc2 = 0;
    for (;; ++start)
    {
        pANTLR3_BASE_TREE t = GetChild(inst, start);
        if (t == 0)
            break;
        int gt = GetType(t);
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
    pANTLR3_BASE_TREE tfrnd = 0;
    for (int i = 0; ; ++i)
    {
        pANTLR3_BASE_TREE t = GetChild(ttype, i);
        if (t == 0)
            break;
        int gt = GetType(t);
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
    int type = GetType(ttype);

    pANTLR3_BASE_TREE dst = GetChild(odst,0);
    pANTLR3_BASE_TREE src1 = GetChild(osrc1,0);
    pANTLR3_BASE_TREE src2 = GetChild(osrc2,0);

    typedef union TYPES {
        long s64;
        int s32;
        short s16;
        unsigned long u64;
        unsigned int u32;
        unsigned short u16;
        float f32;
        double f64;
    } TYPES;

    TYPES * pdst_value;
    TYPES * psrc1_value;
    TYPES * psrc2_value;
    TYPES src1_value;// used if literal
    TYPES src2_value;// used if literal

    Symbol * sdst = 0;
    Symbol * ssrc1 = 0;
    Symbol * ssrc2 = 0;
    assert(GetType(dst) == T_WORD);
    sdst = FindSymbol(GetText(dst));
    char * dummy;

    TYPES value1; // used if literal
    TYPES value2; // used if literal
    TYPES * d = (TYPES*)sdst->pvalue;
    TYPES * s1 = &value1;
    TYPES * s2 = &value2;

    if (GetType(src1) == TREE_CONSTANT_EXPR)
    {
        Constant c = Eval(type, GetChild(src1, 0));
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
    } else if (GetType(src1) == T_WORD)
    {
        ssrc1 = FindSymbol(GetText(src1));
        assert(ssrc1 != 0);
        s1 = (TYPES*)ssrc1->pvalue;
    } else assert(false);

    if (GetType(src2) == TREE_CONSTANT_EXPR)
    {
        Constant c = Eval(type, GetChild(src2, 0));
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
    } else if (GetType(src2) == T_WORD)
    {
        ssrc2 = FindSymbol(GetText(src2));
        assert(ssrc2 != 0);
        s2 = (TYPES*)ssrc2->pvalue;
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

int CUDA_EMULATOR::DoExit(pANTLR3_BASE_TREE inst)
{
    if (this->trace_level > 1)
        std::cout << "EXIT\n";
    return -KI_EXIT;
}


int CUDA_EMULATOR::DoFma(pANTLR3_BASE_TREE inst)
{
    // Multiply register and/or constants, and store in a register.
    int start = 0;
    if (GetType(GetChild(inst, start)) == TREE_PRED)
        start++;
    assert(GetType(GetChild(inst, start)) == KI_FMA);
    start++;
    assert(GetType(GetChild(inst, start)) == TREE_TYPE);
    pANTLR3_BASE_TREE ttype = GetChild(inst, start);
    start++;
    pANTLR3_BASE_TREE odst = 0;
    pANTLR3_BASE_TREE osrc1 = 0;
    pANTLR3_BASE_TREE osrc2 = 0;
    pANTLR3_BASE_TREE osrc3 = 0;
    for (;; ++start)
    {
        pANTLR3_BASE_TREE t = GetChild(inst, start);
        if (t == 0)
            break;
        int gt = GetType(t);
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
    pANTLR3_BASE_TREE tfrnd = 0;
    for (int i = 0; ; ++i)
    {
        pANTLR3_BASE_TREE t = GetChild(ttype, i);
        if (t == 0)
            break;
        int gt = GetType(t);
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
    int type = GetType(ttype);

    pANTLR3_BASE_TREE dst = GetChild(odst,0);
    pANTLR3_BASE_TREE src1 = GetChild(osrc1,0);
    pANTLR3_BASE_TREE src2 = GetChild(osrc2,0);
    pANTLR3_BASE_TREE src3 = GetChild(osrc3,0);

    // Supported types of MUL.
    typedef union TYPES {
        float f32;
        double f64;
    } TYPES;

    Symbol * sdst = 0;
    Symbol * ssrc1 = 0;
    Symbol * ssrc2 = 0;
    Symbol * ssrc3 = 0;
    assert(GetType(dst) == T_WORD);
    sdst = FindSymbol(GetText(dst));
    char * dummy;

    TYPES value1; // used if literal
    TYPES value2; // used if literal
    TYPES value3; // used if literal
    TYPES * d = (TYPES*)sdst->pvalue;
    TYPES * s1 = &value1;
    TYPES * s2 = &value2;
    TYPES * s3 = &value2;

    if (GetType(src1) == TREE_CONSTANT_EXPR)
    {
        Constant c = Eval(type, GetChild(src1, 0));
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
    } else if (GetType(src1) == T_WORD)
    {
        ssrc1 = FindSymbol(GetText(src1));
        assert(ssrc1 != 0);
        s1 = (TYPES*)ssrc1->pvalue;
    } else assert(false);

    if (GetType(src2) == TREE_CONSTANT_EXPR)
    {
        Constant c = Eval(type, GetChild(src2, 0));
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
    } else if (GetType(src2) == T_WORD)
    {
        ssrc2 = FindSymbol(GetText(src2));
        assert(ssrc2 != 0);
        s2 = (TYPES*)ssrc2->pvalue;
    } else assert(false);

    if (GetType(src3) == TREE_CONSTANT_EXPR)
    {
        Constant c = Eval(type, GetChild(src3, 0));
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
    } else if (GetType(src3) == T_WORD)
    {
        ssrc3 = FindSymbol(GetText(src3));
        assert(ssrc3 != 0);
        s3 = (TYPES*)ssrc3->pvalue;
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

int CUDA_EMULATOR::DoLd(pANTLR3_BASE_TREE inst)
{
    int start = 0;
    if (GetType(GetChild(inst, start)) == TREE_PRED)
        start++;
    assert(GetType(GetChild(inst, start)) == KI_LD);
    start++;
    pANTLR3_BASE_TREE ttype = 0;
    pANTLR3_BASE_TREE odst = 0;
    pANTLR3_BASE_TREE osrc = 0;
    for (;; ++start)
    {
        pANTLR3_BASE_TREE t = GetChild(inst, start);
        if (t == 0)
            break;
        int gt = GetType(t);
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
        pANTLR3_BASE_TREE t = GetChild(ttype, i);
        if (t == 0)
            break;
        int gt = GetType(t);
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
    int type = GetType(ttype);

    // Get two operands, assign source to destination.
    pANTLR3_BASE_TREE dst = GetChild(odst, 0);
    pANTLR3_BASE_TREE src = GetChild(osrc, 0);
    Symbol * sdst = 0;
    Symbol * ssrc = 0;
    assert(dst->getType(dst) == T_WORD);
    sdst = FindSymbol(GetText(dst));
    
    assert(src->getType(src) == T_WORD);
    ssrc = FindSymbol(GetText(src));
    pANTLR3_BASE_TREE plus = GetChild(osrc, 1);
    Constant value(0);
    if (plus != 0)
    {
        pANTLR3_BASE_TREE const_expr_tree = GetChild(osrc, 2);
        assert(const_expr_tree != 0);
        assert(GetType(const_expr_tree) == TREE_CONSTANT_EXPR);
        pANTLR3_BASE_TREE const_expr = GetChild(const_expr_tree, 0);
        assert(const_expr != 0);
        value = Eval(K_S32, const_expr);
    }

    typedef union TYPES {
        __int64 s64;
        __int32 s32;
        __int16 s16;;
        __int8 s8;
        unsigned __int64 u64;
        unsigned __int32 u32;
        unsigned __int16 u16;
        unsigned __int8 u8;
        float f32;
        double f64;
    } TYPES;
    TYPES * d = (TYPES*)sdst->pvalue;
    // Unfortunately, different semantics for different storage classes.
    TYPES * s = 0;
    if (ssrc->storage_class != K_REG)
        s = (TYPES*)ssrc->pvalue;
    else if (plus != 0)
    {
        void * addr = *(void**)ssrc->pvalue;
        switch (value.type)
        {
        case K_U8:
            s = (TYPES*)(((unsigned char *)addr) + value.value.u8);
            break;
        case K_U16:
            s = (TYPES*)(((unsigned char *)addr) + value.value.u16);
            break;
        case K_U32:
            s = (TYPES*)(((unsigned char *)addr) + value.value.u32);
            break;
        case K_U64:
            s = (TYPES*)(((unsigned char *)addr) + value.value.u64);
            break;
        case K_S8:
            s = (TYPES*)(((unsigned char *)addr) + value.value.s8);
            break;
        case K_S16:
            s = (TYPES*)(((unsigned char *)addr) + value.value.s16);
            break;
        case K_S32:
            s = (TYPES*)(((unsigned char *)addr) + value.value.s32);
            break;
        case K_S64:
            s = (TYPES*)(((unsigned char *)addr) + value.value.s64);
            break;
        default:
            assert(false);
        }
    }
    else
        s = (TYPES*)ssrc->pvalue;
    
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

int CUDA_EMULATOR::DoLdu(pANTLR3_BASE_TREE inst)
{
    int start = 0;
    if (GetType(GetChild(inst, start)) == TREE_PRED)
        start++;
    assert(GetType(GetChild(inst, start)) == KI_LDU);
    start++;
    pANTLR3_BASE_TREE ttype = 0;
    pANTLR3_BASE_TREE odst = 0;
    pANTLR3_BASE_TREE osrc = 0;
    for (;; ++start)
    {
        pANTLR3_BASE_TREE t = GetChild(inst, start);
        if (t == 0)
            break;
        int gt = GetType(t);
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
        pANTLR3_BASE_TREE t = GetChild(ttype, i);
        if (t == 0)
            break;
        int gt = GetType(t);
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
    int type = GetType(ttype);

    // Get two operands, assign source to destination.
    pANTLR3_BASE_TREE dst = GetChild(odst, 0);
    pANTLR3_BASE_TREE src = GetChild(osrc, 0);
    Symbol * sdst = 0;
    Symbol * ssrc = 0;
    assert(dst->getType(dst) == T_WORD);
    sdst = FindSymbol(GetText(dst));
    
    assert(src->getType(src) == T_WORD);
    ssrc = FindSymbol(GetText(src));
    pANTLR3_BASE_TREE plus = GetChild(osrc, 1);
    Constant value(0);
    if (plus != 0)
    {
        pANTLR3_BASE_TREE const_expr_tree = GetChild(osrc, 2);
        assert(const_expr_tree != 0);
        assert(GetType(const_expr_tree) == TREE_CONSTANT_EXPR);
        pANTLR3_BASE_TREE const_expr = GetChild(const_expr_tree, 0);
        assert(const_expr != 0);
        value = Eval(K_S32, const_expr);
    }

    typedef union TYPES {
        __int64 s64;
        __int32 s32;
        __int16 s16;;
        __int8 s8;
        unsigned __int64 u64;
        unsigned __int32 u32;
        unsigned __int16 u16;
        unsigned __int8 u8;
        float f32;
        double f64;
    } TYPES;
    TYPES * d = (TYPES*)sdst->pvalue;
    // Unfortunately, different semantics for different storage classes.
    TYPES * s = 0;
    if (ssrc->storage_class != K_REG)
        s = (TYPES*)ssrc->pvalue;
    else if (plus != 0)
    {
        void * addr = *(void**)ssrc->pvalue;
        switch (value.type)
        {
        case K_U8:
            s = (TYPES*)(((unsigned char *)addr) + value.value.u8);
            break;
        case K_U16:
            s = (TYPES*)(((unsigned char *)addr) + value.value.u16);
            break;
        case K_U32:
            s = (TYPES*)(((unsigned char *)addr) + value.value.u32);
            break;
        case K_U64:
            s = (TYPES*)(((unsigned char *)addr) + value.value.u64);
            break;
        case K_S8:
            s = (TYPES*)(((unsigned char *)addr) + value.value.s8);
            break;
        case K_S16:
            s = (TYPES*)(((unsigned char *)addr) + value.value.s16);
            break;
        case K_S32:
            s = (TYPES*)(((unsigned char *)addr) + value.value.s32);
            break;
        case K_S64:
            s = (TYPES*)(((unsigned char *)addr) + value.value.s64);
            break;
        default:
            assert(false);
        }
    }
    else
        s = (TYPES*)ssrc->pvalue;
    
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

int CUDA_EMULATOR::DoMad(pANTLR3_BASE_TREE inst)
{
    // Multiply+add register and/or constants, and store in a register.
    int start = 0;
    if (GetType(GetChild(inst, start)) == TREE_PRED)
    start++;
    assert(GetType(GetChild(inst, start)) == KI_MAD);
    start++;
    assert(GetType(GetChild(inst, start)) == TREE_TYPE);
    pANTLR3_BASE_TREE ttype = GetChild(inst, start);
    start++;
    pANTLR3_BASE_TREE odst = 0;
    pANTLR3_BASE_TREE osrc1 = 0;
    pANTLR3_BASE_TREE osrc2 = 0;
    pANTLR3_BASE_TREE osrc3 = 0;
    for (;; ++start)
    {
        pANTLR3_BASE_TREE t = GetChild(inst, start);
        if (t == 0)
            break;
        int gt = GetType(t);
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
    pANTLR3_BASE_TREE twidth = 0;
    pANTLR3_BASE_TREE tfrnd = 0;
    for (int i = 0; ; ++i)
    {
        pANTLR3_BASE_TREE t = GetChild(ttype, i);
        if (t == 0)
            break;
        int gt = GetType(t);
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
    int type = GetType(ttype);
    int width = 0;
    if (twidth != 0)
        width = GetType(twidth);

    pANTLR3_BASE_TREE dst = GetChild(odst,0);
    pANTLR3_BASE_TREE src1 = GetChild(osrc1,0);
    pANTLR3_BASE_TREE src2 = GetChild(osrc2,0);
    pANTLR3_BASE_TREE src3 = GetChild(osrc3,0);

    // Supported types of MAD.
    typedef union TYPES {
        __int64 s64;
        __int32 s32;
        __int16 s16;;
        unsigned __int64 u64;
        unsigned __int32 u32;
        unsigned __int16 u16;
        float f32;
        double f64;
    } TYPES;

    Symbol * sdst = 0;
    Symbol * ssrc1 = 0;
    Symbol * ssrc2 = 0;
    Symbol * ssrc3 = 0;
    assert(GetType(dst) == T_WORD);
    sdst = FindSymbol(GetText(dst));
    char * dummy;

    TYPES value1; // used if literal
    TYPES value2; // used if literal
    TYPES value3; // used if literal
    TYPES * d = (TYPES*)sdst->pvalue;
    TYPES * s1 = &value1;
    TYPES * s2 = &value2;
    TYPES * s3 = &value2;

    if (GetType(src1) == TREE_CONSTANT_EXPR)
    {
        Constant c = Eval(type, GetChild(src1, 0));
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
    } else if (GetType(src1) == T_WORD)
    {
        ssrc1 = FindSymbol(GetText(src1));
        assert(ssrc1 != 0);
        s1 = (TYPES*)ssrc1->pvalue;
    } else assert(false);

    if (GetType(src2) == TREE_CONSTANT_EXPR)
    {
        Constant c = Eval(type, GetChild(src2, 0));
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
    } else if (GetType(src2) == T_WORD)
    {
        ssrc2 = FindSymbol(GetText(src2));
        assert(ssrc2 != 0);
        s2 = (TYPES*)ssrc2->pvalue;
    } else assert(false);

    if (GetType(src3) == TREE_CONSTANT_EXPR)
    {
        Constant c = Eval(type, GetChild(src3, 0));
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
    } else if (GetType(src3) == T_WORD)
    {
        ssrc3 = FindSymbol(GetText(src3));
        assert(ssrc3 != 0);
        s3 = (TYPES*)ssrc3->pvalue;
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

int CUDA_EMULATOR::DoMov(pANTLR3_BASE_TREE inst)
{
    // Assign source to destination.
    int start = 0;
    if (GetType(GetChild(inst, start)) == TREE_PRED)
        start++;
    assert(GetType(GetChild(inst, start)) == KI_MOV);
    start++;
    pANTLR3_BASE_TREE ttype = 0;
    pANTLR3_BASE_TREE odst = 0;
    pANTLR3_BASE_TREE osrc = 0;
    for (;; ++start)
    {
        pANTLR3_BASE_TREE t = GetChild(inst, start);
        if (t == 0)
            break;
        int gt = GetType(t);
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
        pANTLR3_BASE_TREE t = GetChild(ttype, i);
        if (t == 0)
            break;
        int gt = GetType(t);
        if (gt == K_PRED
              || gt == K_U16 || gt == K_U32 || gt == K_U64
              || gt == K_S16 || gt == K_S32 || gt == K_S64
              || gt == K_F32 || gt == K_F64)
            ttype = t;
        else assert(false);
    }
    assert(ttype);
    int type = GetType(ttype);
    pANTLR3_BASE_TREE dst = GetChild(odst, 0);
    pANTLR3_BASE_TREE src = GetChild(osrc,0);
    Symbol * sdst = 0;
    Symbol * ssrc = 0;
    if (dst->getType(dst) == T_WORD)
    {
        sdst = FindSymbol(GetText(dst));
    } else assert(false);

    typedef union TYPES {
        __int64 s64;
        __int32 s32;
        __int16 s16;;
        __int8 s8;
        unsigned __int64 u64;
        unsigned __int32 u32;
        unsigned __int16 u16;
        unsigned __int8 u8;
        float f32;
        double f64;
    } TYPES;
    TYPES * d;
    TYPES * s;
    TYPES value;
    char * dummy;

    d = (TYPES*)sdst->pvalue;
    s = &value;

    if (GetType(src) == TREE_CONSTANT_EXPR)
    {
        Constant c = Eval(type, GetChild(src, 0));
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
    } else if (GetType(src) == T_WORD)
    {
        ssrc = FindSymbol(GetText(src));
        // Various types of id's to handle:
        assert(ssrc != 0);
        if (strcmp(ssrc->type, "dim3") == 0)
        {
            // Get qualifier of the structure.
            pANTLR3_BASE_TREE tqual = (pANTLR3_BASE_TREE)osrc->getChild(osrc, 1);
            assert(tqual != 0);
            int qual = GetType(tqual);
            if (qual == K_X)
            {
                switch (type)
                {
                    case K_U16:
                        s->u16 = ((dim3*)ssrc->pvalue)->x;
                        break;
                    case K_S16:
                        s->s16 = ((dim3*)ssrc->pvalue)->x;
                        break;
                    case K_U32:
                        s->u32 = ((dim3*)ssrc->pvalue)->x;
                        break;
                    case K_S32:
                        s->s32 = ((dim3*)ssrc->pvalue)->x;
                        break;
                    case K_S64:
                        s->s64 = ((dim3*)ssrc->pvalue)->x;
                        break;
                    case K_U64:
                        s->u64 = ((dim3*)ssrc->pvalue)->x;
                        break;
                    default:
                        assert(false);
                }
            } else if (qual == K_Y)
            {
                switch (type)
                {
                    case K_U16:
                        s->u16 = ((dim3*)ssrc->pvalue)->y;
                        break;
                    case K_S16:
                        s->s16 = ((dim3*)ssrc->pvalue)->y;
                        break;
                    case K_U32:
                        s->u32 = ((dim3*)ssrc->pvalue)->y;
                        break;
                    case K_S32:
                        s->s32 = ((dim3*)ssrc->pvalue)->y;
                        break;
                    case K_S64:
                        s->s64 = ((dim3*)ssrc->pvalue)->y;
                        break;
                    case K_U64:
                        s->u64 = ((dim3*)ssrc->pvalue)->y;
                        break;
                    default:
                        assert(false);
                }
            } else if (qual == K_Z)
            {
                switch (type)
                {
                    case K_U16:
                        s->u16 = ((dim3*)ssrc->pvalue)->z;
                        break;
                    case K_S16:
                        s->s16 = ((dim3*)ssrc->pvalue)->z;
                        break;
                    case K_U32:
                        s->u32 = ((dim3*)ssrc->pvalue)->z;
                        break;
                    case K_S32:
                        s->s32 = ((dim3*)ssrc->pvalue)->z;
                        break;
                    case K_S64:
                        s->s64 = ((dim3*)ssrc->pvalue)->z;
                        break;
                    case K_U64:
                        s->u64 = ((dim3*)ssrc->pvalue)->z;
                        break;
                    default:
                        assert(false);
                }
            }
            else assert(false);
        } else {
            s = (TYPES*)ssrc->pvalue;
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

int CUDA_EMULATOR::DoMul(pANTLR3_BASE_TREE inst)
{
    // Multiply register and/or constants, and store in a register.
    int start = 0;
    if (GetType(GetChild(inst, start)) == TREE_PRED)
        start++;
    assert(GetType(GetChild(inst, start)) == KI_MUL);
    start++;
    assert(GetType(GetChild(inst, start)) == TREE_TYPE);
    pANTLR3_BASE_TREE ttype = GetChild(inst, start);
    start++;
    pANTLR3_BASE_TREE odst = 0;
    pANTLR3_BASE_TREE osrc1 = 0;
    pANTLR3_BASE_TREE osrc2 = 0;
    for (;; ++start)
    {
        pANTLR3_BASE_TREE t = GetChild(inst, start);
        if (t == 0)
            break;
        int gt = GetType(t);
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
    pANTLR3_BASE_TREE twidth = 0;
    for (int i = 0; ; ++i)
    {
        pANTLR3_BASE_TREE t = GetChild(ttype, i);
        if (t == 0)
            break;
        int gt = GetType(t);
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
    int type = GetType(ttype);
    int width = 0;
    if (twidth != 0)
        width = GetType(twidth);

    pANTLR3_BASE_TREE dst = GetChild(odst,0);
    pANTLR3_BASE_TREE src1 = GetChild(osrc1,0);
    pANTLR3_BASE_TREE src2 = GetChild(osrc2,0);

    // Supported types of MUL.
    typedef union TYPES {
        long s64;
        int s32;
        short s16;
        unsigned long u64;
        unsigned int u32;
        unsigned short u16;
        float f32;
        double f64;
    } TYPES;

    Symbol * sdst = 0;
    Symbol * ssrc1 = 0;
    Symbol * ssrc2 = 0;
    assert(GetType(dst) == T_WORD);
    sdst = FindSymbol(GetText(dst));
    char * dummy;

    TYPES value1; // used if literal
    TYPES value2; // used if literal
    TYPES * d = (TYPES*)sdst->pvalue;
    TYPES * s1 = &value1;
    TYPES * s2 = &value2;

    if (GetType(src1) == TREE_CONSTANT_EXPR)
    {
        Constant c = Eval(type, GetChild(src1, 0));
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
    } else if (GetType(src1) == T_WORD)
    {
        ssrc1 = FindSymbol(GetText(src1));
        assert(ssrc1 != 0);
        s1 = (TYPES*)ssrc1->pvalue;
    } else assert(false);

    if (GetType(src2) == TREE_CONSTANT_EXPR)
    {
        Constant c = Eval(type, GetChild(src2, 0));
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
    } else if (GetType(src2) == T_WORD)
    {
        ssrc2 = FindSymbol(GetText(src2));
        assert(ssrc2 != 0);
        s2 = (TYPES*)ssrc2->pvalue;
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

int CUDA_EMULATOR::DoMul24(pANTLR3_BASE_TREE inst)
{
    // Multiply 24-bit integer numbers, in register and/or constants,
    // and store in a register.
    int start = 0;
    if (GetType(GetChild(inst, start)) == TREE_PRED)
        start++;
    assert(GetType(GetChild(inst, start)) == KI_MUL24);
    start++;
    assert(GetType(GetChild(inst, start)) == TREE_TYPE);
    pANTLR3_BASE_TREE ttype = GetChild(inst, start);
    start++;
    pANTLR3_BASE_TREE odst = 0;
    pANTLR3_BASE_TREE osrc1 = 0;
    pANTLR3_BASE_TREE osrc2 = 0;
    for (;; ++start)
    {
        pANTLR3_BASE_TREE t = GetChild(inst, start);
        if (t == 0)
            break;
        int gt = GetType(t);
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
    pANTLR3_BASE_TREE twidth = 0;
    for (int i = 0; ; ++i)
    {
        pANTLR3_BASE_TREE t = GetChild(ttype, i);
        if (t == 0)
            break;
        int gt = GetType(t);
        if (gt == K_U32 || gt == K_S32)
            ttype = t;
        else if (gt== K_LO || gt == K_HI)
            twidth = t;
        else assert(false);
    }
    assert(ttype != 0);
    int type = GetType(ttype);
    int width = 0;
    if (twidth != 0)
        width = GetType(twidth);

    pANTLR3_BASE_TREE dst = GetChild(odst,0);
    pANTLR3_BASE_TREE src1 = GetChild(osrc1,0);
    pANTLR3_BASE_TREE src2 = GetChild(osrc2,0);

    // Supported types of MUL24.
    typedef union TYPES {
        int s32;
        unsigned int u32;
    } TYPES;

    Symbol * sdst = 0;
    Symbol * ssrc1 = 0;
    Symbol * ssrc2 = 0;
    assert(GetType(dst) == T_WORD);
    sdst = FindSymbol(GetText(dst));
    char * dummy;

    TYPES value1; // used if literal
    TYPES value2; // used if literal
    TYPES * d = (TYPES*)sdst->pvalue;
    TYPES * s1 = &value1;
    TYPES * s2 = &value2;

    if (GetType(src1) == TREE_CONSTANT_EXPR)
    {
        Constant c = Eval(type, GetChild(src1, 0));
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
    } else if (GetType(src1) == T_WORD)
    {
        ssrc1 = FindSymbol(GetText(src1));
        assert(ssrc1 != 0);
        s1 = (TYPES*)ssrc1->pvalue;
    } else assert(false);

    if (GetType(src2) == TREE_CONSTANT_EXPR)
    {
        Constant c = Eval(type, GetChild(src2, 0));
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
    } else if (GetType(src2) == T_WORD)
    {
        ssrc2 = FindSymbol(GetText(src2));
        assert(ssrc2 != 0);
        s2 = (TYPES*)ssrc2->pvalue;
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

int CUDA_EMULATOR::DoSetp(pANTLR3_BASE_TREE inst)
{
    int start = 0;
    if (GetType(GetChild(inst, start)) == TREE_PRED)
        start++;
    assert(GetType(GetChild(inst, start)) == KI_SETP);
    start++;
    pANTLR3_BASE_TREE ttype = 0;
    pANTLR3_BASE_TREE odst1 = 0;
    pANTLR3_BASE_TREE odst2 = 0;
    pANTLR3_BASE_TREE osrc1 = 0;
    pANTLR3_BASE_TREE osrc2 = 0;
    pANTLR3_BASE_TREE osrc3 = 0;
    for (;; ++start)
    {
        pANTLR3_BASE_TREE t = GetChild(inst, start);
        if (t == 0)
            break;
        int gt = GetType(t);
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
    assert(GetChild(odst1, 1) == 0);
    bool ftz = false;
    int op = 0;
    for (int i = 0; ; ++i)
    {
        pANTLR3_BASE_TREE t = GetChild(ttype, i);
        if (t == 0)
            break;
        int gt = GetType(t);
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

    int type = GetType(ttype);
    pANTLR3_BASE_TREE dst1 = GetChild(odst1,0);
    pANTLR3_BASE_TREE src1 = GetChild(osrc1,0);
    pANTLR3_BASE_TREE src2 = GetChild(osrc2,0);

    Symbol * sdst1 = 0;
    Symbol * ssrc1 = 0;
    Symbol * ssrc2 = 0;
    if (dst1->getType(dst1) == T_WORD)
    {
        sdst1 = FindSymbol(GetText(dst1));
    } else assert(false);
    if (src1->getType(src1) == T_WORD)
    {
        ssrc1 = FindSymbol(GetText(src1));
    } else assert(false);
    if (src2->getType(src2) == T_WORD)
    {
        ssrc2 = FindSymbol(GetText(src2));
    } else assert(false);

    TYPES * d = (TYPES*)sdst1->pvalue;
    TYPES * s1 = (TYPES*)ssrc1->pvalue;
    TYPES * s2 = (TYPES*)ssrc2->pvalue;
    
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

int CUDA_EMULATOR::DoSt(pANTLR3_BASE_TREE inst)
{
    int start = 0;
    if (GetType(GetChild(inst, start)) == TREE_PRED)
        start++;
    pANTLR3_BASE_TREE odst = GetChild(inst, start+2);
    pANTLR3_BASE_TREE dst = GetChild(odst,0);
    pANTLR3_BASE_TREE osrc = GetChild(inst, start+3);
    pANTLR3_BASE_TREE src = GetChild(osrc,0);
    Symbol * sdst = 0;
    Symbol * ssrc = 0;
    if (dst->getType(dst) == T_WORD)
    {
        sdst = FindSymbol(GetText(dst));
    } else assert(false);
    if (src->getType(src) == T_WORD)
    {
        ssrc = FindSymbol(GetText(src));
    } else assert(false);
    *(int*) (((TYPES*)sdst->pvalue)->u32) = ((TYPES*)ssrc->pvalue)->u32;
    return 0;
}

int CUDA_EMULATOR::DoSub(pANTLR3_BASE_TREE inst)
{
    int start = 0;
    if (GetType(GetChild(inst, start)) == TREE_PRED)
        start++;
    assert(GetType(GetChild(inst, start)) == KI_SUB);
    start++;
    pANTLR3_BASE_TREE ttype = 0;
    pANTLR3_BASE_TREE odst = 0;
    pANTLR3_BASE_TREE osrc1 = 0;
    pANTLR3_BASE_TREE osrc2 = 0;
    for (;; ++start)
    {
        pANTLR3_BASE_TREE t = GetChild(inst, start);
        if (t == 0)
            break;
        int gt = GetType(t);
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
    for (int i = 0; ; ++i)
    {
        pANTLR3_BASE_TREE t = GetChild(ttype, i);
        if (t == 0)
            break;
        int gt = GetType(t);
        if (gt == K_SAT)
            sat = true;
        else if (gt == K_U16 || gt == K_U32 || gt == K_U64
                 || gt == K_S16 || gt == K_S32 || gt == K_S64)
            ttype = t;
        else assert(false);
    }
    assert(ttype != 0);
    assert(sat == 0);
    int type = GetType(ttype);
    pANTLR3_BASE_TREE dst = GetChild(odst,0);
    pANTLR3_BASE_TREE src1 = GetChild(osrc1,0);
    pANTLR3_BASE_TREE src2 = GetChild(osrc2,0);

    Symbol * sdst = 0;
    if (dst->getType(dst) == T_WORD)
    {
        sdst = FindSymbol(GetText(dst));
    } else assert(false);
    typedef union TYPES {
        __int64 s64;
        __int32 s32;
        __int16 s16;;
        unsigned __int64 u64;
        unsigned __int32 u32;
        unsigned __int16 u16;
    } TYPES;

    TYPES value1;
    TYPES value2;
    char * dummy;
    TYPES * d = (TYPES*)sdst->pvalue;
    TYPES * s1 = &value1;
    TYPES * s2 = &value2;

    if (GetType(src1) == TREE_CONSTANT_EXPR)
    {
        Constant c = Eval(type, GetChild(src1, 0));
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
    } else if (GetType(src1) == T_WORD)
    {
        Symbol * ssrc1 = FindSymbol(GetText(src1));
        assert(ssrc1 != 0);
        assert(ssrc1->size == Sizeof(type));
        TYPES * psrc1_value = (TYPES*)ssrc1->pvalue;
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
            default:
                assert(false);
        }
    } else assert(false);

    if (GetType(src2) == TREE_CONSTANT_EXPR)
    {
        Constant c = Eval(type, GetChild(src2, 0));
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
    } else if (GetType(src2) == T_WORD)
    {
        Symbol * ssrc2 = FindSymbol(GetText(src2));
        assert(ssrc2 != 0);
        assert(ssrc2->size == Sizeof(type));
        TYPES * psrc2_value = (TYPES*)ssrc2->pvalue;
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
            default:
                assert(false);
        }
    } else assert(false);

    switch (type)
    {
        case K_U16:
            d->u16 = s1->u16 - s2->u16;
            break;
        case K_S16:
            d->s16 = s1->s16 - s2->s16;
            break;
        case K_U32:
            d->u32 = s1->u32 - s2->u32;
            break;
        case K_S32:
            d->s32 = s1->s32 - s2->s32;
            break;
        case K_S64:
            d->s64 = s1->s64 - s2->s64;
            break;
        case K_U64:
            d->u64 = s1->u64 - s2->u64;
            break;
        default:
            assert(false);
    }
    return 0;
}
