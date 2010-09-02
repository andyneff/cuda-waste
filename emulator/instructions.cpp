#include "emulator.h"
#include <assert.h>
#include <iostream>

void CUDA_EMULATOR::DoAdd(pANTLR3_BASE_TREE inst)
{
    std::cout << "ADD\n";
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
    TYPES * psrc1_value;
    TYPES * psrc2_value;

    if (GetType(src1) == T_DEC_LITERAL)
    {
        switch (type)
        {
            case K_U16:
                value1.u16 = atoi(GetText(src1));
                break;
            case K_S16:
                value1.s16 = atoi(GetText(src1));
                break;
            case K_U32:
                value1.u32 = atoi(GetText(src1));
                break;
            case K_S32:
                value1.s32 = atoi(GetText(src1));
                break;
            case K_S64:
                value1.s64 = _atoi64(GetText(src1));
                break;
            case K_U64:
                value1.u64 = _atoi64(GetText(src1));
                break;
            default:
                assert(false);
        }
    } else if (GetType(src1) == T_HEX_LITERAL)
    {
        switch (type)
        {
            case K_U16:
                value1.u16 = strtol(GetText(src1)+2, &dummy, 16);
                break;
            case K_S16:
                value1.s16 = strtol(GetText(src1)+2, &dummy, 16);
                break;
            case K_U32:
                value1.u32 = strtol(GetText(src1)+2, &dummy, 16);
                break;
            case K_S32:
                value1.s32 = strtol(GetText(src1)+2, &dummy, 16);
                break;
            case K_U64:
                value1.u64 = _strtoi64(GetText(src1)+2, &dummy, 16);
                break;
            case K_S64:
                value1.s64 = _strtoi64(GetText(src1)+2, &dummy, 16);
                break;
            default:
                assert(false);
        }
    } else if (GetType(src1) == T_WORD)
    {
        Symbol * ssrc1 = FindSymbol(GetText(src1));
        assert(ssrc1 != 0);
        assert(ssrc1->size == Sizeof(type));
        psrc1_value = (TYPES*)ssrc1->lvalue;
        switch (type)
        {
            case K_U16:
                value1.u16 = psrc1_value->u16;
                break;
            case K_S16:
                value1.s16 = psrc1_value->s16;
                break;
            case K_U32:
                value1.u32 = psrc1_value->u32;
                break;
            case K_S32:
                value1.s32 = psrc1_value->s32;
                break;
            case K_U64:
                value1.u64 = psrc1_value->u64;
                break;
            case K_S64:
                value1.s64 = psrc1_value->s64;
                break;
            default:
                assert(false);
        }
    } else assert(false);

    if (GetType(src2) == T_DEC_LITERAL)
    {
        switch (type)
        {
            case K_U16:
                value2.u16 = atoi(GetText(src2));
                break;
            case K_S16:
                value2.s16 = atoi(GetText(src2));
                break;
            case K_U32:
                value2.u32 = atoi(GetText(src2));
                break;
            case K_S32:
                value2.s32 = atoi(GetText(src2));
                break;
            case K_S64:
                value2.s64 = _atoi64(GetText(src2));
                break;
            case K_U64:
                value2.u64 = _atoi64(GetText(src2));
                break;
            default:
                assert(false);
        }
    } else if (GetType(src2) == T_HEX_LITERAL)
    {
        switch (type)
        {
            case K_U16:
                value2.u16 = strtol(GetText(src2)+2, &dummy, 16);
                break;
            case K_S16:
                value2.s16 = strtol(GetText(src2)+2, &dummy, 16);
                break;
            case K_U32:
                value2.u32 = strtol(GetText(src2)+2, &dummy, 16);
                break;
            case K_S32:
                value2.s32 = strtol(GetText(src2)+2, &dummy, 16);
                break;
            case K_U64:
                value2.u64 = _strtoi64(GetText(src2)+2, &dummy, 16);
                break;
            case K_S64:
                value2.s64 = _strtoi64(GetText(src2)+2, &dummy, 16);
                break;
            default:
                assert(false);
        }
    } else if (GetType(src2) == T_WORD)
    {
        Symbol * ssrc2 = FindSymbol(GetText(src2));
        assert(ssrc2 != 0);
        assert(ssrc2->size == Sizeof(type));
        psrc2_value = (TYPES*)ssrc2->lvalue;
        switch (type)
        {
            case K_U16:
                value2.u16 = psrc2_value->u16;
                break;
            case K_S16:
                value2.s16 = psrc2_value->s16;
                break;
            case K_U32:
                value2.u32 = psrc2_value->u32;
                break;
            case K_S32:
                value2.s32 = psrc2_value->s32;
                break;
            case K_U64:
                value2.u64 = psrc2_value->u64;
                break;
            case K_S64:
                value2.s64 = psrc2_value->s64;
                break;
            default:
                assert(false);
        }
    } else assert(false);

    psrc1_value = &value1;
    psrc2_value = &value2;
    TYPES * pdst_value = (TYPES*)sdst->lvalue;

    switch (type)
    {
        case K_U16:
            pdst_value->u16 = psrc1_value->u16 + psrc2_value->u16;
            break;
        case K_S16:
            pdst_value->s16 = psrc1_value->s16 + psrc2_value->s16;
            break;
        case K_U32:
            pdst_value->u32 = psrc1_value->u32 + psrc2_value->u32;
            break;
        case K_S32:
            pdst_value->s32 = psrc1_value->s32 + psrc2_value->s32;
            break;
        case K_S64:
            pdst_value->s64 = psrc1_value->s64 + psrc2_value->s64;
            break;
        case K_U64:
            pdst_value->u64 = psrc1_value->u64 + psrc2_value->u64;
            break;
        default:
            assert(false);
    }
}

int CUDA_EMULATOR::DoBra(pANTLR3_BASE_TREE inst)
{
    std::cout << "BRA\n";
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
    return (int)sdst->lvalue;
}

void CUDA_EMULATOR::DoCvt(pANTLR3_BASE_TREE inst)
{
    std::cout << "CVT\n";
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

    dst_value = (TYPES*)s1->lvalue;

    // handle .x, .y, .z stuff.
    if (strcmp(s2->type, "dim3") == 0)
    {
        // Get qualifier of the structure.
        pANTLR3_BASE_TREE tqual = GetChild(o2, 1);
        assert(tqual != 0);
        int qual = GetType(tqual);
        if (qual == K_X)
        {
            src_value = (TYPES*) &(((dim3*)s2->lvalue)->x);
        } else if (qual == K_Y)
        {
            src_value = (TYPES*) &(((dim3*)s2->lvalue)->y);
        } else if (qual == K_Z)
        {
            src_value = (TYPES*) &(((dim3*)s2->lvalue)->z);
        } else assert(false);
    } else
        src_value = (TYPES*)s2->lvalue;

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
}

void CUDA_EMULATOR::DoDiv(pANTLR3_BASE_TREE inst)
{
    std::cout << "DIV\n";
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
    pdst_value = (TYPES*)sdst->lvalue;
	char * dummy;

    if (GetType(src1) == T_DEC_LITERAL)
    {
        psrc1_value = &src1_value;
        switch (type)
        {
            case K_U16:
                psrc1_value->u16 = atoi(GetText(src1));
                break;
            case K_S16:
                psrc1_value->s16 = atoi(GetText(src1));
                break;
            case K_U32:
                psrc1_value->u32 = atoi(GetText(src1));
                break;
            case K_S32:
                psrc1_value->s32 = atoi(GetText(src1));
                break;
            default:
                assert(false);
        }
    } else if (GetType(src1) == T_HEX_LITERAL)
    {
        psrc1_value = &src1_value;
        switch (type)
        {
            case K_U16:
                psrc1_value->u16 = strtol(GetText(src1)+2, &dummy, 16);
                break;
            case K_S16:
                psrc1_value->s16 = strtol(GetText(src1)+2, &dummy, 16);
                break;
            case K_U32:
                psrc1_value->u32 = strtol(GetText(src1)+2, &dummy, 16);
                break;
            case K_S32:
                psrc1_value->s32 = strtol(GetText(src1)+2, &dummy, 16);
                break;
            default:
                assert(false);
        }
    } else if (GetType(src1) == T_WORD)
    {
        ssrc1 = FindSymbol(GetText(src1));
        assert(ssrc1 != 0);
        psrc1_value = (TYPES*)ssrc1->lvalue;
    } else assert(false);

    if (GetType(src2) == T_DEC_LITERAL)
    {
        psrc2_value = &src2_value;
        switch (type)
        {
            case K_U16:
                psrc2_value->u16 = atoi(GetText(src2));
                break;
            case K_S16:
                psrc2_value->s16 = atoi(GetText(src2));
                break;
            case K_U32:
                psrc2_value->u32 = atoi(GetText(src2));
                break;
            case K_S32:
                psrc2_value->s32 = atoi(GetText(src2));
                break;
            default:
                assert(false);
        }
    } else if (GetType(src2) == T_HEX_LITERAL)
    {
        psrc2_value = &src2_value;
        switch (type)
        {
            case K_U16:
                psrc2_value->u16 = strtol(GetText(src2)+2, &dummy, 16);
                break;
            case K_S16:
                psrc2_value->s16 = strtol(GetText(src2)+2, &dummy, 16);
                break;
            case K_U32:
                psrc2_value->u32 = strtol(GetText(src2)+2, &dummy, 16);
                break;
            case K_S32:
                psrc2_value->s32 = strtol(GetText(src2)+2, &dummy, 16);
                break;
            default:
                assert(false);
        }
    } else if (GetType(src2) == T_WORD)
    {
        ssrc2 = FindSymbol(GetText(src2));
        assert(ssrc2 != 0);
        psrc2_value = (TYPES*)ssrc2->lvalue;
    } else assert(false);

    switch (type)
    {
	case K_U16:
	    pdst_value->u16 = psrc1_value->u16 / psrc2_value->u16;
	    break;
	case K_S16:
	    pdst_value->s16 = psrc1_value->s16 / psrc2_value->s16;
	    break;
	case K_U32:
	    pdst_value->u32 = psrc1_value->u32 / psrc2_value->u32;
	    break;
	case K_S32:
	    pdst_value->s32 = psrc1_value->s32 / psrc2_value->s32;
	    break;
	case K_U64:
	    pdst_value->u64 = psrc1_value->u64 / psrc2_value->u64;
	    break;
	case K_S64:
	    pdst_value->s64 = psrc1_value->s64 / psrc2_value->s64;
	    break;
	case K_F32:
	    pdst_value->f32 = psrc1_value->f32 / psrc2_value->f32;
	    break;
	case K_F64:
	    pdst_value->f64 = psrc1_value->f64 / psrc2_value->f64;
	    break;
        default:
            assert(false);
    }
}

void CUDA_EMULATOR::DoExit(pANTLR3_BASE_TREE inst)
{
    std::cout << "EXIT\n";
}

void CUDA_EMULATOR::DoLd(pANTLR3_BASE_TREE inst)
{
    std::cout << "LD\n";
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
    assert(ss != 0);
    int type = GetType(ttype);

    // Get two operands, assign source to destination.
    pANTLR3_BASE_TREE dst = GetChild(odst, 0);
    pANTLR3_BASE_TREE src = GetChild(osrc, 0);
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
    TYPES * d = (TYPES*)sdst->lvalue;
    TYPES * s = (TYPES*)ssrc->lvalue;
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
}

void CUDA_EMULATOR::DoMov(pANTLR3_BASE_TREE inst)
{
    std::cout << "MOV\n";
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

    d = (TYPES*)sdst->lvalue;
    s = &value;
    
    if (GetType(src) == T_DEC_LITERAL)
    {
        switch (type)
        {
            case K_U16:
                s->u16 = atoi(GetText(src));
                break;
            case K_S16:
                s->s16 = atoi(GetText(src));
                break;
            case K_U32:
                s->u32 = atoi(GetText(src));
                break;
            case K_S32:
                s->s32 = atoi(GetText(src));
                break;
            case K_S64:
                s->s64 = _atoi64(GetText(src));
                break;
            case K_U64:
                s->u64 = _atoi64(GetText(src));
                break;
            default:
                assert(false);
        }
    } else if (GetType(src) == T_HEX_LITERAL)
    {
        switch (type)
        {
            case K_U16:
                s->u16 = strtol(GetText(src)+2, &dummy, 16);
                break;
            case K_S16:
                s->s16 = strtol(GetText(src)+2, &dummy, 16);
                break;
            case K_U32:
                s->u32 = strtol(GetText(src)+2, &dummy, 16);
                break;
            case K_S32:
                s->s32 = strtol(GetText(src)+2, &dummy, 16);
                break;
            case K_U64:
                s->u64 = _strtoi64(GetText(src)+2, &dummy, 16);
                break;
            case K_S64:
                s->s64 = _strtoi64(GetText(src)+2, &dummy, 16);
                break;
            default:
                assert(false);
        }
    } else if (GetType(src) == T_FLT_LITERAL)
    {
        switch (type)
        {
            case K_F32:
                s->u32 = strtol(GetText(src)+2, &dummy, 16);
                break;
            case K_F64:
                s->u64 = _strtoi64(GetText(src)+2, &dummy, 16);
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
                        s->u16 = ((dim3*)ssrc->lvalue)->x;
                        break;
                    case K_S16:
                        s->s16 = ((dim3*)ssrc->lvalue)->x;
                        break;
                    case K_U32:
                        s->u32 = ((dim3*)ssrc->lvalue)->x;
                        break;
                    case K_S32:
                        s->s32 = ((dim3*)ssrc->lvalue)->x;
                        break;
                    case K_S64:
                        s->s64 = ((dim3*)ssrc->lvalue)->x;
                        break;
                    case K_U64:
                        s->u64 = ((dim3*)ssrc->lvalue)->x;
                        break;
                    default:
                        assert(false);
                }
            } else if (qual == K_Y)
            {
                switch (type)
                {
                    case K_U16:
                        s->u16 = ((dim3*)ssrc->lvalue)->y;
                        break;
                    case K_S16:
                        s->s16 = ((dim3*)ssrc->lvalue)->y;
                        break;
                    case K_U32:
                        s->u32 = ((dim3*)ssrc->lvalue)->y;
                        break;
                    case K_S32:
                        s->s32 = ((dim3*)ssrc->lvalue)->y;
                        break;
                    case K_S64:
                        s->s64 = ((dim3*)ssrc->lvalue)->y;
                        break;
                    case K_U64:
                        s->u64 = ((dim3*)ssrc->lvalue)->y;
                        break;
                    default:
                        assert(false);
                }
            } else if (qual == K_Z)
            {
                switch (type)
                {
                    case K_U16:
                        s->u16 = ((dim3*)ssrc->lvalue)->z;
                        break;
                    case K_S16:
                        s->s16 = ((dim3*)ssrc->lvalue)->z;
                        break;
                    case K_U32:
                        s->u32 = ((dim3*)ssrc->lvalue)->z;
                        break;
                    case K_S32:
                        s->s32 = ((dim3*)ssrc->lvalue)->z;
                        break;
                    case K_S64:
                        s->s64 = ((dim3*)ssrc->lvalue)->z;
                        break;
                    case K_U64:
                        s->u64 = ((dim3*)ssrc->lvalue)->z;
                        break;
                    default:
                        assert(false);
                }
            }
            else assert(false);
        } else {
            s = (TYPES*)ssrc->lvalue;
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
}

void CUDA_EMULATOR::DoMul(pANTLR3_BASE_TREE inst)
{
    std::cout << "MUL\n";
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
    pdst_value = (TYPES*)sdst->lvalue;
	char * dummy;

    if (GetType(src1) == T_DEC_LITERAL)
    {
        psrc1_value = &src1_value;
        switch (type)
        {
            case K_U16:
                psrc1_value->u16 = atoi(GetText(src1));
                break;
            case K_S16:
                psrc1_value->s16 = atoi(GetText(src1));
                break;
            case K_U32:
                psrc1_value->u32 = atoi(GetText(src1));
                break;
            case K_S32:
                psrc1_value->s32 = atoi(GetText(src1));
                break;
            default:
                assert(false);
        }
    } else if (GetType(src1) == T_HEX_LITERAL)
    {
        psrc1_value = &src1_value;
        switch (type)
        {
            case K_U16:
                psrc1_value->u16 = strtol(GetText(src1)+2, &dummy, 16);
                break;
            case K_S16:
                psrc1_value->s16 = strtol(GetText(src1)+2, &dummy, 16);
                break;
            case K_U32:
                psrc1_value->u32 = strtol(GetText(src1)+2, &dummy, 16);
                break;
            case K_S32:
                psrc1_value->s32 = strtol(GetText(src1)+2, &dummy, 16);
                break;
            default:
                assert(false);
        }
    } else if (GetType(src1) == T_WORD)
    {
        ssrc1 = FindSymbol(GetText(src1));
        assert(ssrc1 != 0);
        psrc1_value = (TYPES*)ssrc1->lvalue;
    } else assert(false);

    if (GetType(src2) == T_DEC_LITERAL)
    {
        psrc2_value = &src2_value;
        switch (type)
        {
            case K_U16:
                psrc2_value->u16 = atoi(GetText(src2));
                break;
            case K_S16:
                psrc2_value->s16 = atoi(GetText(src2));
                break;
            case K_U32:
                psrc2_value->u32 = atoi(GetText(src2));
                break;
            case K_S32:
                psrc2_value->s32 = atoi(GetText(src2));
                break;
            default:
                assert(false);
        }
    } else if (GetType(src2) == T_HEX_LITERAL)
    {
        psrc2_value = &src2_value;
        switch (type)
        {
            case K_U16:
                psrc2_value->u16 = strtol(GetText(src2)+2, &dummy, 16);
                break;
            case K_S16:
                psrc2_value->s16 = strtol(GetText(src2)+2, &dummy, 16);
                break;
            case K_U32:
                psrc2_value->u32 = strtol(GetText(src2)+2, &dummy, 16);
                break;
            case K_S32:
                psrc2_value->s32 = strtol(GetText(src2)+2, &dummy, 16);
                break;
            default:
                assert(false);
        }
    } else if (GetType(src2) == T_WORD)
    {
        ssrc2 = FindSymbol(GetText(src2));
        assert(ssrc2 != 0);
        psrc2_value = (TYPES*)ssrc2->lvalue;
    } else assert(false);
    
    switch (type)
    {
        case K_U16:
            if (width == K_LO)
                pdst_value->u16 = psrc1_value->u16 * psrc2_value->u16;
            else if (width == K_HI)
                pdst_value->u16 = (psrc1_value->u16 * psrc2_value->u16 ) >> 16;
            else if (width == K_WIDE)
                pdst_value->u32 = psrc1_value->u16 * psrc2_value->u16;
            else assert(false);
            break;
        case K_S16:
            if (width == K_LO)
                pdst_value->s16 = psrc1_value->s16 * psrc2_value->s16;
            else if (width == K_HI)
                pdst_value->s16 = (psrc1_value->s16 * psrc2_value->s16 ) >> 16;
            else if (width == K_WIDE)
                pdst_value->s32 = psrc1_value->s16 * psrc2_value->s16;
            else assert(false);
            break;
        case K_U32:
            if (width == K_LO)
                pdst_value->u32 = psrc1_value->u32 * psrc2_value->u32;
            else if (width == K_HI)
                pdst_value->u32 = (psrc1_value->u32 * psrc2_value->u32 ) >> 16;
            else if (width == K_WIDE)
                pdst_value->u64 = psrc1_value->u32 * psrc2_value->u32;
            else assert(false);
            break;
        case K_S32:
            if (width == K_LO)
                pdst_value->s32 = psrc1_value->s32 * psrc2_value->s32;
            else if (width == K_HI)
                pdst_value->s32 = (psrc1_value->s32 * psrc2_value->s32 ) >> 16;
            else if (width == K_WIDE)
                pdst_value->s64 = psrc1_value->s32 * psrc2_value->s32;
            else assert(false);
            break;
        case K_U64:
            if (width == K_LO)
                pdst_value->u64 = psrc1_value->u64 * psrc2_value->u64;
            else if (width == K_HI)
                pdst_value->u64 = (psrc1_value->u64 * psrc2_value->u64 ) >> 16;
            else assert(false);
            break;
        case K_S64:
            if (width == K_LO)
                pdst_value->s64 = psrc1_value->s64 * psrc2_value->s64;
            else if (width == K_HI)
                pdst_value->s64 = (psrc1_value->s64 * psrc2_value->s64 ) >> 16;
            else assert(false);
            break;
        case K_F32:
            pdst_value->f32 = psrc1_value->f32 * psrc2_value->f32;
            break;
        case K_F64:
            pdst_value->f64 = psrc1_value->f64 * psrc2_value->f64;
            break;
        default:
            assert(false);
    }
}

void CUDA_EMULATOR::DoSetp(pANTLR3_BASE_TREE inst)
{
    std::cout << "SETP\n";
    int start = 0;
    if (GetType(GetChild(inst, start)) == TREE_PRED)
        start++;
    pANTLR3_BASE_TREE odst = GetChild(inst, start+2);
    pANTLR3_BASE_TREE dst = GetChild(odst,0);
    pANTLR3_BASE_TREE osrc1 = GetChild(inst, start+3);
    pANTLR3_BASE_TREE src1 = GetChild(osrc1,0);
    pANTLR3_BASE_TREE osrc2 = GetChild(inst, start+4);
    pANTLR3_BASE_TREE src2 = GetChild(osrc2,0);
    Symbol * sdst = 0;
    Symbol * ssrc1 = 0;
    Symbol * ssrc2 = 0;
    if (dst->getType(dst) == T_WORD)
    {
        sdst = FindSymbol(GetText(dst));
    } else assert(false);
    if (src1->getType(src1) == T_WORD)
    {
        ssrc1 = FindSymbol(GetText(src1));
    } else assert(false);
    if (src2->getType(src2) == T_WORD)
    {
        ssrc2 = FindSymbol(GetText(src2));
    } else assert(false);
    char * op = GetText(GetChild(GetChild(inst, start+1), 0));
    if (strcmp(op, ".eq") == 0)
        *((bool*)sdst->lvalue) = *(int*)ssrc1->lvalue == *(int*)ssrc2->lvalue;
    else if (strcmp(op, ".gt") == 0)
        *((bool*)sdst->lvalue) = *(int*)ssrc1->lvalue > *(int*)ssrc2->lvalue;
    else if (strcmp(op, ".lt") == 0)
        *((bool*)sdst->lvalue) = *(int*)ssrc1->lvalue < *(int*)ssrc2->lvalue;
    else if (strcmp(op, ".ge") == 0)
        *((bool*)sdst->lvalue) = *(int*)ssrc1->lvalue >= *(int*)ssrc2->lvalue;
    else if (strcmp(op, ".le") == 0)
        *((bool*)sdst->lvalue) = *(int*)ssrc1->lvalue <= *(int*)ssrc2->lvalue;
    else if (strcmp(op, ".hi") == 0)
        *((bool*)sdst->lvalue) = *(unsigned int*)ssrc1->lvalue > *(unsigned int*)ssrc2->lvalue;
    else if (strcmp(op, ".lo") == 0)
        *((bool*)sdst->lvalue) = *(unsigned int*)ssrc1->lvalue < *(unsigned int*)ssrc2->lvalue;
    else if (strcmp(op, ".hs") == 0)
        *((bool*)sdst->lvalue) = *(unsigned int*)ssrc1->lvalue >= *(unsigned int*)ssrc2->lvalue;
    else if (strcmp(op, ".ls") == 0)
        *((bool*)sdst->lvalue) = *(unsigned int*)ssrc1->lvalue <= *(unsigned int*)ssrc2->lvalue;
}

void CUDA_EMULATOR::DoSt(pANTLR3_BASE_TREE inst)
{
    std::cout << "ST\n";
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
    **((int**)sdst->lvalue) = *(int*)ssrc->lvalue;
}
