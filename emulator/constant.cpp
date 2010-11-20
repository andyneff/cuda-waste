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
#include "emulator.h"
#include <assert.h>
#include <fstream>
#include <iostream>
#include <queue>
#include "tree.h"
#include <process.h>    /* _beginthread, _endthread */
#include "thread.h"
#include "symbol-table.h"
#include "string-table.h"
#include "symbol.h"
#include "constant.h"
#include "types.h"
#include "../wrapper/memdbg.h"

#define new new(_CLIENT_BLOCK,__FILE__, __LINE__)


CONSTANT CONSTANT::Eval(int expected_type, TREE * const_expr)
{
    // Perform bottom-up evaluation of a constant expression.
    CONSTANT result;
    result.type = expected_type;
    char * dummy;
    char * text = const_expr->GetText();
    if (const_expr->GetType() == T_DEC_LITERAL)
    {
        switch (expected_type)
        {
            case K_U8:
                result.value.u8 = _strtoi64(text, &dummy, 10);
                break;
            case K_U16:
                result.value.u16 = _strtoi64(text, &dummy, 10);
                break;
            case K_U32:
                result.value.u32 = _strtoi64(text, &dummy, 10);
                break;
            case K_U64:
                result.value.u64 = _strtoi64(text, &dummy, 10);
                break;
            case K_S8:
                result.value.u8 = _strtoi64(text, &dummy, 10);
                break;
            case K_S16:
                result.value.s16 = _strtoi64(text, &dummy, 10);
                break;
            case K_S32:
                result.value.s32 = _strtoi64(text, &dummy, 10);
                break;
            case K_S64:
                result.value.s64 = _strtoi64(text, &dummy, 10);
                break;
            case K_B8:
                result.value.b8 = _strtoi64(text, &dummy, 10);
                break;
            case K_B16:
                result.value.b16 = _strtoi64(text, &dummy, 10);
                break;
            case K_B32:
                result.value.b32 = _strtoi64(text, &dummy, 10);
                break;
            case K_B64:
                result.value.b64 = _strtoi64(text, &dummy, 10);
                break;
            default:
                assert(false);
        }
    } else if (const_expr->GetType() == T_HEX_LITERAL)
    {
        text += 2;
        switch (expected_type)
        {
            case K_U8:
                result.value.u8 = _strtoi64(text, &dummy, 16);
                break;
            case K_U16:
                result.value.u16 = _strtoi64(text, &dummy, 16);
                break;
            case K_U32:
                result.value.u32 = _strtoi64(text, &dummy, 16);
                break;
            case K_U64:
                result.value.u64 = _strtoi64(text, &dummy, 16);
                break;
            case K_S8:
                result.value.u8 = _strtoi64(text, &dummy, 16);
                break;
            case K_S16:
                result.value.s16 = _strtoi64(text, &dummy, 16);
                break;
            case K_S32:
                result.value.s32 = _strtoi64(text, &dummy, 16);
                break;
            case K_S64:
                result.value.s64 = _strtoi64(text, &dummy, 16);
                break;
            case K_B8:
                result.value.b8 = _strtoi64(text, &dummy, 16);
                break;
            case K_B16:
                result.value.b16 = _strtoi64(text, &dummy, 16);
                break;
            case K_B32:
                result.value.b32 = _strtoi64(text, &dummy, 16);
                break;
            case K_B64:
                result.value.b64 = _strtoi64(text, &dummy, 16);
                break;
            default:
                assert(false);
        }
    } else if (const_expr->GetType() == T_FLT_LITERAL)
    {
        // Three cases:
        // "0F...", or "0f..." (hex 32-bit float)
        // "0D...", or "0f..." (hex 64-bit float)
        // "3.14159..." (float with decimal point)
        int len = strlen(text);
        if (len >= 2 && (text[1] == 'f' || text[1] == 'F'))
        {
            text += 2;
            switch (expected_type)
            {
                case K_F32:
                    result.value.u32 = _strtoi64(text, &dummy, 16);
                    break;
                case K_F64:
                    result.value.u64 = _strtoi64(text, &dummy, 16);
                    break;
                default:
                    assert(false);
            }
        } else if (len >= 2 && (text[1] == 'd' || text[1] == 'D'))
        {
            text += 2;
            switch (expected_type)
            {
                case K_F32:
                    result.value.u32 = _strtoi64(text, &dummy, 16);
                    break;
                case K_F64:
                    result.value.u64 = _strtoi64(text, &dummy, 16);
                    break;
                default:
                    assert(false);
            }
        } else
        {
            switch (expected_type)
            {
                case K_F32:
                    result.value.f32 = strtod(text, &dummy);
                    break;
                case K_F64:
                    result.value.f64 = strtod(text, &dummy);
                    break;
                default:
                    assert(false);
            }
        }
    } else if (const_expr->GetType() == T_QUESTION)
    {
        throw new EMULATOR::EMU_ERROR("Question operator in constant expression not supported.\n");
    } else if (const_expr->GetType() == T_OROR)
    {
        // Perform boolean OR.
        CONSTANT lhs = Eval(expected_type, const_expr->GetChild(0));
        CONSTANT rhs = Eval(expected_type, const_expr->GetChild(1));
        switch (expected_type)
        {
            case K_U8:
                result.value.u8 = lhs.value.u8 || rhs.value.u8;
                break;
            case K_U16:
                result.value.u16 = lhs.value.u16 || rhs.value.u16;
                break;
            case K_U32:
                result.value.u32 = lhs.value.u32 || rhs.value.u32;
                break;
            case K_U64:
                result.value.u64 = lhs.value.u64 || rhs.value.u64;
                break;
            case K_S8:
                result.value.s8 = lhs.value.s8 || rhs.value.s8;
                break;
            case K_S16:
                result.value.s16 = lhs.value.s16 || rhs.value.s16;
                break;
            case K_S32:
                result.value.s32 = lhs.value.s32 || rhs.value.s32;
                break;
            case K_S64:
                result.value.s64 = lhs.value.s64 || rhs.value.s64;
                break;
            case K_B8:
                result.value.b8 = lhs.value.b8 || rhs.value.b8;
                break;
            case K_B16:
                result.value.b16 = lhs.value.b16 || rhs.value.b16;
                break;
            case K_B32:
                result.value.b32 = lhs.value.b32 || rhs.value.b32;
                break;
            case K_B64:
                result.value.b64 = lhs.value.b64 || rhs.value.b64;
                break;
            default:
                assert(false);
        }
    } else if (const_expr->GetType() == T_ANDAND)
    {
        // Perform boolean AND.
        CONSTANT lhs = Eval(expected_type, const_expr->GetChild(0));
        CONSTANT rhs = Eval(expected_type, const_expr->GetChild(1));
        switch (expected_type)
        {
            case K_U8:
                result.value.u8 = lhs.value.u8 && rhs.value.u8;
                break;
            case K_U16:
                result.value.u16 = lhs.value.u16 && rhs.value.u16;
                break;
            case K_U32:
                result.value.u32 = lhs.value.u32 && rhs.value.u32;
                break;
            case K_U64:
                result.value.u64 = lhs.value.u64 && rhs.value.u64;
                break;
            case K_S8:
                result.value.s8 = lhs.value.s8 && rhs.value.s8;
                break;
            case K_S16:
                result.value.s16 = lhs.value.s16 && rhs.value.s16;
                break;
            case K_S32:
                result.value.s32 = lhs.value.s32 && rhs.value.s32;
                break;
            case K_S64:
                result.value.s64 = lhs.value.s64 && rhs.value.s64;
                break;
            case K_B8:
                result.value.b8 = lhs.value.b8 && rhs.value.b8;
                break;
            case K_B16:
                result.value.b16 = lhs.value.b16 && rhs.value.b16;
                break;
            case K_B32:
                result.value.b32 = lhs.value.b32 && rhs.value.b32;
                break;
            case K_B64:
                result.value.b64 = lhs.value.b64 && rhs.value.b64;
                break;
            default:
                assert(false);
        }
    } else if (const_expr->GetType() == T_OR)
    {
        // Perform bit OR.
        CONSTANT lhs = Eval(expected_type, const_expr->GetChild(0));
        CONSTANT rhs = Eval(expected_type, const_expr->GetChild(1));
        switch (expected_type)
        {
            case K_U8:
                result.value.u8 = lhs.value.u8 | rhs.value.u8;
                break;
            case K_U16:
                result.value.u16 = lhs.value.u16 | rhs.value.u16;
                break;
            case K_U32:
                result.value.u32 = lhs.value.u32 | rhs.value.u32;
                break;
            case K_U64:
                result.value.u64 = lhs.value.u64 | rhs.value.u64;
                break;
            case K_S8:
                result.value.s8 = lhs.value.s8 | rhs.value.s8;
                break;
            case K_S16:
                result.value.s16 = lhs.value.s16 | rhs.value.s16;
                break;
            case K_S32:
                result.value.s32 = lhs.value.s32 | rhs.value.s32;
                break;
            case K_S64:
                result.value.s64 = lhs.value.s64 | rhs.value.s64;
                break;
            case K_B8:
                result.value.b8 = lhs.value.b8 | rhs.value.b8;
                break;
            case K_B16:
                result.value.b16 = lhs.value.b16 | rhs.value.b16;
                break;
            case K_B32:
                result.value.b32 = lhs.value.b32 | rhs.value.b32;
                break;
            case K_B64:
                result.value.b64 = lhs.value.b64 | rhs.value.b64;
                break;
            default:
                assert(false);
        }
    } else if (const_expr->GetType() == T_XOR)
    {
        // Perform bit XOR.
        CONSTANT lhs = Eval(expected_type, const_expr->GetChild(0));
        CONSTANT rhs = Eval(expected_type, const_expr->GetChild(1));
        switch (expected_type)
        {
            case K_U8:
                result.value.u8 = lhs.value.u8 ^ rhs.value.u8;
                break;
            case K_U16:
                result.value.u16 = lhs.value.u16 ^ rhs.value.u16;
                break;
            case K_U32:
                result.value.u32 = lhs.value.u32 ^ rhs.value.u32;
                break;
            case K_U64:
                result.value.u64 = lhs.value.u64 ^ rhs.value.u64;
                break;
            case K_S8:
                result.value.s8 = lhs.value.s8 ^ rhs.value.s8;
                break;
            case K_S16:
                result.value.s16 = lhs.value.s16 ^ rhs.value.s16;
                break;
            case K_S32:
                result.value.s32 = lhs.value.s32 ^ rhs.value.s32;
                break;
            case K_S64:
                result.value.s64 = lhs.value.s64 ^ rhs.value.s64;
                break;
            case K_B8:
                result.value.b8 = lhs.value.b8 ^ rhs.value.b8;
                break;
            case K_B16:
                result.value.b16 = lhs.value.b16 ^ rhs.value.b16;
                break;
            case K_B32:
                result.value.b32 = lhs.value.b32 ^ rhs.value.b32;
                break;
            case K_B64:
                result.value.b64 = lhs.value.b64 ^ rhs.value.b64;
                break;
            default:
                assert(false);
        }
    } else if (const_expr->GetType() == T_AND)
    {
        // Perform bit OR.
        CONSTANT lhs = Eval(expected_type, const_expr->GetChild(0));
        CONSTANT rhs = Eval(expected_type, const_expr->GetChild(1));
        switch (expected_type)
        {
            case K_U8:
                result.value.u8 = lhs.value.u8 & rhs.value.u8;
                break;
            case K_U16:
                result.value.u16 = lhs.value.u16 & rhs.value.u16;
                break;
            case K_U32:
                result.value.u32 = lhs.value.u32 & rhs.value.u32;
                break;
            case K_U64:
                result.value.u64 = lhs.value.u64 & rhs.value.u64;
                break;
            case K_S8:
                result.value.s8 = lhs.value.s8 & rhs.value.s8;
                break;
            case K_S16:
                result.value.s16 = lhs.value.s16 & rhs.value.s16;
                break;
            case K_S32:
                result.value.s32 = lhs.value.s32 & rhs.value.s32;
                break;
            case K_S64:
                result.value.s64 = lhs.value.s64 & rhs.value.s64;
                break;
            case K_B8:
                result.value.b8 = lhs.value.b8 & rhs.value.b8;
                break;
            case K_B16:
                result.value.b16 = lhs.value.b16 & rhs.value.b16;
                break;
            case K_B32:
                result.value.b32 = lhs.value.b32 & rhs.value.b32;
                break;
            case K_B64:
                result.value.b64 = lhs.value.b64 & rhs.value.b64;
                break;
            default:
                assert(false);
        }
    } else if (const_expr->GetType() == T_EQEQ)
    {
        // Perform EQ.
        CONSTANT lhs = Eval(expected_type, const_expr->GetChild(0));
        CONSTANT rhs = Eval(expected_type, const_expr->GetChild(1));
        switch (expected_type)
        {
            case K_U8:
                result.value.u8 = lhs.value.u8 == rhs.value.u8;
                break;
            case K_U16:
                result.value.u16 = lhs.value.u16 == rhs.value.u16;
                break;
            case K_U32:
                result.value.u32 = lhs.value.u32 == rhs.value.u32;
                break;
            case K_U64:
                result.value.u64 = lhs.value.u64 == rhs.value.u64;
                break;
            case K_S8:
                result.value.s8 = lhs.value.s8 == rhs.value.s8;
                break;
            case K_S16:
                result.value.s16 = lhs.value.s16 == rhs.value.s16;
                break;
            case K_S32:
                result.value.s32 = lhs.value.s32 == rhs.value.s32;
                break;
            case K_S64:
                result.value.s64 = lhs.value.s64 == rhs.value.s64;
                break;
            case K_B8:
                result.value.b8 = lhs.value.b8 == rhs.value.b8;
                break;
            case K_B16:
                result.value.b16 = lhs.value.b16 == rhs.value.b16;
                break;
            case K_B32:
                result.value.b32 = lhs.value.b32 == rhs.value.b32;
                break;
            case K_B64:
                result.value.b64 = lhs.value.b64 == rhs.value.b64;
                break;
            default:
                assert(false);
        }
    } else if (const_expr->GetType() == T_NOTEQ)
    {
        // Perform bit !=.
        CONSTANT lhs = Eval(expected_type, const_expr->GetChild(0));
        CONSTANT rhs = Eval(expected_type, const_expr->GetChild(1));
        switch (expected_type)
        {
            case K_U8:
                result.value.u8 = lhs.value.u8 != rhs.value.u8;
                break;
            case K_U16:
                result.value.u16 = lhs.value.u16 != rhs.value.u16;
                break;
            case K_U32:
                result.value.u32 = lhs.value.u32 != rhs.value.u32;
                break;
            case K_U64:
                result.value.u64 = lhs.value.u64 != rhs.value.u64;
                break;
            case K_S8:
                result.value.s8 = lhs.value.s8 != rhs.value.s8;
                break;
            case K_S16:
                result.value.s16 = lhs.value.s16 != rhs.value.s16;
                break;
            case K_S32:
                result.value.s32 = lhs.value.s32 != rhs.value.s32;
                break;
            case K_S64:
                result.value.s64 = lhs.value.s64 != rhs.value.s64;
                break;
            case K_B8:
                result.value.b8 = lhs.value.b8 != rhs.value.b8;
                break;
            case K_B16:
                result.value.b16 = lhs.value.b16 != rhs.value.b16;
                break;
            case K_B32:
                result.value.b32 = lhs.value.b32 != rhs.value.b32;
                break;
            case K_B64:
                result.value.b64 = lhs.value.b64 != rhs.value.b64;
                break;
            default:
                assert(false);
        }
    } else if (const_expr->GetType() == T_LE)
    {
        // Perform bit LE.
        CONSTANT lhs = Eval(expected_type, const_expr->GetChild(0));
        CONSTANT rhs = Eval(expected_type, const_expr->GetChild(1));
        switch (expected_type)
        {
            case K_U8:
                result.value.u8 = lhs.value.u8 <= rhs.value.u8;
                break;
            case K_U16:
                result.value.u16 = lhs.value.u16 <= rhs.value.u16;
                break;
            case K_U32:
                result.value.u32 = lhs.value.u32 <= rhs.value.u32;
                break;
            case K_U64:
                result.value.u64 = lhs.value.u64 <= rhs.value.u64;
                break;
            case K_S8:
                result.value.s8 = lhs.value.s8 <= rhs.value.s8;
                break;
            case K_S16:
                result.value.s16 = lhs.value.s16 <= rhs.value.s16;
                break;
            case K_S32:
                result.value.s32 = lhs.value.s32 <= rhs.value.s32;
                break;
            case K_S64:
                result.value.s64 = lhs.value.s64 <= rhs.value.s64;
                break;
            case K_B8:
                result.value.b8 = lhs.value.b8 <= rhs.value.b8;
                break;
            case K_B16:
                result.value.b16 = lhs.value.b16 <= rhs.value.b16;
                break;
            case K_B32:
                result.value.b32 = lhs.value.b32 <= rhs.value.b32;
                break;
            case K_B64:
                result.value.b64 = lhs.value.b64 <= rhs.value.b64;
                break;
            default:
                assert(false);
        }
    } else if (const_expr->GetType() == T_GE)
    {
        // Perform bit GE
        CONSTANT lhs = Eval(expected_type, const_expr->GetChild(0));
        CONSTANT rhs = Eval(expected_type, const_expr->GetChild(1));
        switch (expected_type)
        {
            case K_U8:
                result.value.u8 = lhs.value.u8 >= rhs.value.u8;
                break;
            case K_U16:
                result.value.u16 = lhs.value.u16 >= rhs.value.u16;
                break;
            case K_U32:
                result.value.u32 = lhs.value.u32 >= rhs.value.u32;
                break;
            case K_U64:
                result.value.u64 = lhs.value.u64 >= rhs.value.u64;
                break;
            case K_S8:
                result.value.s8 = lhs.value.s8 >= rhs.value.s8;
                break;
            case K_S16:
                result.value.s16 = lhs.value.s16 >= rhs.value.s16;
                break;
            case K_S32:
                result.value.s32 = lhs.value.s32 >= rhs.value.s32;
                break;
            case K_S64:
                result.value.s64 = lhs.value.s64 >= rhs.value.s64;
                break;
            case K_B8:
                result.value.b8 = lhs.value.b8 >= rhs.value.b8;
                break;
            case K_B16:
                result.value.b16 = lhs.value.b16 >= rhs.value.b16;
                break;
            case K_B32:
                result.value.b32 = lhs.value.b32 >= rhs.value.b32;
                break;
            case K_B64:
                result.value.b64 = lhs.value.b64 >= rhs.value.b64;
                break;
            default:
                assert(false);
        }
    } else if (const_expr->GetType() == T_LT)
    {
        // Perform bit <.
        CONSTANT lhs = Eval(expected_type, const_expr->GetChild(0));
        CONSTANT rhs = Eval(expected_type, const_expr->GetChild(1));
        switch (expected_type)
        {
            case K_U8:
                result.value.u8 = lhs.value.u8 < rhs.value.u8;
                break;
            case K_U16:
                result.value.u16 = lhs.value.u16 < rhs.value.u16;
                break;
            case K_U32:
                result.value.u32 = lhs.value.u32 < rhs.value.u32;
                break;
            case K_U64:
                result.value.u64 = lhs.value.u64 < rhs.value.u64;
                break;
            case K_S8:
                result.value.s8 = lhs.value.s8 < rhs.value.s8;
                break;
            case K_S16:
                result.value.s16 = lhs.value.s16 < rhs.value.s16;
                break;
            case K_S32:
                result.value.s32 = lhs.value.s32 < rhs.value.s32;
                break;
            case K_S64:
                result.value.s64 = lhs.value.s64 < rhs.value.s64;
                break;
            case K_B8:
                result.value.b8 = lhs.value.b8 < rhs.value.b8;
                break;
            case K_B16:
                result.value.b16 = lhs.value.b16 < rhs.value.b16;
                break;
            case K_B32:
                result.value.b32 = lhs.value.b32 < rhs.value.b32;
                break;
            case K_B64:
                result.value.b64 = lhs.value.b64 < rhs.value.b64;
                break;
            default:
                assert(false);
        }
    } else if (const_expr->GetType() == T_GT)
    {
        // Perform bit OR.
        CONSTANT lhs = Eval(expected_type, const_expr->GetChild(0));
        CONSTANT rhs = Eval(expected_type, const_expr->GetChild(1));
        switch (expected_type)
        {
            case K_U8:
                result.value.u8 = lhs.value.u8 > rhs.value.u8;
                break;
            case K_U16:
                result.value.u16 = lhs.value.u16 > rhs.value.u16;
                break;
            case K_U32:
                result.value.u32 = lhs.value.u32 > rhs.value.u32;
                break;
            case K_U64:
                result.value.u64 = lhs.value.u64 > rhs.value.u64;
                break;
            case K_S8:
                result.value.s8 = lhs.value.s8 > rhs.value.s8;
                break;
            case K_S16:
                result.value.s16 = lhs.value.s16 > rhs.value.s16;
                break;
            case K_S32:
                result.value.s32 = lhs.value.s32 > rhs.value.s32;
                break;
            case K_S64:
                result.value.s64 = lhs.value.s64 > rhs.value.s64;
                break;
            case K_B8:
                result.value.b8 = lhs.value.b8 > rhs.value.b8;
                break;
            case K_B16:
                result.value.b16 = lhs.value.b16 > rhs.value.b16;
                break;
            case K_B32:
                result.value.b32 = lhs.value.b32 > rhs.value.b32;
                break;
            case K_B64:
                result.value.b64 = lhs.value.b64 > rhs.value.b64;
                break;
            default:
                assert(false);
        }
    } else if (const_expr->GetType() == T_LTLT)
    {
        // Perform bit <<
        CONSTANT lhs = Eval(expected_type, const_expr->GetChild(0));
        CONSTANT rhs = Eval(expected_type, const_expr->GetChild(1));
        switch (expected_type)
        {
            case K_U8:
                result.value.u8 = lhs.value.u8 << rhs.value.u8;
                break;
            case K_U16:
                result.value.u16 = lhs.value.u16 << rhs.value.u16;
                break;
            case K_U32:
                result.value.u32 = lhs.value.u32 << rhs.value.u32;
                break;
            case K_U64:
                result.value.u64 = lhs.value.u64 << rhs.value.u64;
                break;
            case K_S8:
                result.value.s8 = lhs.value.s8 << rhs.value.s8;
                break;
            case K_S16:
                result.value.s16 = lhs.value.s16 << rhs.value.s16;
                break;
            case K_S32:
                result.value.s32 = lhs.value.s32 << rhs.value.s32;
                break;
            case K_S64:
                result.value.s64 = lhs.value.s64 << rhs.value.s64;
                break;
            case K_B8:
                result.value.b8 = lhs.value.b8 << rhs.value.b8;
                break;
            case K_B16:
                result.value.b16 = lhs.value.b16 << rhs.value.b16;
                break;
            case K_B32:
                result.value.b32 = lhs.value.b32 << rhs.value.b32;
                break;
            case K_B64:
                result.value.b64 = lhs.value.b64 << rhs.value.b64;
                break;
            default:
                assert(false);
        }
    } else if (const_expr->GetType() == T_GTGT)
    {
        // Perform bit >>
        CONSTANT lhs = Eval(expected_type, const_expr->GetChild(0));
        CONSTANT rhs = Eval(expected_type, const_expr->GetChild(1));
        switch (expected_type)
        {
            case K_U8:
                result.value.u8 = lhs.value.u8 >> rhs.value.u8;
                break;
            case K_U16:
                result.value.u16 = lhs.value.u16 >> rhs.value.u16;
                break;
            case K_U32:
                result.value.u32 = lhs.value.u32 >> rhs.value.u32;
                break;
            case K_U64:
                result.value.u64 = lhs.value.u64 >> rhs.value.u64;
                break;
            case K_S8:
                result.value.s8 = lhs.value.s8 >> rhs.value.s8;
                break;
            case K_S16:
                result.value.s16 = lhs.value.s16 >> rhs.value.s16;
                break;
            case K_S32:
                result.value.s32 = lhs.value.s32 >> rhs.value.s32;
                break;
            case K_S64:
                result.value.s64 = lhs.value.s64 >> rhs.value.s64;
                break;
            case K_B8:
                result.value.b8 = lhs.value.b8 >> rhs.value.b8;
                break;
            case K_B16:
                result.value.b16 = lhs.value.b16 >> rhs.value.b16;
                break;
            case K_B32:
                result.value.b32 = lhs.value.b32 >> rhs.value.b32;
                break;
            case K_B64:
                result.value.b64 = lhs.value.b64 >> rhs.value.b64;
                break;
            default:
                assert(false);
        }
    } else if (const_expr->GetType() == T_PLUS && const_expr->GetChild(1) != 0)
    {
        // Perform bit '+'.
        CONSTANT lhs = Eval(expected_type, const_expr->GetChild(0));
        CONSTANT rhs = Eval(expected_type, const_expr->GetChild(1));
        switch (expected_type)
        {
            case K_U8:
                result.value.u8 = lhs.value.u8 + rhs.value.u8;
                break;
            case K_U16:
                result.value.u16 = lhs.value.u16 + rhs.value.u16;
                break;
            case K_U32:
                result.value.u32 = lhs.value.u32 + rhs.value.u32;
                break;
            case K_U64:
                result.value.u64 = lhs.value.u64 + rhs.value.u64;
                break;
            case K_S8:
                result.value.s8 = lhs.value.s8 + rhs.value.s8;
                break;
            case K_S16:
                result.value.s16 = lhs.value.s16 + rhs.value.s16;
                break;
            case K_S32:
                result.value.s32 = lhs.value.s32 + rhs.value.s32;
                break;
            case K_S64:
                result.value.s64 = lhs.value.s64 + rhs.value.s64;
                break;
            case K_B8:
                result.value.b8 = lhs.value.b8 + rhs.value.b8;
                break;
            case K_B16:
                result.value.b16 = lhs.value.b16 + rhs.value.b16;
                break;
            case K_B32:
                result.value.b32 = lhs.value.b32 + rhs.value.b32;
                break;
            case K_B64:
                result.value.b64 = lhs.value.b64 + rhs.value.b64;
                break;
            default:
                assert(false);
        }
    } else if (const_expr->GetType() == T_MINUS && const_expr->GetChild(1) != 0)
    {
        // Perform bit '-'
        CONSTANT lhs = Eval(expected_type, const_expr->GetChild(0));
        CONSTANT rhs = Eval(expected_type, const_expr->GetChild(1));
        switch (expected_type)
        {
            case K_U8:
                result.value.u8 = lhs.value.u8 - rhs.value.u8;
                break;
            case K_U16:
                result.value.u16 = lhs.value.u16 - rhs.value.u16;
                break;
            case K_U32:
                result.value.u32 = lhs.value.u32 - rhs.value.u32;
                break;
            case K_U64:
                result.value.u64 = lhs.value.u64 - rhs.value.u64;
                break;
            case K_S8:
                result.value.s8 = lhs.value.s8 - rhs.value.s8;
                break;
            case K_S16:
                result.value.s16 = lhs.value.s16 - rhs.value.s16;
                break;
            case K_S32:
                result.value.s32 = lhs.value.s32 - rhs.value.s32;
                break;
            case K_S64:
                result.value.s64 = lhs.value.s64 - rhs.value.s64;
                break;
            case K_B8:
                result.value.b8 = lhs.value.b8 - rhs.value.b8;
                break;
            case K_B16:
                result.value.b16 = lhs.value.b16 - rhs.value.b16;
                break;
            case K_B32:
                result.value.b32 = lhs.value.b32 - rhs.value.b32;
                break;
            case K_B64:
                result.value.b64 = lhs.value.b64 - rhs.value.b64;
                break;
            default:
                assert(false);
        }
    } else if (const_expr->GetType() == T_STAR)
    {
        // Perform bit '*'
        CONSTANT lhs = Eval(expected_type, const_expr->GetChild(0));
        CONSTANT rhs = Eval(expected_type, const_expr->GetChild(1));
        switch (expected_type)
        {
            case K_U8:
                result.value.u8 = lhs.value.u8 * rhs.value.u8;
                break;
            case K_U16:
                result.value.u16 = lhs.value.u16 * rhs.value.u16;
                break;
            case K_U32:
                result.value.u32 = lhs.value.u32 * rhs.value.u32;
                break;
            case K_U64:
                result.value.u64 = lhs.value.u64 * rhs.value.u64;
                break;
            case K_S8:
                result.value.s8 = lhs.value.s8 * rhs.value.s8;
                break;
            case K_S16:
                result.value.s16 = lhs.value.s16 * rhs.value.s16;
                break;
            case K_S32:
                result.value.s32 = lhs.value.s32 * rhs.value.s32;
                break;
            case K_S64:
                result.value.s64 = lhs.value.s64 * rhs.value.s64;
                break;
            case K_B8:
                result.value.b8 = lhs.value.b8 * rhs.value.b8;
                break;
            case K_B16:
                result.value.b16 = lhs.value.b16 * rhs.value.b16;
                break;
            case K_B32:
                result.value.b32 = lhs.value.b32 * rhs.value.b32;
                break;
            case K_B64:
                result.value.b64 = lhs.value.b64 * rhs.value.b64;
                break;
            default:
                assert(false);
        }
    } else if (const_expr->GetType() == T_SLASH)
    {
        // Perform bit '/'
        CONSTANT lhs = Eval(expected_type, const_expr->GetChild(0));
        CONSTANT rhs = Eval(expected_type, const_expr->GetChild(1));
        switch (expected_type)
        {
            case K_U8:
                result.value.u8 = lhs.value.u8 / rhs.value.u8;
                break;
            case K_U16:
                result.value.u16 = lhs.value.u16 / rhs.value.u16;
                break;
            case K_U32:
                result.value.u32 = lhs.value.u32 / rhs.value.u32;
                break;
            case K_U64:
                result.value.u64 = lhs.value.u64 / rhs.value.u64;
                break;
            case K_S8:
                result.value.s8 = lhs.value.s8 / rhs.value.s8;
                break;
            case K_S16:
                result.value.s16 = lhs.value.s16 / rhs.value.s16;
                break;
            case K_S32:
                result.value.s32 = lhs.value.s32 / rhs.value.s32;
                break;
            case K_S64:
                result.value.s64 = lhs.value.s64 / rhs.value.s64;
                break;
            case K_B8:
                result.value.b8 = lhs.value.b8 / rhs.value.b8;
                break;
            case K_B16:
                result.value.b16 = lhs.value.b16 / rhs.value.b16;
                break;
            case K_B32:
                result.value.b32 = lhs.value.b32 / rhs.value.b32;
                break;
            case K_B64:
                result.value.b64 = lhs.value.b64 / rhs.value.b64;
                break;
            default:
                assert(false);
        }
    } else if (const_expr->GetType() == T_PERCENT)
    {
        // Perform bit '%'
        CONSTANT lhs = Eval(expected_type, const_expr->GetChild(0));
        CONSTANT rhs = Eval(expected_type, const_expr->GetChild(1));
        switch (expected_type)
        {
            case K_U8:
                result.value.u8 = lhs.value.u8 % rhs.value.u8;
                break;
            case K_U16:
                result.value.u16 = lhs.value.u16 % rhs.value.u16;
                break;
            case K_U32:
                result.value.u32 = lhs.value.u32 % rhs.value.u32;
                break;
            case K_U64:
                result.value.u64 = lhs.value.u64 % rhs.value.u64;
                break;
            case K_S8:
                result.value.s8 = lhs.value.s8 % rhs.value.s8;
                break;
            case K_S16:
                result.value.s16 = lhs.value.s16 % rhs.value.s16;
                break;
            case K_S32:
                result.value.s32 = lhs.value.s32 % rhs.value.s32;
                break;
            case K_S64:
                result.value.s64 = lhs.value.s64 % rhs.value.s64;
                break;
            case K_B8:
                result.value.b8 = lhs.value.b8 % rhs.value.b8;
                break;
            case K_B16:
                result.value.b16 = lhs.value.b16 % rhs.value.b16;
                break;
            case K_B32:
                result.value.b32 = lhs.value.b32 % rhs.value.b32;
                break;
            case K_B64:
                result.value.b64 = lhs.value.b64 % rhs.value.b64;
                break;
            default:
                assert(false);
        }
    } else if (const_expr->GetType() == T_PLUS)
    {
        // Perform bit >>
        CONSTANT lhs = Eval(expected_type, const_expr->GetChild(0));
        switch (expected_type)
        {
            case K_U8:
                result.value.u8 = lhs.value.u8;
                break;
            case K_U16:
                result.value.u16 = lhs.value.u16;
                break;
            case K_U32:
                result.value.u32 = lhs.value.u32;
                break;
            case K_U64:
                result.value.u64 = lhs.value.u64;
                break;
            case K_S8:
                result.value.s8 = lhs.value.s8;
                break;
            case K_S16:
                result.value.s16 = lhs.value.s16;
                break;
            case K_S32:
                result.value.s32 = lhs.value.s32;
                break;
            case K_S64:
                result.value.s64 = lhs.value.s64;
                break;
            case K_B8:
                result.value.b8 = lhs.value.b8;
                break;
            case K_B16:
                result.value.b16 = lhs.value.b16;
                break;
            case K_B32:
                result.value.b32 = lhs.value.b32;
                break;
            case K_B64:
                result.value.b64 = lhs.value.b64;
                break;
            default:
                assert(false);
        }
    } else if (const_expr->GetType() == T_MINUS)
    {
        // Perform bit >>
        CONSTANT lhs = Eval(expected_type, const_expr->GetChild(0));
        switch (expected_type)
        {
            case K_U8:
                result.value.u8 = - lhs.value.u8;
                break;
            case K_U16:
                result.value.u16 = - lhs.value.u16;
                break;
            case K_U32:
                result.value.u32 = - lhs.value.u32;
                break;
            case K_U64:
                result.value.u64 = - lhs.value.u64;
                break;
            case K_S8:
                result.value.s8 = - lhs.value.s8;
                break;
            case K_S16:
                result.value.s16 = - lhs.value.s16;
                break;
            case K_S32:
                result.value.s32 = - lhs.value.s32;
                break;
            case K_S64:
                result.value.s64 = - lhs.value.s64;
                break;
            case K_B8:
                result.value.b8 = - lhs.value.b8;
                break;
            case K_B16:
                result.value.b16 = - lhs.value.b16;
                break;
            case K_B32:
                result.value.b32 = - lhs.value.b32;
                break;
            case K_B64:
                result.value.b64 = - lhs.value.b64;
                break;
            default:
                assert(false);
        }
    } else if (const_expr->GetType() == T_TILDE)
    {
        // Perform bit >>
        CONSTANT lhs = Eval(expected_type, const_expr->GetChild(0));
        switch (expected_type)
        {
            case K_U8:
                result.value.u8 = ~ lhs.value.u8;
                break;
            case K_U16:
                result.value.u16 = ~ lhs.value.u16;
                break;
            case K_U32:
                result.value.u32 = ~ lhs.value.u32;
                break;
            case K_U64:
                result.value.u64 = ~ lhs.value.u64;
                break;
            case K_S8:
                result.value.s8 = ~ lhs.value.s8;
                break;
            case K_S16:
                result.value.s16 = ~ lhs.value.s16;
                break;
            case K_S32:
                result.value.s32 = ~ lhs.value.s32;
                break;
            case K_S64:
                result.value.s64 = ~ lhs.value.s64;
                break;
            case K_B8:
                result.value.b8 = ~ lhs.value.b8;
                break;
            case K_B16:
                result.value.b16 = ~ lhs.value.b16;
                break;
            case K_B32:
                result.value.b32 = ~ lhs.value.b32;
                break;
            case K_B64:
                result.value.b64 = ~ lhs.value.b64;
                break;
            default:
                assert(false);
        }
    } else if (const_expr->GetType() == T_NOT)
    {
        // Perform !
        CONSTANT lhs = Eval(expected_type, const_expr->GetChild(0));
        switch (expected_type)
        {
            case K_U8:
                result.value.u8 = ! lhs.value.u8;
                break;
            case K_U16:
                result.value.u16 = ! lhs.value.u16;
                break;
            case K_U32:
                result.value.u32 = ! lhs.value.u32;
                break;
            case K_U64:
                result.value.u64 = ! lhs.value.u64;
                break;
            case K_S8:
                result.value.s8 = ! lhs.value.s8;
                break;
            case K_S16:
                result.value.s16 = ! lhs.value.s16;
                break;
            case K_S32:
                result.value.s32 = ! lhs.value.s32;
                break;
            case K_S64:
                result.value.s64 = ! lhs.value.s64;
                break;
            case K_B8:
                result.value.b8 = ! lhs.value.b8;
                break;
            case K_B16:
                result.value.b16 = ! lhs.value.b16;
                break;
            case K_B32:
                result.value.b32 = ! lhs.value.b32;
                break;
            case K_B64:
                result.value.b64 = ! lhs.value.b64;
                break;
            default:
                assert(false);
        }
    }
    else
        assert(false);
    return result;
}
