/*
 * Software License Agreement (BSD License)
 *
 * Based on original stdio.c released by Atmel
 * Copyright (c) 2008, Atmel Corporation
 * All rights reserved.
 *
 * Modified by Roel Verdult, Copyright (c) 2010
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holders nor the
 * names of its contributors may be used to endorse or promote products
 * derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

//------------------------------------------------------------------------------
//         Headers
//------------------------------------------------------------------------------

#include <stdio.h>
#include <stdarg.h>
#include <stdint.h>
#include <type_traits>
#include "k_stdio.h"

//------------------------------------------------------------------------------
//         Local Definitions
//------------------------------------------------------------------------------

// Maximum string size allowed (in bytes).
#define MAX_STRING_SIZE         255

//------------------------------------------------------------------------------
//         Global Variables
//------------------------------------------------------------------------------

// Required for proper compilation.
//struct _reent r = {0, (FILE*) 0, (FILE*) 1, (FILE*) 0};
//struct _reent *_impure_ptr = &r;

//------------------------------------------------------------------------------
//         Local Functions
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Writes a character inside the given string. Returns 1.
// \param pStr  Storage string.
// \param c  Character to write.
//------------------------------------------------------------------------------
signed int append_char(char *pStr, char c)
{
    *pStr = c;
    return 1;
}

//------------------------------------------------------------------------------
// Writes a string inside the given string.
// Returns the size of the written
// string.
// \param pStr  Storage string.
// \param pSource  Source string.
//------------------------------------------------------------------------------
signed int PutString(char *pStr, char fill, signed int width, const char *pSource)
{
    signed int num = 0;

    while (*pSource != 0) {

        *pStr++ = *pSource++;
        num++;
    }

        width -= num;
        while (width > 0) {

        *pStr++ = fill;
                num++;
                width--;
        }

    return num;
}

//------------------------------------------------------------------------------
// Writes an unsigned int inside the given string, using the provided fill &
// width parameters.
// Returns the size in characters of the written integer.
// \param pStr  Storage string.
// \param fill  Fill character.
// \param width  Minimum integer width.
// \param value  Integer value.
//------------------------------------------------------------------------------
signed int PutUnsignedInt(
    char *pStr,
    char fill,
    signed int width,
    unsigned int value)
{
    signed int num = 0;

    // Take current digit into account when calculating width
    width--;

    // Recursively write upper digits
    if ((value / 10) > 0) {

        num = PutUnsignedInt(pStr, fill, width, value / 10);
        pStr += num;
    }
    // Write filler characters
    else {

        while (width > 0) {

            append_char(pStr, fill);
            pStr++;
            num++;
            width--;
        }
    }

    // Write lower digit
    num += append_char(pStr, (value % 10) + '0');

    return num;
}

//------------------------------------------------------------------------------
// Writes a signed int inside the given string, using the provided fill & width
// parameters.
// Returns the size of the written integer.
// \param pStr  Storage string.
// \param fill  Fill character.
// \param width  Minimum integer width.
// \param value  Signed integer value.
//------------------------------------------------------------------------------
signed int PutSignedInt(
    char *pStr,
    char fill,
    signed int width,
    signed int value)
{
    signed int num = 0;
    unsigned int absolute;

    // Compute absolute value
    if (value < 0) {

        absolute = -value;
    }
    else {

        absolute = value;
    }

    // Take current digit into account when calculating width
    width--;

    // Recursively write upper digits
    if ((absolute / 10) > 0) {

        if (value < 0) {
        
            num = PutSignedInt(pStr, fill, width, -(absolute / 10));
        }
        else {

            num = PutSignedInt(pStr, fill, width, absolute / 10);
        }
        pStr += num;
    }
    else {

        // Reserve space for sign
        if (value < 0) {

            width--;
        }

        // Write filler characters
        while (width > 0) {

            append_char(pStr, fill);
            pStr++;
            num++;
            width--;
        }

        // Write sign
        if (value < 0) {

            num += append_char(pStr, '-');
            pStr++;
        }
    }

    // Write lower digit
    num += append_char(pStr, (absolute % 10) + '0');

    return num;
}

//------------------------------------------------------------------------------
// Writes an hexadecimal value into a string, using the given fill, width &
// capital parameters.
// Returns the number of char written.
// \param pStr  Storage string.
// \param fill  Fill character.
// \param width  Minimum integer width.
// \param maj  Indicates if the letters must be printed in lower- or upper-case.
// \param value  Hexadecimal value.
//------------------------------------------------------------------------------
signed int PutHexa(
    char *pStr,
    char fill,
    signed int width,
    unsigned char maj,
    unsigned int value)
{
    signed int num = 0;

    // Decrement width
    width--;

    // Recursively output upper digits
    if ((value >> 4) > 0) {

        num += PutHexa(pStr, fill, width, maj, value >> 4);
        pStr += num;
    }
    // Write filler chars
    else {

        while (width > 0) {

            append_char(pStr, fill);
            pStr++;
            num++;
            width--;
        }
    }

    // Write current digit
    if ((value & 0xF) < 10) {

        append_char(pStr, (value & 0xF) + '0');
    }
    else if (maj) {

        append_char(pStr, (value & 0xF) - 10 + 'A');
    }
    else {

        append_char(pStr, (value & 0xF) - 10 + 'a');
    }
    num++;

    return num;
}

//------------------------------------------------------------------------------
//         Global Functions
//------------------------------------------------------------------------------


#define _INTSIZEOF(n)   ( (sizeof(n) + sizeof(int) - 1) & ~(sizeof(int) - 1) )

#define my_va_start(ap,v)  ( ap = (va_list)_ADDRESSOF(v) + _INTSIZEOF(v) )
#define my_va_arg(ap,t)    ( *(t *)((ap += _INTSIZEOF(t)) - _INTSIZEOF(t)) )
#define my_va_end(ap)      ( ap = (va_list)0 )

//------------------------------------------------------------------------------
/// Stores the result of a formatted string into another string. Format
/// arguments are given in a va_list instance.
/// Return the number of characters written.
/// \param pStr    Destination string.
/// \param length  Length of Destination string.
/// \param pFormat Format string.
/// \param ap      Argument list.
//------------------------------------------------------------------------------
//template<class  T>
//signed int k_vsnprintf(char *pStr, size_t length, const char *pFormat, va_list ap)
//{
//    char          fill;
//    unsigned char width;
//    signed int    num = 0;
//    signed int    size = 0;
//
//    // Clear the string
//    if (pStr) {
//
//        *pStr = 0;
//    }
//
//    // Phase string
//    while (*pFormat != 0 && size < length) {
//
//        // Normal character
//        if (*pFormat != '%') {
//
//            *pStr++ = *pFormat++;
//            size++;
//        }
//        // Escaped '%'
//        else if (*(pFormat+1) == '%') {
//
//            *pStr++ = '%';
//            pFormat += 2;
//            size++;
//        }
//        // Token delimiter
//        else {
//
//            fill = ' ';
//            width = 0;
//            pFormat++;
//
//            // Parse filler
//            if (*pFormat == '0') {
//
//                fill = '0';
//                pFormat++;
//            }
//
//            // Ignore justifier
//            if (*pFormat == '-') {
//                pFormat++;
//            }
//
//            // Parse width
//            while ((*pFormat >= '0') && (*pFormat <= '9')) {
//        
//                width = (width*10) + *pFormat-'0';
//                pFormat++;
//            }
//
//            // Check if there is enough space
//            if (size + width > length) {
//
//                width = length - size;
//            }
//        
//			// Parse type
//            switch (*pFormat) {
//            case 'd': 
//            case 'i': num = PutSignedInt(pStr, fill, width, my_va_arg(ap, std::make_signed<T>::type)); break;
//            case 'u': num = PutUnsignedInt(pStr, fill, width, my_va_arg(ap, std::make_unsigned<T>::type)); break;
//            case 'x': num = PutHexa(pStr, fill, width, 0, my_va_arg(ap, std::make_unsigned<T>::type)); break;
//            case 'X': num = PutHexa(pStr, fill, width, 1, my_va_arg(ap, std::make_unsigned<T>::type)); break;
//            case 's': num = PutString(pStr, fill, width, my_va_arg(ap, char *)); break;
//            case 'c': num = append_char(pStr, my_va_arg(ap, std::make_unsigned<T>::type)); break;
//            default:
//                return EOF;
//            }
//
//            pFormat++;
//            pStr += num;
//            size += num;
//        }
//    }
//
//    // NULL-terminated (final \0 is not counted)
//    if (size < length) {
//
//        *pStr = 0;
//    }
//    else {
//
//        *(--pStr) = 0;
//        size--;
//    }
//
//    return size;
//}

extern "C" int _output(char *stream, const char *format, va_list args);

template<class  T>
signed int k_vsnprintf(char *pStr, size_t length, const char *pFormat, va_list ap)
{
	return _output(pStr, pFormat, ap);
}

//------------------------------------------------------------------------------
/// Stores the result of a formatted string into another string. Format
/// arguments are given in a va_list instance.
/// Return the number of characters written.
/// \param pString Destination string.
/// \param length  Length of Destination string.
/// \param pFormat Format string.
/// \param ...     Other arguments
//------------------------------------------------------------------------------
template<class  T>
signed int k_snprintf(char *pString, size_t length, const char *pFormat, ...)
{
    va_list    ap;
    signed int rc;

    va_start(ap, pFormat);
    rc = k_vsnprintf<T>(pString, length, pFormat, ap);
    va_end(ap);

    return rc;
}

//------------------------------------------------------------------------------
/// Stores the result of a formatted string into another string. Format
/// arguments are given in a va_list instance.
/// Return the number of characters written.
/// \param pString  Destination string.
/// \param pFormat  Format string.
/// \param ap       Argument list.
//------------------------------------------------------------------------------
template<class  T>
signed int k_vsprintf(char *pString, const char *pFormat, va_list ap)
{
    return k_vsnprintf<T>(pString, MAX_STRING_SIZE, pFormat, ap);
}

//------------------------------------------------------------------------------
/// Outputs a formatted string on the DBGU stream. Format arguments are given
/// in a va_list instance.
/// \param pFormat  Format string
/// \param ap  Argument list.
//------------------------------------------------------------------------------
template<class  T>
signed int k_vprintf(const char *pFormat, va_list ap)
{
  char pStr[MAX_STRING_SIZE];
  char pError[] = "stdio.c: increase MAX_STRING_SIZE\r\n";
  
  // Write formatted string in buffer
  if (k_vsprintf<T>(pStr, pFormat, ap) >= MAX_STRING_SIZE) {
    
    puts(pError);
    while (1); // Increase MAX_STRING_SIZE
  }
  
  // Display string
  return puts(pStr);
}

//------------------------------------------------------------------------------
/// Outputs a formatted string on the DBGU stream, using a variable number of
/// arguments.
/// \param pFormat  Format string.
//------------------------------------------------------------------------------
template<class  T>
signed int k_printf(const char *pFormat, ...)
{
    va_list ap;
    signed int result;

    // Forward call to vprintf
    va_start(ap, pFormat);
    result = k_vprintf<T>(pFormat, ap);
    va_end(ap);

    return result;
}


//------------------------------------------------------------------------------
/// Writes a formatted string inside another string.
/// \param pStr  Storage string.
/// \param pFormat  Format string.
//------------------------------------------------------------------------------
template<class  T>
signed int k_sprintf(char *pStr, const char *pFormat, ...)
{
    va_list ap;
    signed int result;

    // Forward call to vsprintf
    va_start(ap, pFormat);
    result = k_vsprintf<T>(pStr, pFormat, ap);
    va_end(ap);

    return result;
}

template
signed int k_vprintf<__int32>(const char *pFormat, va_list ap);
