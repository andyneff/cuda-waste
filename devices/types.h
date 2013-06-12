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
#pragma once

class TYPES {
public:
    typedef union Types {
        signed __int64 s64;
        signed __int32 s32;
        signed __int16 s16;;
        signed __int8 s8;
        unsigned __int64 u64;
        unsigned __int32 u32;
        unsigned __int16 u16;
        unsigned __int8 u8;
        unsigned __int64 b64;
        unsigned __int32 b32;
        unsigned __int16 b16;
        unsigned __int8 b8;
        float f16;  // not really supported.
        float f32;
        double f64;
        bool pred;
        void * pvoid;
        char * pchar;
        unsigned char * puchar;
    } Types;
};

