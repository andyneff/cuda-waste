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

class EMULATED_DEVICE;

class SYMBOL
{
public:
    char * name;
    void * pvalue;
    void * cache;//for debugging.
    size_t size;
    char * typestring;
    int type;
    bool array;
    size_t total_size;
    int storage_class;
    EMULATED_DEVICE * emulator;
    ~SYMBOL();
};

