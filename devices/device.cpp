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
#include <string.h>

#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <string>
#include <sstream>
#include <crtdbg.h>
#include "emulated-device.h"
#include <assert.h>
#include <fstream>
#include <iostream>
#include <queue>
#include <map>
#include "tree.h"
#include <process.h>    /* _beginthread, _endthread */
#include "thread.h"
#include "symbol-table.h"
#include "string-table.h"
#include "symbol.h"
#include "constant.h"
#include "types.h"
#include "../wrapper/memdbg.h"
#include "../waste/version.h"
#include "../wrapper/cuda-wrapper.h"
#include <stdio.h>
#include <assert.h>
#define ZLIB_WINAPI 1
#include "zlib.h"
#include <__cudaFatFormat.h>
#include "../wrapper/call-stack-info.h"
#include "entry.h"
#include "texref.h"
#include "texture.h"
#include "array.h"
#include "texarr.h"
#include "module.h"
#include "regvar.h"
#include "errors.h"


#define new new(_CLIENT_BLOCK,__FILE__, __LINE__)


DEVICE::DEVICE()
{
}

DEVICE::~DEVICE()
{
}
