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


/*
 * All these functions are for the interpretation of kernel printf's.
 *
 * E.g., printf("int x = %d, y = %d\n", i, j);
 *
 * The alignment and sizes of a data type in the kernel may not be the
 * same as the host as on the device depending on how the CUDA code and
 * host code are compiled (32-bit or 64-bit). Therefore, we need to take
 * that into account when stepping through an va_list.
 */

template<class  T>
signed int k_vprintf(const char *pFormat, va_list ap);


