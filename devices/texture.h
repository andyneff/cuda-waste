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
 * Class TEXTURE is used to represent a CUDA texture registered with the runtime.
 */
class TEXTURE
{
public:
	TEXTURE()
	{
	};
	
	struct textureReference *hostVar; // used to find the associated TEXREF/TEXARR structures.
	void **deviceAddress;
	char *textureName;
	int dim;
	int norm;
	int ext;
};
