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
 * Class ARRAY is used to represent a CUDA "Array".
 */
class ARRAY
{
private:
	struct cudaChannelFormatDesc *desc;
	size_t width;
	size_t height;
	size_t depth;
	size_t pitch;
	int dim;
	unsigned int flags;
	unsigned char * memory;

public:
	ARRAY(struct cudaChannelFormatDesc * _desc, size_t _width, 	size_t _height, unsigned int _flags)
	{
		this->desc = _desc;
		this->width = _width;
		this->height = _height;
		this->depth = 1;
		this->dim = 2;
		this->flags = _flags;
		this->memory = 0;
	};

	ARRAY(struct cudaChannelFormatDesc * _desc, size_t _width, 	size_t _height, size_t _depth, unsigned int _flags)
	{
		this->desc = _desc;
		this->width = _width;
		this->height = _height;
		this->depth = _depth;
		this->dim = 3;
		this->flags = _flags;
		this->memory = 0;
	};

	~ARRAY()
	{
		delete this->memory;
	};

	const struct cudaChannelFormatDesc * Desc()
	{
		return const_cast<const struct cudaChannelFormatDesc *>(this->desc);
	}

	size_t Width()
	{
		return this->width;
	}

	size_t Height()
	{
		return this->height;
	}

	unsigned int Flags()
	{
		return this->flags;
	}

	unsigned char * Memory()
	{
		return this->memory;
	}
	
	void Memory(unsigned char * _memory)
	{
		this->memory = _memory;
	}
	
};
