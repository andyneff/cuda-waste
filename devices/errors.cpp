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

#include "errors.h"

ERRS * ERRS::last_error;

int ERRS::Code()
{
	return this->_code;
}

char * ERRS::String()
{
	return this->_string;
}

char * ERRS::Location()
{
	return this->_location;
}

ERRS::ERRS(int code, char * string, char * location)
{
	this->_code = code;
	this->_string = string;
	this->_location = location;
	ERRS::last_error = this;
}

ERRS * ERRS::LastError()
{
	return ERRS::last_error;
}
