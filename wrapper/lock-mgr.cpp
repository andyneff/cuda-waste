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
//////////////////////////////////////////////////////////////////////////////////////////////////
//
// Based upon the code in API Hooking Revealed (http://www.codeproject.com/KB/system/hooksys.aspx)
// by Ivo Ivanov (ivopi@hotmail.com).
//                                                                         
//////////////////////////////////////////////////////////////////////////////////////////////////
#include "stdafx.h"
#include "lock-mgr.h"

CriticalSection::CriticalSection()
{
	::InitializeCriticalSection( &m_cs );
}

CriticalSection::~CriticalSection()
{
	::DeleteCriticalSection( &m_cs );
}


void CriticalSection::Enter()
{
	::EnterCriticalSection( &m_cs );
}

void CriticalSection::Leave()
{
	::LeaveCriticalSection( &m_cs );
}
