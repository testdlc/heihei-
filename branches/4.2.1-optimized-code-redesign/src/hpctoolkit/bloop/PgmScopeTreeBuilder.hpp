// -*-Mode: C++;-*-
// $Id$
// * BeginRiceCopyright *****************************************************
// 
// Copyright ((c)) 2002, Rice University 
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// 
// * Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
// 
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
// 
// * Neither the name of Rice University (RICE) nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
// 
// This software is provided by RICE and contributors "as is" and any
// express or implied warranties, including, but not limited to, the
// implied warranties of merchantability and fitness for a particular
// purpose are disclaimed. In no event shall RICE or contributors be
// liable for any direct, indirect, incidental, special, exemplary, or
// consequential damages (including, but not limited to, procurement of
// substitute goods or services; loss of use, data, or profits; or
// business interruption) however caused and on any theory of liability,
// whether in contract, strict liability, or tort (including negligence
// or otherwise) arising in any way out of the use of this software, even
// if advised of the possibility of such damage. 
// 
// ******************************************************* EndRiceCopyright *

//***************************************************************************
//
// File:
//    $Source$
//
// Purpose:
//    [The purpose of this file]
//
// Description:
//    [The set of functions, macros, etc. defined in the file]
//
//***************************************************************************

#ifndef PgmScopeTreeBuilder_H 
#define PgmScopeTreeBuilder_H

//************************* System Include Files ****************************

//*************************** User Include Files ****************************

#include <include/general.h> 

#include <lib/prof-juicy/PgmScopeTree.hpp>
#include <lib/binutils/LoadModule.hpp>

//*************************** Forward Declarations ***************************

class PCToSrcLineXMap;

//*************************** Forward Declarations ***************************

// Functions to build and dump a scope tree from an Executable

namespace ScopeTreeBuilder {
  
  PgmScopeTree* 
  BuildFromLM(LoadModule* lm, PCToSrcLineXMap* &map,
	      String canonicalPathList = "",
	      bool normalizeScopeTree = true,
	      bool unsafeNormalizations = true,
	      bool irreducibleIntervalIsLoop = false,
	      bool verboseMode = false);
  
  bool 
  Normalize(PgmScopeTree* pgmScopeTree, 
	    bool unsafeNormalizations = true);
  
}

void
WriteScopeTree(std::ostream& os, PgmScopeTree* pgmScopeTree,
	       bool prettyPrint = true);

//****************************************************************************

#endif
