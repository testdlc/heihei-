// -*-Mode: C++;-*-
// $Id$

// * BeginRiceCopyright *****************************************************
// 
// Copyright ((c)) 2002-2007, Rice University 
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

#ifndef support_FileUtil_hpp
#define support_FileUtil_hpp 

//************************* System Include Files ****************************

#include <string>
#include <vector>

#include <fnmatch.h>

//*************************** User Include Files ****************************

//*************************** Forward Declarations ***************************

//****************************************************************************

namespace FileUtil {


// ... is a NULL terminated list of file names 
// CopyFile appends these files into destFile 
//          returns NULL upon success
//          otherwise returns an error message in a static variable 
//          which is overwritten with each call to CopyFile
extern const char* 
CopyFile(const char* destFile, ...); 


extern int 
mkdir(const char* dir);


// mkdirUnique: 
std::pair<std::string, bool>
mkdirUnique(const char* dirnm);

inline std::pair<std::string, bool>
mkdirUnique(const std::string& dirnm)
{
  return mkdirUnique(dirnm.c_str());
}


// retuns a name that can safely be used for a temporary file 
// in a static variable, which is overwritten with each call to 
// tmpname
extern const char* 
tmpname(); 


// count how often char appears in file
// return that number or -1 upon failure to open file for reading
extern int 
CountChar(const char* file, char c); 


// deletes fname (unlink) 
extern int 
DeleteFile(const char *fname);


extern bool 
isReadable(const char *fileName);


// 'basename': returns the 'fname.ext' component of fname=/path/fname.ext
extern std::string 
basename(const char* fname); 

inline std::string 
basename(const std::string& fname)
{
  return basename(fname.c_str());
}


// 'dirname': returns the '/path' component of fname=/path/fname.ext
extern std::string 
dirname(const char* fname); 

inline std::string 
dirname(const std::string& fname)
{
  return dirname(fname.c_str());
}


static inline bool 
fnmatch(const std::string pattern, const char* string, int flags = 0)
{
  int fnd = ::fnmatch(pattern.c_str(), string, flags);
  return (fnd == 0);
#if 0
  if (fnd == 0) {
    return true;
  }
  else if (fnd != FNM_NOMATCH) {
    // error
  }
#endif
}


bool 
fnmatch(const std::vector<std::string>& patternVec, 
	const char* string, 
	int flags = 0);
  
} // end of FileUtil namespace

#endif // support_FileUtil_hpp
