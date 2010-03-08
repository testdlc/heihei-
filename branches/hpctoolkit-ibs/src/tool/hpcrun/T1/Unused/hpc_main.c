// -*-Mode: C++;-*- // technically C99

// * BeginRiceCopyright *****************************************************
//
// $HeadURL$
// $Id$
//
// -----------------------------------
// Part of HPCToolkit (hpctoolkit.org)
// -----------------------------------
// 
// Copyright ((c)) 2002-2010, Rice University 
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

#include <linux/unistd.h>
#include <stdlib.h>
#include <string.h>

#define M1 "hpc_main calling setup"
#define M2 "set up itimer"
#define M3 "write out data"

extern int MAIN_(int arc, char **argv);
extern void csprof_init_internal(void);
extern void csprof_fini_internal(void);

#ifdef NO
void csprof_fini_internal(void){
  write(2,M3 "\n",strlen(M3)+1);
}

void csprof_init_internal(void){
  write(2,M2 "\n",strlen(M2)+1);
}
#endif

extern char *static_epoch_xname;

int HPC_MAIN_(int argc, char **argv){
  int ret;

  write(2,M1 "\n",strlen(M1)+1);
  atexit(csprof_fini_internal);
  static_epoch_xname = strdup(argv[0]);
  csprof_init_internal();
  asm(".globl monitor_unwind_fence1");
  asm("monitor_unwind_fence1:");
  ret = MAIN_(argc,argv);
  asm(".globl monitor_unwind_fence2");
  asm("monitor_unwind_fence2:");
}
