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

#include <math.h>
#include <stdlib.h>
#include <alloca.h>

#define LIMIT_OUTER 100
#define LIMIT 100000

int bar() {
	int x, y;
	x = 1; y = 2;
	return x+y;
}

void foob(double *x){
  *x = (*x) * 3.14 + atan(*x);
}

double moo_leaf(double *x){
  int sz = (drand48() * 10) + 3; // at least 3
  double* mem = (double*)alloca(sz * sizeof(double));
  mem[0] = *x;
  mem[1] = 3.14;
  mem[2] = atan(mem[0]);
  *x = mem[0] * mem[1] + mem[2];
  return *x;
}

void moo_interior(double *x){
  int sz = (drand48() * 10) + 2; // at least 2
  double* mem = (double*)alloca(sz * sizeof(double));
  mem[0] = *x;
  mem[1] = moo_leaf(&mem[0]);
  mem[0] = sin(mem[1]);
  *x = mem[0] + mem[1];
}


int main(int argc, char *argv[]){
  double x,y,z;
  int i,j;

  srand48(5);
  
  x = 2.78;
  z = 3.14;
  foob(&x);
  for (i=0; i < LIMIT_OUTER; i++){
    for (j=0; j < LIMIT; j++){
      y = x * x + sin(y);
      x = log(y) + cos(x);
      moo_interior(&z);
    }
  }
  return 0;
}
