// -*-Mode: C;-*-
// $Id$

/*
  Copyright ((c)) 2002, Rice University 
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  * Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  * Neither the name of Rice University (RICE) nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  This software is provided by RICE and contributors "as is" and any
  express or implied warranties, including, but not limited to, the
  implied warranties of merchantability and fitness for a particular
  purpose are disclaimed. In no event shall RICE or contributors be
  liable for any direct, indirect, incidental, special, exemplary, or
  consequential damages (including, but not limited to, procurement of
  substitute goods or services; loss of use, data, or profits; or
  business interruption) however caused and on any theory of liability,
  whether in contract, strict liability, or tort (including negligence
  or otherwise) arising in any way out of the use of this software, even
  if advised of the possibility of such damage.
*/
#ifndef CSPROF_EPOCH_H
#define CSPROF_EPOCH_H

#include <stdio.h>

/* an "epoch" in csprof-speak is a period of time during which our
   dynamically loaded libraries are stable.  we start a new epoch when
   a library is dlopen()'d; we don't have to worrying about what happens
   after a dlclose() because the dlclose()'d library will not be
   referenced again and it doesn't hurt us to keep around some information
   related to it. */

typedef struct csprof_epoch csprof_epoch_t;
typedef struct csprof_epoch_module csprof_epoch_module_t;

/* an individual load module--shared library or program binary */
struct csprof_epoch_module
{
    struct csprof_epoch_module *next; /* just what it sounds like */
    char *module_name;
    void *vaddr;                /* the preferred virtual address */
    void *mapaddr;              /* the actual mapped address */
    size_t size;		/* just what it sounds like */
};

struct csprof_epoch
{
    struct csprof_epoch *next;  /* the next epoch */
    unsigned int id;            /* an identifier for disk writeouts */
    unsigned int num_modules;   /* how many modules are loaded? */
    struct csprof_epoch_module *loaded_modules;
};


csprof_epoch_t *csprof_epoch_new();
csprof_epoch_t *csprof_get_epoch();

/* avoid weird dynamic loading conflicts */
void csprof_epoch_lock();
void csprof_epoch_unlock();
int csprof_epoch_is_locked();

void csprof_write_all_epochs(FILE *);

#endif /* CSPROF_EPOCH_H */
