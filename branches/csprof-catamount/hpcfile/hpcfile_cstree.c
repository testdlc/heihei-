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

//***************************************************************************
//
// File:
//    hpcfile_cstree.c
//
// Purpose:
//    Low-level types and functions for reading/writing a call stack
//    tree from/to a binary file.
//
//    These routines *must not* allocate dynamic memory; if such memory
//    is needed, callbacks to the user's allocator should be used.
//
// Description:
//    [The set of functions, macros, etc. defined in the file]
//
//***************************************************************************

//************************* System Include Files ****************************

#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>

#include <sys/stat.h>
#include <assert.h>

//*************************** User Include Files ****************************

#include "hpcfile_general.h"
#include "hpcfile_cstreelib.h"
#include "hpcfile_cstree.h"

//#include <lib/support/diagnostics.h>

//*************************** Forward Declarations **************************

#define CSTREE_ID_ROOT 1
#define DBG_READ_METRICS 0
#define WRITE_ALL_NODES 0

static int
hpcfile_cstree_write_node(FILE* fs, void* tree, void* node, 
			  hpcfile_cstree_node_t *tmp_node,
			  hpcfile_uint_t id_parent,
			  hpcfile_uint_t *id,
			  hpcfile_cstree_cb__get_data_fn_t get_node_data_fn,
			  hpcfile_cstree_cb__get_first_child_fn_t get_first_child_fn,
			  hpcfile_cstree_cb__get_sibling_fn_t get_sibling_fn,
			  int children_only);

static int 
hpcfile_cstree_read_hdr(FILE* fs, hpcfile_cstree_hdr_t* hdr);

// HPC_CSTREE format details: 
//
// As the nodes are written, each will be given a persistent id and
// each node will know its parent's persistent id.  The id numbers are
// between CSTREE_ID_ROOT and (CSTREE_ID_ROOT + n - 1) where n is the
// number of nodes in the tree.  Furthermore, parents are always
// written before any of their children.  This scheme allows
// hpcfile_cstree_read to easily and efficiently construct the tree
// since the persistent ids correspond to simple array indices and
// parent nodes will be read before children.
//
// The id's will be will be assigned so that given a node x and its
// children, the id of x is less than the id of any child and less
// than the id of any child's descendent.



static int
hpcfile_cstree_count_nodes(void* tree, void* node, 
			  hpcfile_cstree_cb__get_first_child_fn_t get_first_child_fn,
			  hpcfile_cstree_cb__get_sibling_fn_t get_sibling_fn,
			  int levels_to_skip)
{
  int skipped_subtree_count = 0;

  if (node) { 
    if (levels_to_skip-- > 0) {
      skipped_subtree_count++; // count self

      // ---------------------------------------------------------
      // count skipped children 
      // (handles either a circular or non-circular structure)
      // ---------------------------------------------------------
      int kids = 0;
      void *first = get_first_child_fn(tree, node);
      void *c = first; 
      while (c) {
	kids += hpcfile_cstree_count_nodes(tree, c, get_first_child_fn,
					   get_sibling_fn, levels_to_skip);
	c = get_sibling_fn(tree, c);
	if (c == first) { break; }
      }
      skipped_subtree_count += kids; 
    }
  }
  return skipped_subtree_count; 
}


static int
node_child_count(void* tree, void* node, 
		 hpcfile_cstree_cb__get_first_child_fn_t get_first_child_fn,
		 hpcfile_cstree_cb__get_sibling_fn_t get_sibling_fn)
{
  int children = 0;
  if (node) { 
    // ---------------------------------------------------------
    // count children 
    // (handles either a circular or non-circular structure)
    // ---------------------------------------------------------
    void *first = get_first_child_fn(tree, node);
    void *c = first; 
    while (c) {
      children++;
      c = get_sibling_fn(tree, c);
      if (c == first) { break; }
    }
  }
  return children; 
}

//***************************************************************************
// hpcfile_cstree_write()
//***************************************************************************

// See header file for documentation of public interface.
// Cf. 'HPC_CSTREE format details' above.
int
hpcfile_cstree_write(FILE* fs, void* tree, void* root, void* tree_ctxt,
		     hpcfile_uint_t num_metrics,
		     hpcfile_uint_t num_nodes,
                     hpcfile_uint_t epoch,
		     hpcfile_cstree_cb__write_context_fn_t write_context_fn,
		     hpcfile_cstree_cb__get_data_fn_t get_data_fn,
		     hpcfile_cstree_cb__get_first_child_fn_t get_first_child_fn,
		     hpcfile_cstree_cb__get_sibling_fn_t get_sibling_fn)
{
  int ret;
  int levels_to_skip = 0;

  if (tree_ctxt != 0) {
    levels_to_skip = 2;
#if 0
  } else {
    // this case needs to be coordinated with the context writer. it should skip the same.
    int children = node_child_count(tree, root, get_first_child_fn, get_sibling_fn);
    if (children == 1) levels_to_skip = 1;
#endif
  }
    

  if (!fs) { return HPCFILE_ERR; }

  // -------------------------------------------------------
  // When tree_ctxt is non-null, we will peel off the top
  // node in the tree to eliminate the call frame for
  // monitor_pthread_start_routine
  // -------------------------------------------------------
  
  // -------------------------------------------------------
  // Write header
  // -------------------------------------------------------
  hpcfile_cstree_hdr_t fhdr;

  hpcfile_cstree_hdr__init(&fhdr);
  fhdr.epoch = epoch;
  fhdr.num_nodes = num_nodes;
  if (num_nodes > 0 && levels_to_skip > 0) { // FIXME: better way...
    int skipped = hpcfile_cstree_count_nodes(tree, root, get_first_child_fn,
					     get_sibling_fn, levels_to_skip);
    fhdr.num_nodes -= skipped;
  }
  ret = hpcfile_cstree_hdr__fwrite(&fhdr, fs); 
  if (ret != HPCFILE_OK) {
    return HPCFILE_ERR; 
  }

  // -------------------------------------------------------
  // Write context
  // -------------------------------------------------------
  hpcfile_uint_t id_root = CSTREE_ID_ROOT;
  unsigned int num_ctxt_nodes = 0;
  write_context_fn(fs, tree_ctxt, id_root, &num_ctxt_nodes);

  // -------------------------------------------------------
  // Write each node, beginning with root
  // -------------------------------------------------------
  hpcfile_uint_t id_ctxt = CSTREE_ID_ROOT + num_ctxt_nodes - 1;
  hpcfile_uint_t id = id_ctxt + 1;
  hpcfile_cstree_node_t tmp_node;

  hpcfile_cstree_node__init(&tmp_node);
  tmp_node.data.num_metrics = num_metrics;
  tmp_node.data.metrics = malloc(num_metrics * sizeof(hpcfile_uint_t));

  ret = hpcfile_cstree_write_node(fs, tree, root, &tmp_node, id_ctxt, &id,
				  get_data_fn,
				  get_first_child_fn,
				  get_sibling_fn, levels_to_skip);
  free(tmp_node.data.metrics);
  
  return ret;
}

static int
hpcfile_cstree_write_node(FILE* fs, void* tree, void* node, 
			  hpcfile_cstree_node_t *tmp_node,
			  hpcfile_uint_t id_parent,
			  hpcfile_uint_t *id,
			  hpcfile_cstree_cb__get_data_fn_t get_data_fn,
			  hpcfile_cstree_cb__get_first_child_fn_t get_first_child_fn,
			  hpcfile_cstree_cb__get_sibling_fn_t get_sibling_fn,
			  int levels_to_skip)
{
  hpcfile_uint_t myid;
  void* first, *c;
  int ret;

  if (!node) { return HPCFILE_OK; }

  // ---------------------------------------------------------
  // Write this node
  // ---------------------------------------------------------
  if (levels_to_skip > 0) {
    myid = id_parent;
    levels_to_skip--;
  }
  else {
    tmp_node->id = myid = *id;
    tmp_node->id_parent = id_parent;
    get_data_fn(tree, node, &(tmp_node->data));

    ret = hpcfile_cstree_node__fwrite(tmp_node, fs);
    if (ret != HPCFILE_OK) { 
      return HPCFILE_ERR; 
    }
    
    // Prepare next id -- assigned in pre-order
    (*id)++;
  }
  
  // ---------------------------------------------------------
  // Write children (handles either a circular or non-circular structure)
  // ---------------------------------------------------------
  first = c = get_first_child_fn(tree, node);
  while (c) {
    ret = hpcfile_cstree_write_node(fs, tree, c, tmp_node, myid, id,
				    get_data_fn, get_first_child_fn,
				    get_sibling_fn, levels_to_skip);
    if (ret != HPCFILE_OK) {
      return HPCFILE_ERR;
    }

    c = get_sibling_fn(tree, c);
    if (c == first) { break; }
  }
  
  return HPCFILE_OK;
}

//***************************************************************************
// hpcfile_cstree_read()
//***************************************************************************

// See header file for documentation of public interface.
// Cf. 'HPC_CSTREE format details' above.
int
hpcfile_cstree_read(FILE* fs, void* tree, 
		    int num_metrics,
		    hpcfile_cstree_cb__create_node_fn_t create_node_fn,
		    hpcfile_cstree_cb__link_parent_fn_t link_parent_fn,
		    hpcfile_cb__alloc_fn_t alloc_fn,
		    hpcfile_cb__free_fn_t free_fn)
{
  hpcfile_cstree_hdr_t fhdr;
  hpcfile_cstree_node_t tmp_node;
  void* node, *parent;
  int i, ret = HPCFILE_ERR;
  
  // A vector storing created tree nodes.  The vector indices correspond to
  // the nodes' persistent ids.
  void** node_vec = NULL;

  if (!fs) { return HPCFILE_ERR; }
  
  // Read and sanity check header
  if (hpcfile_cstree_read_hdr(fs, &fhdr) != HPCFILE_OK) { 
    return HPCFILE_ERR; 
  }

  // node id upper bound
  unsigned int id_ub = fhdr.num_nodes + CSTREE_ID_ROOT;

  // Allocate space for 'node_vec'
  if (fhdr.num_nodes != 0) {
    node_vec = alloc_fn(sizeof(void*) * id_ub);
    for (int i = 0; i <= CSTREE_ID_ROOT; ++i) {
      node_vec[i] = NULL;
    }
  }
  
  // Read each node, creating it and linking it to its parent 
  tmp_node.data.num_metrics = num_metrics;
  tmp_node.data.metrics = malloc(num_metrics * sizeof(hpcfile_uint_t));
  
  for (i = CSTREE_ID_ROOT; i < id_ub; ++i) {
    if (hpcfile_cstree_node__fread(&tmp_node, fs) != HPCFILE_OK) { 
      goto cstree_read_cleanup; // HPCFILE_ERR
    }
    if (tmp_node.id_parent >= id_ub) { 
      goto cstree_read_cleanup; // HPCFILE_ERR
    } 
    
    parent = node_vec[tmp_node.id_parent];

    // parent should already exist unless id == CSTREE_ID_ROOT
    if (!parent && tmp_node.id != CSTREE_ID_ROOT) {
      goto cstree_read_cleanup; // HPCFILE_ERR
    }

    // Create node and link to parent
    node = create_node_fn(tree, &tmp_node.data);
    node_vec[tmp_node.id] = node;

    if (parent) {
      link_parent_fn(tree, node, parent);
    }
  }

  free(tmp_node.data.metrics);


  // Success! Note: We assume that it is possible for other data to
  // exist beyond this point in the stream; don't check for EOF.
  ret = HPCFILE_OK;
  
  // Close file and cleanup
 cstree_read_cleanup:
  if (node_vec) { free_fn(node_vec); }
  
  return ret;
}

//***************************************************************************
// hpcfile_cstree_fprint()
//***************************************************************************

// See header file for documentation of public interface.
int
hpcfile_cstree_fprint(FILE* infs, int num_metrics, FILE* outfs)
{
  // FIXME: could share some code with the reader
  
  hpcfile_cstree_hdr_t fhdr;
  hpcfile_cstree_node_t tmp_node;
  int i;
  
  // Open file for reading; read and sanity check header
  if (hpcfile_cstree_read_hdr(infs, &fhdr) != HPCFILE_OK) {
    fputs("** Error reading header **\n", outfs);
    return HPCFILE_ERR;
  }
  
  // Print header
  hpcfile_cstree_hdr__fprint(&fhdr, outfs);
  fputs("\n", outfs);

  // Read and print each node
  unsigned int id_ub = fhdr.num_nodes + CSTREE_ID_ROOT;
  for (i = CSTREE_ID_ROOT; i < id_ub; ++i) {
    tmp_node.data.num_metrics = num_metrics;

    if (hpcfile_cstree_node__fread(&tmp_node, infs) != HPCFILE_OK) {
      fprintf(outfs, "** Error reading node number %d **\n", i);
      return HPCFILE_ERR;
    }
    
    hpcfile_cstree_node__fprint(&tmp_node, outfs, "  ");
  }
  
  // Success! Note: We assume that it is possible for other data to
  // exist beyond this point in the stream; don't check for EOF.
  return HPCFILE_OK;
}

//***************************************************************************

// hpcfile_cstree_read_hdr: Read the cstree header from the file
// stream 'fs' into 'hdr' and sanity check header info.  Returns
// HPCFILE_OK upon success; HPCFILE_ERR on error.
int
hpcfile_cstree_read_hdr(FILE* fs, hpcfile_cstree_hdr_t* hdr)
{
  // Read header
  if (hpcfile_cstree_hdr__fread(hdr, fs) != HPCFILE_OK) { 
    return HPCFILE_ERR; 
  }
  
  // Sanity check file id
  if (strncmp(hdr->fid.magic_str, HPCFILE_CSTREE_MAGIC_STR, 
	      HPCFILE_CSTREE_MAGIC_STR_LEN) != 0) { 
    return HPCFILE_ERR; 
  }
  if (strncmp(hdr->fid.version, HPCFILE_CSTREE_VERSION, 
	      HPCFILE_CSTREE_VERSION_LEN) != 0) { 
    return HPCFILE_ERR; 
  }
  if (hdr->fid.endian != HPCFILE_CSTREE_ENDIAN) { return HPCFILE_ERR; }
  
  // Sanity check header
  if (hdr->vma_sz != 8) { return HPCFILE_ERR; }
  if (hdr->uint_sz != 8) { return HPCFILE_ERR; }
  
  return HPCFILE_OK;
}

//***************************************************************************

int 
hpcfile_cstree_id__init(hpcfile_cstree_id_t* x)
{
  memset(x, 0, sizeof(*x));

  strncpy(x->magic_str, HPCFILE_CSTREE_MAGIC_STR, HPCFILE_CSTREE_MAGIC_STR_LEN);
  strncpy(x->version, HPCFILE_CSTREE_VERSION, HPCFILE_CSTREE_VERSION_LEN);
  x->endian = HPCFILE_CSTREE_ENDIAN;

  return HPCFILE_OK;
}

int 
hpcfile_cstree_id__fini(hpcfile_cstree_id_t* x)
{
  return HPCFILE_OK;
}

int 
hpcfile_cstree_id__fread(hpcfile_cstree_id_t* x, FILE* fs)
{
  size_t sz;
  int c;

  sz = fread((char*)x->magic_str, 1, HPCFILE_CSTREE_MAGIC_STR_LEN, fs);
  if (sz != HPCFILE_CSTREE_MAGIC_STR_LEN) { return HPCFILE_ERR; }
  
  sz = fread((char*)x->version, 1, HPCFILE_CSTREE_VERSION_LEN, fs);
  if (sz != HPCFILE_CSTREE_VERSION_LEN) { return HPCFILE_ERR; }

  c = fgetc(fs);
  if (c == EOF) { return HPCFILE_ERR; }
  x->endian = (char)c;

  return HPCFILE_OK;
}

int 
hpcfile_cstree_id__fwrite(hpcfile_cstree_id_t* x, FILE* fs)
{
  size_t sz;

  sz = fwrite((char*)x->magic_str, 1, HPCFILE_CSTREE_MAGIC_STR_LEN, fs);
  if (sz != HPCFILE_CSTREE_MAGIC_STR_LEN) { return HPCFILE_ERR; }

  sz = fwrite((char*)x->version, 1, HPCFILE_CSTREE_VERSION_LEN, fs);
  if (sz != HPCFILE_CSTREE_VERSION_LEN) { return HPCFILE_ERR; }
  
  if (fputc(x->endian, fs) == EOF) { return HPCFILE_ERR; }

  return HPCFILE_OK;
}

int 
hpcfile_cstree_id__fprint(hpcfile_cstree_id_t* x, FILE* fs, const char* pre)
{
  fprintf(fs, "%s{fileid: (magic: %s) (ver: %s) (end: %c)}\n", 
	  pre, x->magic_str, x->version, x->endian);

  return HPCFILE_OK;
}

//***************************************************************************

int 
hpcfile_cstree_hdr__init(hpcfile_cstree_hdr_t* x)
{
  memset(x, 0, sizeof(*x));
  hpcfile_cstree_id__init(&x->fid);
  
  x->vma_sz  = sizeof(hpcfile_vma_t);
  x->uint_sz = sizeof(hpcfile_uint_t);

  return HPCFILE_OK;
}

int 
hpcfile_cstree_hdr__fini(hpcfile_cstree_hdr_t* x)
{
  hpcfile_cstree_id__fini(&x->fid);
  return HPCFILE_OK;
}

int 
hpcfile_cstree_hdr__fread(hpcfile_cstree_hdr_t* x, FILE* fs)
{
  size_t sz;
    
  if (hpcfile_cstree_id__fread(&x->fid, fs) != HPCFILE_OK) { return HPCFILE_ERR; }
  
  sz = hpc_fread_le4(&x->vma_sz, fs);
  if (sz != sizeof(x->vma_sz)) { return HPCFILE_ERR; }
  
  sz = hpc_fread_le4(&x->uint_sz, fs);
  if (sz != sizeof(x->uint_sz)) { return HPCFILE_ERR; }
  
  sz = hpc_fread_le8(&x->num_nodes, fs);
  if (sz != sizeof(x->num_nodes)) { return HPCFILE_ERR; }

  sz = hpc_fread_le4(&x->epoch, fs);
  if (sz != sizeof(x->epoch)) { return HPCFILE_ERR; }

  return HPCFILE_OK;
}

int 
hpcfile_cstree_hdr__fwrite(hpcfile_cstree_hdr_t* x, FILE* fs)
{
  size_t sz;
  
  if (hpcfile_cstree_id__fwrite(&x->fid, fs) != HPCFILE_OK) { return HPCFILE_ERR; }

  sz = hpc_fwrite_le4(&x->vma_sz, fs);
  if (sz != sizeof(x->vma_sz)) { return HPCFILE_ERR; }
 
  sz = hpc_fwrite_le4(&x->uint_sz, fs);
  if (sz != sizeof(x->uint_sz)) { return HPCFILE_ERR; }

  sz = hpc_fwrite_le8(&x->num_nodes, fs);
  if (sz != sizeof(x->num_nodes)) { return HPCFILE_ERR; }

  sz = hpc_fwrite_le4(&x->epoch, fs);
  if (sz != sizeof(x->epoch)) { return HPCFILE_ERR; }

  return HPCFILE_OK;
}

int 
hpcfile_cstree_hdr__fprint(hpcfile_cstree_hdr_t* x, FILE* fs)
{
  const char* pre = "  ";
  fputs("{cstree_hdr:\n", fs);

  hpcfile_cstree_id__fprint(&x->fid, fs, pre);

  fprintf(fs, "%s(vma_sz: %u) (uint_sz: %u) (num_nodes: %"PRIu64")}\n", 
	  pre, x->vma_sz, x->uint_sz, x->num_nodes);
  
  return HPCFILE_OK;
}

//***************************************************************************

int 
hpcfile_cstree_nodedata__init(hpcfile_cstree_nodedata_t* x)
{
  memset(x, 0, sizeof(*x));
  return HPCFILE_OK;
}

int 
hpcfile_cstree_nodedata__fini(hpcfile_cstree_nodedata_t* x)
{
  return HPCFILE_OK;
}

int 
hpcfile_cstree_nodedata__fread(hpcfile_cstree_nodedata_t* x, FILE* fs)
{
  // ASSUMES: space for metrics has been allocated
  
  size_t sz;
  int i;

  sz = hpc_fread_le8(&x->ip, fs);
  if (sz != sizeof(x->ip)) { return HPCFILE_ERR; }

  sz = hpc_fread_le8(&x->sp, fs);
  if (sz != sizeof(x->sp)) { return HPCFILE_ERR; }

  //DIAG_MsgIf(DBG_READ_METRICS, "reading node ip=%"PRIx64, x->ip);
  for (i = 0; i < x->num_metrics; ++i) {
    sz = hpc_fread_le8(&x->metrics[i], fs);
    //DIAG_MsgIf(DBG_READ_METRICS, "metrics[%d]=%"PRIu64, i, x->metrics[i]);
    if (sz != sizeof(x->metrics[i])) {
      return HPCFILE_ERR;
    }
  }

  return HPCFILE_OK;
}


int 
hpcfile_cstree_nodedata__fwrite(hpcfile_cstree_nodedata_t* x, FILE* fs) 
{
  size_t sz;
  int i;

  sz = hpc_fwrite_le8(&x->ip, fs);
  if (sz != sizeof(x->ip)) { return HPCFILE_ERR; }

  sz = hpc_fwrite_le8(&x->sp, fs);
  if (sz != sizeof(x->sp)) { return HPCFILE_ERR; }

  for (i = 0; i < x->num_metrics; ++i) {
    sz = hpc_fwrite_le8(&x->metrics[i], fs);
    if (sz != sizeof(x->metrics[i])) {
      return HPCFILE_ERR;
    }
  }

  return HPCFILE_OK;
}

int 
hpcfile_cstree_nodedata__fprint(hpcfile_cstree_nodedata_t* x, FILE* fs, 
				const char* pre)
{
  int i;
  fprintf(fs, "%s{nodedata: (ip: %"PRIx64") ", pre, x->ip);
  for (i = 0; i < x->num_metrics; ++i) {
    fprintf(fs, " %"PRIu64" ", x->metrics[i]);
  }
  fprintf(fs, "}\n");
  
  return HPCFILE_OK;
}

//***************************************************************************

int 
hpcfile_cstree_node__init(hpcfile_cstree_node_t* x)
{
  memset(x, 0, sizeof(*x));
  hpcfile_cstree_nodedata__init(&x->data);
  return HPCFILE_OK;
}

int 
hpcfile_cstree_node__fini(hpcfile_cstree_node_t* x)
{
  hpcfile_cstree_nodedata__fini(&x->data);  
  return HPCFILE_OK;
}

int 
hpcfile_cstree_node__fread(hpcfile_cstree_node_t* x, FILE* fs)
{
  size_t sz;

  sz = hpc_fread_le8(&x->id, fs);
  if (sz != sizeof(x->id)) { return HPCFILE_ERR; }
  
  sz = hpc_fread_le8(&x->id_parent, fs);
  if (sz != sizeof(x->id_parent)) { return HPCFILE_ERR; }
  
  if (hpcfile_cstree_nodedata__fread(&x->data, fs) != HPCFILE_OK) {
    return HPCFILE_ERR; 
  }
  
  return HPCFILE_OK;
}

int 
hpcfile_cstree_node__fwrite(hpcfile_cstree_node_t* x, FILE* fs)
{
  size_t sz;

  sz = hpc_fwrite_le8(&x->id, fs);
  if (sz != sizeof(x->id)) { return HPCFILE_ERR; }
  
  sz = hpc_fwrite_le8(&x->id_parent, fs);
  if (sz != sizeof(x->id_parent)) { return HPCFILE_ERR; }
  
  if (hpcfile_cstree_nodedata__fwrite(&x->data, fs) != HPCFILE_OK) {
    return HPCFILE_ERR; 
  }
  
  return HPCFILE_OK;
}

int 
hpcfile_cstree_node__fprint(hpcfile_cstree_node_t* x, FILE* fs, const char* pre)
{
  fprintf(fs, "{node:\n");

  fprintf(fs, "%s(id: %"PRIu64") (id_parent: %"PRIu64")}\n", 
	  pre, x->id, x->id_parent);

  hpcfile_cstree_nodedata__fprint(&x->data, fs, pre);
  
  return HPCFILE_OK;
}

//***************************************************************************

