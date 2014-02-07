// -*-Mode: C++;-*- // technically C99
// $Id: lushi-cb.h 1485 2008-06-23 19:45:36Z eraxxon $

//***************************************************************************
//
// File: 
//    $Source$
//
// Purpose:
//    LUSH Interface: Callback Interface for LUSH agents
//
// Description:
//    [The set of functions, macros, etc. defined in the file]
//
// Author:
//    Nathan Tallent, Rice University.
//
//***************************************************************************

#ifndef lush_lush_cb_h
#define lush_lush_cb_h

//************************* System Include Files ****************************

#include <stdlib.h>

//*************************** User Include Files ****************************

#include "lush-support-rt.h"

//*************************** Forward Declarations **************************

// **************************************************************************
// A LUSH agent expects the following callbacks:
// **************************************************************************

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------
// Interface for 'heap memory' allocation
// ---------------------------------------------------------

LUSHI_DECL(void*, LUSHCB_malloc, (size_t size));
LUSHI_DECL(void,  LUSHCB_free, ());

// ---------------------------------------------------------
// Facility for unwinding physical stack
// ---------------------------------------------------------

typedef unw_cursor_t LUSHCB_cursor_t;

// LUSHCB_step: Given a cursor, step the cursor to the next (less
// deeply nested) frame.  Conforms to the semantics of libunwind's
// unw_step.  In particular, returns:
//   > 0 : successfully advanced cursor to next frame
//     0 : previous frame was the end of the unwind
//   < 0 : error condition
LUSHI_DECL(int, LUSHCB_step, (LUSHCB_cursor_t* cursor));


LUSHI_DECL(int, LUSHCB_loadmap_find, (void* addr, 
				      char *module_name,
				      void** start, 
				      void** end));

#ifdef __cplusplus
}
#endif

// **************************************************************************

#endif /* lush_lush_cb_h */