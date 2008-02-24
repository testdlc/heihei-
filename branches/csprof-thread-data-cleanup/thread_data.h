#ifndef THREAD_DATA_H
#define THREAD_DATA_H

// This is called "thread data", but it applies to process as well
// (there is just 1 thread).

#include <setjmp.h>
#include "bad_unwind.h"
#include "mem.h"
#include "state.h"

typedef struct _td_t {
  int id;
  csprof_state_t  *state;
  csprof_mem_t    *memstore;
  sigjmp_buf_t    bad_unwind;
  int             eventSet;     // for PAPI
  int             handling_sample;
} thread_data_t;

#define TD_GET(field) csprof_get_thread_data()->field

extern thread_data_t *(*csprof_get_thread_data)(void);
extern thread_data_t *csprof_allocate_thread_data(void);
extern void           csprof_init_pthread_key(void);
extern void           csprof_thread_data_init(int id,offset_t sz1,offset_t sz2);
extern void           csprof_set_thread_data(thread_data_t *td);
extern void           csprof_set_thread0_data(void);
extern void           csprof_unthreaded_data(void);
extern void           csprof_threaded_data(void);

// utilities to match previous api
#define csprof_get_state()  TD_GET(state)


#endif // !defined(THREAD_DATA_H)
