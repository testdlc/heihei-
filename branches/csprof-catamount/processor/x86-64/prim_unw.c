// NOTE: For static linked versions, this MUST BE compiled for
//       the target system, as the include file that defines the
//       structure for the contexts may be different for the
//       target system vs the build system.
// !! ESPECIALLY FOR CATAMOUNT !!
//
#include <setjmp.h>
#include <ucontext.h>

#include "bad_unwind.h"
#include "find.h"
#include "general.h"
#include "intervals.h"
#include "pmsg.h"

#include "prim_unw_cursor.h"

#include "splay.h"

#if defined(__LIBCATAMOUNT__)
#undef __CRAYXT_CATAMOUNT_TARGET
#define __CRAYXT_CATAMOUNT_TARGET
#endif

#define NO 1
#define BUILD_INT 1

void unw_init(void){
  extern void xed_init(void);

  
  PMSG(UNW,"UNW: xed, splay tree init");
  xed_init();
  csprof_interval_tree_init();
}

void unw_init_f_mcontext(void *context,unw_cursor_t *frame){

  PMSG(UNW,"init prim unw (mcontext) called: context = %p, cursor_p = %p\n",context,frame);
  mcontext_t *ctx = (mcontext_t *) context;


#ifdef __CRAYXT_CATAMOUNT_TARGET
  frame->pc = (void *)ctx->sc_rip;
  frame->bp = (void **)ctx->sc_rbp;
  frame->sp = (void **)ctx->sc_rsp;
#else
  frame->pc = (void *)ctx->gregs[REG_RIP];
  frame->bp = (void **)ctx->gregs[REG_RBP];
  frame->sp = (void **)ctx->gregs[REG_RSP];
#endif

  PMSG(UNW,"UNW_INIT:frame pc = %p, frame bp = %p, frame sp = %p",frame->pc,frame->bp,
      frame->sp);

  frame->intvl = csprof_addr_to_interval((unsigned long)frame->pc);
  PMSG(UNW,"UNW_INIT: returned interval = %p",frame->intvl);
}

void unw_init_f_ucontext(void *context,unw_cursor_t *frame){

  PMSG(UNW,"init prim unw called w ucontext: context = %p, cursor_p = %p\n",context,frame);
  ucontext_t *ctx = (ucontext_t *) context;

  unw_init_f_mcontext((void *) &(ctx->uc_mcontext),frame);
}

// This get_reg just extracts the pc, regardless of REGID

int unw_get_reg(unw_cursor_t *cursor,int REGID,void **regv){

  *regv = cursor->pc;
  
  return 0;
}

int unw_step (unw_cursor_t *cursor){
  void **bp, **spr_sp, **spr_bp;
  void *sp,*pc,*spr_pc;
  unwind_interval *uw;

  // current frame
  bp         = cursor->bp;
  sp         = cursor->sp;
  pc         = cursor->pc;
  uw         = cursor->intvl;
  if (! uw ){
    EMSG("!!! No interval found!!!,cursor pc = %p",pc);
    _jb *it = get_bad_unwind();
    siglongjmp(it->jb,9);
  } else {
     PMSG(UNW,"dumping the found interval");
     // idump(uw); // debug for now
  }
  // spr rel step
  // FIXME: next bp needs to check if this is a frame procedure
   
  spr_sp     = ((void **)((unsigned long) sp + uw->ra_pos));
  spr_pc     = *spr_sp;
  spr_bp     = (void **) *(spr_sp -1);
  spr_sp    += 1; 

  cursor->pc = spr_pc;
  cursor->bp = spr_bp;
  cursor->sp = spr_sp;
  cursor->intvl = csprof_addr_to_interval((unsigned long)spr_pc);

#ifdef NO
  PMSG(UNW,"NEXT frame pc = %p, frame bp = %p\n",cursor->pc,cursor->bp);
#endif

  return 1;
}
