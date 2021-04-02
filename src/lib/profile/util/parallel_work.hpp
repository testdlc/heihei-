// -*-Mode: C++;-*-

// * BeginRiceCopyright *****************************************************
//
// $HeadURL$
// $Id$
//
// --------------------------------------------------------------------------
// Part of HPCToolkit (hpctoolkit.org)
//
// Information about sources of support for research and development of
// HPCToolkit is at 'hpctoolkit.org' and in 'README.Acknowledgments'.
// --------------------------------------------------------------------------
//
// Copyright ((c)) 2019-2020, Rice University
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

#ifndef HPCTOOLKIT_PROFILE_UTIL_PARALLEL_WORK_H
#define HPCTOOLKIT_PROFILE_UTIL_PARALLEL_WORK_H

#include "vgannotations.hpp"

#include <atomic>
#include <limits>
#include <functional>
#include <thread>
#include <vector>

namespace hpctoolkit::util {

/// Helper structure for the results of a workshare contribution.
struct WorkshareResult final {
  WorkshareResult(bool a, bool b) : contributed(a), completed(b) {};
  ~WorkshareResult() = default;

  WorkshareResult(const WorkshareResult&) = default;
  WorkshareResult(WorkshareResult&&) = default;
  WorkshareResult& operator=(const WorkshareResult&) = default;
  WorkshareResult& operator=(WorkshareResult&&) = default;

  /// Whether the request managed to contribute any work to the workshare.
  bool contributed : 1;
  /// Whether any work remains in the workshare for later calls.
  bool completed : 1;
};

/// Parallel version of std::for_each, that allows for multiple threads to
/// contribute their cycles at will and on a whim.
template<class T>
class ParallelForEach {
public:
  ParallelForEach(std::function<void(T&)> f, std::size_t blockSize = 1)
    : action(std::move(f)), blockSize(blockSize),
      nextitem(std::numeric_limits<std::size_t>::max()),
      doneitemcnt(std::numeric_limits<std::size_t>::max()) {};
  ~ParallelForEach() = default;

  ParallelForEach(const ParallelForEach&) = delete;
  ParallelForEach(ParallelForEach&&) = delete;
  ParallelForEach& operator=(const ParallelForEach&) = delete;
  ParallelForEach& operator=(ParallelForEach&&) = delete;

  /// Fill the workqueue with work to be distributed among contributors.
  // MT: Externally Synchronized, Internally Synchronized with contribute().
  void fill(std::vector<T> items) noexcept {
    workitems = std::move(items);
    doneitemcnt.store(0, std::memory_order_relaxed);
    ANNOTATE_HAPPENS_BEFORE(&nextitem);
    nextitem.store(0, std::memory_order_release);
  }

  /// Reset the workshare, allowing it to be used again. Note that this function
  /// must be externally synchronized w.r.t. any contributing threads.
  // MT: Externally Synchronized
  void reset() noexcept {
    workitems.clear();
    nextitem.store(std::numeric_limits<std::size_t>::max(), std::memory_order_relaxed);
    doneitemcnt.store(std::numeric_limits<std::size_t>::max(), std::memory_order_relaxed);
  }

  /// Contribute to the workshare by processing a block of items, if any are
  /// currently available. Returns the result of this request.
  // MT: Internally Synchronized
  [[nodiscard]] WorkshareResult contribute() noexcept {
    auto val = nextitem.load(std::memory_order_acquire);
    ANNOTATE_HAPPENS_AFTER(&nextitem);
    if(val == std::numeric_limits<std::size_t>::max()) return {false, false};
    std::size_t end;
    do {
      if(val > workitems.size()) return {false, true};
      end = std::min(val+blockSize, workitems.size());
    } while(!nextitem.compare_exchange_weak(val, end,
                                            std::memory_order_acquire,
                                            std::memory_order_relaxed));
    for(std::size_t i = val; i < end; ++i) action(workitems[i]);
    ANNOTATE_HAPPENS_BEFORE(&doneitemcnt);
    doneitemcnt.fetch_add(end-val, std::memory_order_release);
    return {true, end >= workitems.size()};
  }

  struct loop_t {};
  constexpr inline loop_t loop() { return {}; }

  /// Contribute to the workshare until there is no more work to be shared.
  /// Note that this does not wait for contributors to complete, just that there
  /// is no more work to be allocated.
  // MT: Internally Synchronized
  void contribute(loop_t) noexcept {
    WorkshareResult res{false, false};
    do {
      res = contribute();
      if(!res.contributed) std::this_thread::yield();
    } while(!res.completed);
  }

  struct wait_t {};
  constexpr inline wait_t wait() { return {}; }

  /// Contribute to the workshare and wait until all work has completed.
  // MT: Internally Synchronized
  void contribute(wait_t) noexcept {
    contribute(loop());
    while(doneitemcnt.load(std::memory_order_acquire) < workitems.size())
      std::this_thread::yield();
    ANNOTATE_HAPPENS_AFTER(&doneitemcnt);
  }

private:
  const std::function<void(T&)> action;
  const size_t blockSize;
  std::vector<T> workitems;
  std::atomic<size_t> nextitem;
  std::atomic<size_t> doneitemcnt;
};

/// Wrapper around ParallelForEach that allows for work to be added in parallel
/// to calls to `contribute()`.
template<class T>
class RepeatingParallelForEach {
private:
  struct Subshare final {
    Subshare(std::function<void(T&)> f, std::size_t blockSize)
      : pfe(std::move(f), blockSize), next(nullptr) {};
    ParallelForEach<T> pfe;
    std::atomic<Subshare*> next;
  };

public:
  RepeatingParallelForEach(std::function<void(T&)> f, std::size_t blockSize = 1)
    : action(std::move(f)), blockSize(blockSize), isComplete(false),
      workhead(nullptr), worknow(nullptr), worktail(nullptr) {};
  ~RepeatingParallelForEach() {
    for(Subshare* w = workhead.load(std::memory_order_relaxed); w != nullptr; ) {
      auto* n = w->next.load(std::memory_order_relaxed);
      delete w;
      w = n;
    }
  }

  RepeatingParallelForEach(const RepeatingParallelForEach&) = delete;
  RepeatingParallelForEach(RepeatingParallelForEach&&) = delete;
  RepeatingParallelForEach& operator=(const RepeatingParallelForEach&) = delete;
  RepeatingParallelForEach& operator=(RepeatingParallelForEach&&) = delete;

  /// Add additional work to the workqueue, to be distributed among contributors.
  // MT: Externally Synchronized, Internally Synchronized with contribute().
  void fill(std::vector<T> items) noexcept {
    Subshare* w = new Subshare([this](T& v){ return action(v); }, blockSize);
    if(worktail == nullptr) workhead.store(w, std::memory_order_release);
    else worktail->next.store(w, std::memory_order_release);
    worktail = w;
    w->pfe.fill(std::move(items));
  }

  /// Mark this workshare as complete. Do not call `fill` after this.
  // MT: Internally Synchronized
  void complete() noexcept {
    isComplete.store(true, std::memory_order_release);
  }

private:
  // Returns the usual result and whether no more work was currently available.
  template<class F>
  std::pair<WorkshareResult, bool> rawcontribute(const F& contrib) noexcept {
    // Try to contribute to the current Subshare
    WorkshareResult res = {false, false};
    Subshare* w = worknow.load(std::memory_order_acquire);
    Subshare* next = nullptr;
    do {
      if(w == nullptr) next = workhead.load(std::memory_order_acquire);
      else {
        res = contrib(*w);
        if(!res.completed) return {res, false};  // We'll need to come back
        res.completed = false;
        next = w->next.load(std::memory_order_acquire);
      }

      // The current Subshare is complete, so try to move to the next
      if(next == nullptr) break;  // Nowhere new to move to
    } while(next != nullptr && !worknow.compare_exchange_weak(w, next,
        std::memory_order_acq_rel, std::memory_order_acquire));

    // Nothing more is available, see if anything more could come
    res.completed = isComplete.load(std::memory_order_acquire);
    return {res, true};
  }

public:
  /// Contribute to the workshare by processing a block of items, if any are
  /// currently available. Returns the result of this request.
  // MT: Internally Synchronized
  [[nodiscard]] WorkshareResult contribute() noexcept {
    return rawcontribute([](Subshare& s){ return s.pfe.contribute(); }).first;
  }

  struct loop_t {};
  constexpr inline loop_t loop() { return {}; }

  /// Contribute to the workshare until there is no more work to be shared.
  /// Unlike the ParallelForEach version, this will return prior to completion
  /// once no more work is available.
  // MT: Internally Synchronized
  [[nodiscard]] WorkshareResult contribute(loop_t) noexcept {
    std::pair<WorkshareResult, bool> res({false, false}, false);
    do {
      res = rawcontribute([](Subshare& s) -> WorkshareResult {
        s.pfe.contribute(s.pfe.loop());
        return {false, true};
      });
      if(!res.first.contributed) std::this_thread::yield();
    } while(!res.second);
    return res.first;
  }

  struct loop_complete_t {};
  constexpr inline loop_complete_t loop_complete() { return {}; }

  /// Contribute to the workshare until there is no more work to be shared.
  /// This will wait until after a call to `complete()`.
  // MT: Internally Synchronized
  void contribute(loop_complete_t) noexcept {
    WorkshareResult res{false, false};
    do {
      res = rawcontribute([](Subshare& s) -> WorkshareResult{
        s.pfe.contribute(s.pfe.loop());
        return {false, true};
      }).first;
      if(!res.contributed) std::this_thread::yield();
    } while(!res.completed);
  }

  struct wait_t {};
  constexpr inline wait_t wait() { return {}; }

  /// Contribute to the workshare and wait until all work has completed.
  /// This will wait until after a call to `complete()`.
  // MT: Internally Synchronized
  void contribute(wait_t) noexcept {
    contribute(loop_complete());
    // At this point no more work should be added. Wait for all the Subshares.
    for(Subshare* w = workhead.load(std::memory_order_relaxed); w != nullptr;
        w = w->next.load(std::memory_order_relaxed)) {
      w->pfe.contribute(w->pfe.wait());
    }
  }

private:
  const std::function<void(T&)> action;
  const size_t blockSize;
  std::atomic<bool> isComplete;
  std::atomic<Subshare*> workhead;
  std::atomic<Subshare*> worknow;
  Subshare* worktail;
};

}

#endif  // HPCTOOLKIT_PROFILE_UTIL_ONCE_H
