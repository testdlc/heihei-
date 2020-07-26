
//************************* System Include Files ****************************

#include <fstream>
#include <iostream>

#include <climits>
#include <cstdio>
#include <cstring>
#include <string>

#include <algorithm>
#include <stack>
#include <typeinfo>
#include <unordered_map>

#include <sys/stat.h>

//*************************** User Include Files ****************************

#include <include/gcc-attr.h>
#include <include/gpu-metric-names.h>
#include <include/uint.h>

#include "GPUOptimizer.hpp"
#include "Inspection.hpp"

using std::string;

#include <lib/prof/CCT-Tree.hpp>
#include <lib/prof/Metric-ADesc.hpp>
#include <lib/prof/Metric-Mgr.hpp>
#include <lib/prof/Struct-Tree.hpp>

#include <lib/profxml/PGMReader.hpp>
#include <lib/profxml/XercesUtil.hpp>

#include <lib/prof-lean/hpcrun-metric.h>

#include <lib/binutils/LM.hpp>
#include <lib/binutils/VMAInterval.hpp>

#include <lib/xml/xml.hpp>

#include <lib/support/IOUtil.hpp>
#include <lib/support/Logic.hpp>
#include <lib/support/StrUtil.hpp>
#include <lib/support/diagnostics.h>

#include <iostream>
#include <vector>

namespace Analysis {

std::stack<Prof::Struct::Alien *>
InspectionFormatter::getInlineStack(Prof::Struct::ACodeNode *stmt) {
  std::stack<Prof::Struct::Alien *> st;
  Prof::Struct::Alien *alien = stmt->ancestorAlien();

  while (alien) {
    st.push(alien);
    auto *stmt = alien->parent();
    if (stmt) {
      alien = stmt->ancestorAlien();
    } else {
      break;
    }
  };

  return st;
}

std::string SimpleInspectionFormatter::formatInlineStack(
    std::stack<Prof::Struct::Alien *> &inline_stack) {
  std::stringstream ss;

  ss << "Inline stack: " << std::endl;
  while (inline_stack.empty() == false) {
    auto *inline_struct = inline_stack.top();
    inline_stack.pop();
    // Current inline stack line mapping information is not accurate
    //ss << "Line " << inline_struct->begLine() <<
    ss << inline_struct->fileName() << std::endl;
  }

  return ss.str();
}

std::string SimpleInspectionFormatter::format(const Inspection &inspection) {
  std::stringstream ss;

  std::string sep = "------------------------------------------"
    "--------------------------------------------------";

  // Overview
  ss << "Apply " << inspection.optimization << " optimization,";

  ss << " ratio " << inspection.ratios.back() * 100 << "%,";

  ss << " estimate speedup " << inspection.speedups.back() << "x";

  ss << std::endl << std::endl << inspection.hint << std::endl << std::endl;

  // Specific suggestion
  if (inspection.active_warp_count.first != -1) {
    ss << "Adjust #active_warps: " << inspection.active_warp_count.first;

    if (inspection.active_warp_count.second != -1) {
      ss << " to " << inspection.active_warp_count.second;
    }

    ss << std::endl;
  }

  if (inspection.thread_count.first != -1) {
    ss << "Adjust #threads: " << inspection.thread_count.first;

    if (inspection.thread_count.second != -1) {
      ss << " to " << inspection.thread_count.second;
    }

    ss << std::endl;
  }

  if (inspection.block_count.first != -1) {
    ss << "Adjust #blocks: " << inspection.block_count.first;

    if (inspection.block_count.second != -1) {
      ss << " to " << inspection.block_count.second;
    }

    ss << std::endl;
  }

  if (inspection.reg_count.first != -1) {
    ss << "Adjust #regs: " << inspection.reg_count.first;

    if (inspection.reg_count.second != -1) {
      ss << " to " << inspection.reg_count.second;
    }

    ss << std::endl;
  }

  // Hot regions
  for (auto index = 0; index < inspection.top_regions.size(); ++index) {
    auto &inst_blame = inspection.top_regions[index];
    auto ratio = 0.0;
    auto speedup = 0.0;
    if (inspection.loop) {
      ratio = inspection.ratios[index];
      speedup = inspection.speedups[index];
    } else {
      auto metric = inspection.stall ? inst_blame.stall_blame : inst_blame.lat_blame;
      ratio = metric / inspection.total;
    }
    ss << index + 1 << ". Hot " << inst_blame.blame_name << " code, ratio " <<
      ratio * 100 << "%, ";
    if (speedup != 0.0) {
      ss << "speedup " << speedup << "x";
    }
    ss << std::endl;

    auto *src_struct = inst_blame.src_struct;
    auto *dst_struct = inst_blame.dst_struct;
    auto *src_func = src_struct->ancestorProc();
    auto *dst_func = dst_struct->ancestorProc();
    auto src_vma = inst_blame.src_inst == NULL ? src_struct->vmaSet().begin()->beg() :
        (inst_blame.src_inst)->pc - src_func->vmaSet().begin()->beg();
    auto dst_vma = inst_blame.dst_inst == NULL ? dst_struct->vmaSet().begin()->beg() :
        (inst_blame.dst_inst)->pc - dst_func->vmaSet().begin()->beg();

    // Print inline call stack
    std::stack<Prof::Struct::Alien *> src_inline_stack =
        getInlineStack(src_struct);
    std::stack<Prof::Struct::Alien *> dst_inline_stack =
        getInlineStack(dst_struct);

    auto *src_file = src_struct->ancestorFile();
    ss << "From " << src_func->name() << " at " << src_file->name() << ":" <<
      src_file->begLine() << std::endl;
    if (src_inline_stack.empty() == false) {
      ss << formatInlineStack(src_inline_stack);
    }
    ss << std::hex << "0x" << src_vma << std::dec << " at " <<
      "Line " << src_struct->begLine();
    if (inspection.loop) {
      auto *loop = src_struct->ancestorLoop();
      if (loop) {
        ss << " in Loop at Line " << loop->begLine();
      }
    }
    ss  << std::endl;

    auto *dst_file = dst_struct->ancestorFile();
    ss << "To " << dst_func->name() << " at " << dst_file->name() << ":" <<
      dst_file->begLine() << std::endl;
    if (dst_inline_stack.empty() == false) {
      ss << formatInlineStack(dst_inline_stack);
    }
    ss << std::hex << "0x" << dst_vma << std::dec << " at " <<
      "Line " << dst_struct->begLine();
    if (inspection.loop) {
      auto *loop = dst_struct->ancestorLoop();
      if (loop) {
        ss << " in Loop at Line " << loop->begLine();
      }
    }
    ss  << std::endl;

    if (inspection.callback != NULL) {
      ss << inspection.callback(inst_blame) << std::endl;
    }

    ss << std::endl;
  }

  ss << sep << std::endl;

  return ss.str();
};

} // namespace Analysis
