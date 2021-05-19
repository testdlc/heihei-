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

#include "experimentxml4.hpp"

#include "hpctracedb2.hpp"
#include "../util/xml.hpp"

#include <iomanip>
#include <algorithm>
#include <sstream>
#include <limits>
#include <iostream>

using namespace hpctoolkit;
using namespace sinks;
namespace fs = stdshim::filesystem;

// Function name transformation table
namespace fancynames {
  using type = std::pair<std::string, int>;

  static const type program_root     = {"<program root>", 4};
  static const type thread_root      = {"<thread root>", 4};
  static const type omp_idle         = {"<omp idle>", 1};
  static const type omp_overhead     = {"<omp overhead>", 1};
  static const type omp_barrier_wait = {"<omp barrier wait>", 1};
  static const type omp_task_wait    = {"<omp task wait>", 1};
  static const type omp_mutex_wait   = {"<omp mutex wait>", 1};
  static const type no_thread        = {"<no thread>", 1};
  static const type partial_unwind   = {"<partial call paths>", 4};
  static const type no_activity      = {"<no activity>", 1};
  static const type gpu_copy         = {"<gpu copy>", 1};
  static const type gpu_copyin       = {"<gpu copyin>", 1};
  static const type gpu_copyout      = {"<gpu copyout>", 1};
  static const type gpu_alloc        = {"<gpu alloc>", 1};
  static const type gpu_delete       = {"<gpu delete>", 1};
  static const type gpu_sync         = {"<gpu sync>", 1};
  static const type gpu_kernel       = {"<gpu kernel>", 1};

  static const type unknown_proc     = {"<unknown procedure>", 0};
}
static const std::unordered_map<std::string, const fancynames::type&> nametrans = {
  {"monitor_main", fancynames::program_root},
  {"monitor_main_fence1", fancynames::program_root},
  {"monitor_main_fence2", fancynames::program_root},
  {"monitor_main_fence3", fancynames::program_root},
  {"monitor_main_fence4", fancynames::program_root},
  {"monitor_begin_thread", fancynames::thread_root},
  {"monitor_thread_fence1", fancynames::thread_root},
  {"monitor_thread_fence2", fancynames::thread_root},
  {"monitor_thread_fence3", fancynames::thread_root},
  {"monitor_thread_fence4", fancynames::thread_root},
  {"ompt_idle_state", fancynames::omp_idle},
  {"ompt_idle", fancynames::omp_idle},
  {"ompt_overhead_state", fancynames::omp_overhead},
  {"omp_overhead", fancynames::omp_overhead},
  {"ompt_barrier_wait_state", fancynames::omp_barrier_wait},
  {"ompt_barrier_wait", fancynames::omp_barrier_wait},
  {"ompt_task_wait_state", fancynames::omp_task_wait},
  {"ompt_task_wait", fancynames::omp_task_wait},
  {"ompt_mutex_wait_state", fancynames::omp_mutex_wait},
  {"ompt_mutex_wait", fancynames::omp_mutex_wait},
  {"NO_THREAD", fancynames::no_thread},
  {"gpu_op_copy", fancynames::gpu_copy},
  {"gpu_op_copyin", fancynames::gpu_copyin},
  {"gpu_op_copyout", fancynames::gpu_copyout},
  {"gpu_op_alloc", fancynames::gpu_alloc},
  {"gpu_op_delete", fancynames::gpu_delete},
  {"gpu_op_sync", fancynames::gpu_sync},
  {"gpu_op_kernel", fancynames::gpu_kernel},
  {"gpu_op_trace", fancynames::gpu_kernel},
  {"hpcrun_no_activity", fancynames::no_activity},
};

// ud Module bits

ExperimentXML4::udModule::udModule(const Module& m, ExperimentXML4& exml)
  : id(m.userdata[exml.src.identifier()]+1), unknown_file(exml, m),
    used(false) {};

ExperimentXML4::udModule::udModule(ExperimentXML4& exml)
  : id(0), unknown_file(exml), used(true) {
  std::ostringstream ss;
  ss << "<LoadModule i=\"" << id << "\" n=" << util::xmlquoted("unknown module") << "/>\n";
  tag = ss.str();
}

void ExperimentXML4::udModule::incr(const Module& mod, ExperimentXML4&) {
  if(!used.exchange(true, std::memory_order_relaxed)) {
    std::ostringstream ss;
    ss << "<LoadModule i=\"" << id << "\" n=" << util::xmlquoted(mod.path().string()) << "/>\n";
    tag = ss.str();
  }
}

// ud File bits

ExperimentXML4::udFile::udFile(const File& f, ExperimentXML4& exml)
  : id(f.userdata[exml.src.identifier()]), used(false), fl(&f),
    m(nullptr) {};

ExperimentXML4::udFile::udFile(ExperimentXML4& exml, const Module& mm)
  : id(exml.next_id.fetch_sub(1, std::memory_order_relaxed)), used(false),
    fl(nullptr), m(&mm) {};

ExperimentXML4::udFile::udFile(ExperimentXML4& exml)
  : id(exml.next_id.fetch_sub(1, std::memory_order_relaxed)), used(false),
    fl(nullptr), m(nullptr) {};

void ExperimentXML4::udFile::incr(ExperimentXML4& exml) {
  namespace fs = stdshim::filesystem;
  if(!used.exchange(true, std::memory_order_relaxed)) {
    std::ostringstream ss;
    if(fl == nullptr) {
      ss << "<File i=\"" << id << "\" n=\"&lt;unknown file&gt;";
      if(m != nullptr) ss << " [" << util::xmlquoted(m->path().filename().string(), false) << "]";
      ss << "\"/>\n";
    } else {
      ss << "<File i=\"" << id << "\" n=";
      const fs::path& rp = fl->userdata[exml.src.resolvedPath()];
      if(exml.include_sources && !rp.empty()) {
        fs::path p = fs::path("src") / rp.relative_path().lexically_normal();
        ss << util::xmlquoted("./" + p.string());
        p = exml.dir / p;
        if(!exml.dir.empty()) {
          fs::create_directories(p.parent_path());
          fs::copy_file(rp, p, fs::copy_options::overwrite_existing);
        }
      } else ss << util::xmlquoted(fl->path().string());
      ss << "/>\n";
    }
    tag = ss.str();
  }
}

// ud Metric bits

static void combineFormula(std::ostream& os, unsigned int id,
                           const StatisticPartial& p) {
  os << "<MetricFormula t=\"combine\" frm=\"";
  switch(p.combinator()) {
  case Statistic::combination_t::sum: os << "sum"; break;
  case Statistic::combination_t::min: os << "min"; break;
  case Statistic::combination_t::max: os << "max"; break;
  }
  os << "($" << id << ", $" << id << ")\"/>\n";
}

static void finalizeFormula(std::ostream& os, const std::string& mode,
                            unsigned int idbase, const Statistic& s) {
  os << "<MetricFormula t=\"" << mode << "\" frm=\"";
  for(const auto& e: s.finalizeFormula()) {
    if(std::holds_alternative<size_t>(e))
      os << "$" << (idbase + std::get<size_t>(e));
    else if(std::holds_alternative<std::string>(e))
      os << std::get<std::string>(e);
  }
  os << "\"/>\n";
}

ExperimentXML4::udMetric::udMetric(const Metric& m, ExperimentXML4& exml) {
  if(!m.scopes().has(MetricScope::function) && !m.scopes().has(MetricScope::execution))
    util::log::fatal{} << "Metric " << m.name() << " has neither function nor execution!";
  if(m.partials().size() > 64 || m.statistics().size() > 64)
    util::log::fatal{} << "Too many Statistics/Partials!";
  const auto& ids = m.userdata[exml.src.mscopeIdentifiers()];
  maxId = (std::max(ids.execution, ids.function) << 8) + ((1<<8)-1);

  {
    std::ostringstream ss;

    // First pass: get all the Partials out there.
    for(size_t idx = 0; idx < m.partials().size(); idx++) {
      const auto& partial = m.partials()[idx];
      const std::string name = m.name() + ":PARTIAL:" + std::to_string(idx);

      const auto f = [&](MetricScope ms, MetricScope p_ms,
                         unsigned int id, unsigned int p_id,
                         std::string suffix, std::string type) {
        if(m.scopes().has(ms)) {
          ss << "<Metric i=\"" << id << "\" o=\"" << id << "\" "
                            "n=" << util::xmlquoted(m.scopes().has(p_ms) ? name+suffix : name) << " "
                            "md=" << util::xmlquoted(m.description()) << " "
                            "v=\"derived-incr\" "
                            "t=\"" << type << "\" partner=\"" << p_id << "\" "
                            "show=\"4\" show-percent=\"0\">\n";
          combineFormula(ss, id, partial);
          ss << "<Info><NV n=\"units\" v=\"events\"/></Info>\n"
                "</Metric>\n";
        } else {
          ss << "<Metric i=\"" << id << "\" o=\"" << id << "\" "
                        "n=" << util::xmlquoted(name+" INTERNAL") << " "
                        "v=\"derived-incr\" "
                        "t=\"" << type << "\" partner=\"" << p_id << "\" "
                        "show=\"4\" show-percent=\"0\"/>\n";
        }
      };

      const auto exec_id = m.scopes().has(MetricScope::execution)
                           ? (ids.execution << 8) + idx
                           : (ids.function << 8) + 64 + idx;
      const auto func_id = m.scopes().has(MetricScope::function)
                           ? (ids.function << 8) + idx
                           : (ids.execution << 8) + 64 + idx;
      f(MetricScope::execution, MetricScope::function, exec_id, func_id,
        " (I)", "inclusive");
      f(MetricScope::function, MetricScope::execution, func_id, exec_id,
        " (E)", "exclusive");
    }

    // Second pass: handle all the Statistics.
    for(size_t idx = 0; idx < m.statistics().size(); idx++) {
      const auto& stat = m.statistics()[idx];
      const std::string name = m.name() + ":" + stat.suffix();
      const auto f = [&](MetricScope ms, MetricScope p_ms,
                         unsigned int id, unsigned int p_id,
                         unsigned int base, std::string suffix, std::string type) {
        if(m.scopes().has(ms)) {
          ss << "<Metric i=\"" << id << "\" o=\"" << id << "\" "
                            "n=" << util::xmlquoted(m.scopes().has(p_ms) ? name+suffix : name) << " "
                            "md=" << util::xmlquoted(m.description()) << " "
                            "v=\"derived-incr\" "
                            "t=\"" << type << "\" partner=\"" << p_id << "\" "
                            "show=\"" << (stat.visibleByDefault() ? "1" : "0") << "\" "
                            "show-percent=\"" << (stat.showPercent() ? "1" : "0") << "\">\n";
          finalizeFormula(ss, "view", base, stat);
          ss << "<Info><NV n=\"units\" v=\"events\"/></Info>\n"
                "</Metric>\n";
        } else {
          ss << "<Metric i=\"" << id << "\" o=\"" << id << "\" "
                        "n=" << util::xmlquoted(name+" INTERNAL") << " "
                        "v=\"derived-incr\" "
                        "t=\"" << type << "\" partner=\"" << p_id << "\" "
                        "show=\"4\" show-percent=\"0\"/>\n";
        }
      };

      const auto exec_id = m.scopes().has(MetricScope::execution)
                           ? (ids.execution << 8) + 256-m.statistics().size() + idx
                           : (ids.function << 8) + 256-m.statistics().size() + 64 + idx;
      const auto func_id = m.scopes().has(MetricScope::function)
                           ? (ids.function << 8) + 256-m.statistics().size() + idx
                           : (ids.execution << 8) + 256-m.statistics().size() + 64 + idx;
      f(MetricScope::execution, MetricScope::function, exec_id, func_id,
        ids.execution << 8, " (I)", "inclusive");
      f(MetricScope::function, MetricScope::execution, func_id, exec_id,
        ids.function << 8, " (E)", "exclusive");
    }

    metric_tags = ss.str();
  }

  std::ostringstream ss2;
  const auto f = [&](MetricScope ms, MetricScope p_ms,
                     unsigned int id, std::string suffix) {
    if(!m.scopes().has(ms)) return;
    ss2 << "<MetricDB i=\"" << id << "\""
                    " n=" << util::xmlquoted(m.scopes().has(p_ms) ? m.name()+suffix : m.name())
        << "/>\n";
  };
  f(MetricScope::execution, MetricScope::function, ids.execution, " (I)");
  f(MetricScope::function, MetricScope::execution, ids.function, " (E)");
  metricdb_tags = ss2.str();
}

std::string ExperimentXML4::eStatMetricTags(const ExtraStatistic& es, unsigned int& id) {
  std::ostringstream ss;
  const auto f = [&](MetricScope ms, MetricScope p_ms,
                     unsigned int id, unsigned int p_id,
                     std::string suffix, std::string type) {
    if(es.scopes().has(ms)) {
      ss << "<Metric i=\"" << id << "\" o=\"" << id << "\" "
                        "n=" << util::xmlquoted(es.scopes().has(p_ms) ? es.name()+suffix : es.name()) << " "
                        "md=" << util::xmlquoted(es.description()) << " "
                        "v=\"derived-incr\" "
                        "t=\"" << type << "\" partner=\"" << p_id << "\" "
                        "show=\"" << (es.visibleByDefault() ? "1" : "0") << "\" "
                        "show-percent=\"" << (es.showPercent() ? "1" : "0") << "\" ";
      if(!es.format().empty())
        ss << "fmt=" << util::xmlquoted(es.format()) << " ";
      ss << ">\n"
            "<MetricFormula t=\"view\" frm=\"";
      for(const auto& e: es.formula()) {
        if(std::holds_alternative<std::string>(e)) {
          ss << std::get<std::string>(e);
        } else {
          const auto& mp = std::get<ExtraStatistic::MetricPartialRef>(e);
          const auto& ids = mp.metric.userdata[src.mscopeIdentifiers()];
          ss << "$" << ((ids.get(ms) << 8) + mp.partialIdx);
        }
      }
      ss << "\"/>\n"
            "<Info><NV n=\"units\" v=\"events\"/></Info>\n"
            "</Metric>\n";
    } else {
      ss << "<Metric i=\"" << id << "\" o=\"" << id << "\" "
                    "n=" << util::xmlquoted(es.name()+" INTERNAL") << " "
                    "v=\"derived-incr\" "
                    "t=\"" << type << "\" partner=\"" << p_id << "\" "
                    "show=\"4\" show-percent=\"0\"/>\n";
    }
  };

  const auto exec_id = ++id;
  const auto func_id = ++id;
  f(MetricScope::execution, MetricScope::function, exec_id, func_id,
    " (I)", "inclusive");
  f(MetricScope::function, MetricScope::execution, func_id, exec_id,
    " (E)", "exclusive");

  return ss.str();
}

// ud Context bits

ExperimentXML4::udContext::udContext(const Context& c, ExperimentXML4& exml)
  : premetrics(false) {
  const auto& s = c.scope();
  auto& proc = exml.getProc(s);
  switch(s.type()) {
  case Scope::Type::unknown: {
    auto& uproc = (c.direct_parent()->scope().type() == Scope::Type::global)
                   ? exml.proc_partial() : exml.proc_unknown();
    exml.file_unknown.incr(exml);
    std::ostringstream ss;
    ss << "<PF i=\"" << c.userdata[exml.src.identifier()] << "\""
             " n=\"" << uproc.id << "\" s=\"" << uproc.id << "\""
             " f=\"" << exml.file_unknown.id << "\""
             " l=\"0\">\n"
          "<C i=\"" << c.userdata[exml.src.identifier()] << "\""
             " s=\"" << uproc.id << "\" v=\"0\" l=\"0\"";
    open = ss.str();
    close = "</C>\n";
    post = "</PF>\n";
    break;
  }
  case Scope::Type::loop: {
    auto fl = s.line_data();
    std::ostringstream ss;
    ss << "<L i=\"" << c.userdata[exml.src.identifier()] << "\""
             " s=\"" << proc.id << "\" v=\"0\""
             " l=\"" << fl.second << "\""
             " f=\"" << fl.first.userdata[exml.ud].id << "\"";
    open = ss.str();
    close = "</L>\n";
    break;
  }
  case Scope::Type::global:
    open = "<SecCallPathProfileData";
    close = "</SecCallPathProfileData>\n";
    break;
  case Scope::Type::point:
  case Scope::Type::classified_point:
  case Scope::Type::line:
  case Scope::Type::concrete_line:
  case Scope::Type::call:
  case Scope::Type::classified_call: {
    std::pair<const Module*, uint64_t> mo{nullptr, 0};
    if(s.type() != Scope::Type::line) {
      auto mmo = s.point_data();
      mo.first = &mmo.first;
      mo.second = mmo.second;
    }
    std::pair<const File*, uint64_t> fl{nullptr, 0};
    if(s.type() != Scope::Type::point && s.type() != Scope::Type::call) {
      auto ffl = s.line_data();
      fl.first = &ffl.first;
      fl.second = ffl.second;
    }
    const auto pty = c.direct_parent()->scope().type();
    if(pty == Scope::Type::point || pty == Scope::Type::classified_point
       || pty == Scope::Type::line || pty == Scope::Type::concrete_line
       || pty == Scope::Type::call || pty == Scope::Type::classified_call) {
      if(proc.prep()) {  // We're in charge of the tag, and this is a tag we want.
        std::ostringstream ss;
        ss << fancynames::unknown_proc.first << " "
              "0x" << std::hex << mo.second << " "
              "[" << (mo.first ? mo.first->path().filename().string() : "unknown module") << "]";
        proc.setTag(ss.str(), mo.second, fancynames::unknown_proc.second);
      }
      auto& udm = mo.first ? mo.first->userdata[exml.ud] : exml.unknown_module;
      auto& udf = fl.first ? fl.first->userdata[exml.ud] : udm.unknown_file;
      udf.incr(exml);
      std::ostringstream ss;
      ss << "<PF i=\"" << c.userdata[exml.src.identifier()] << "\""
               " lm=\"" << udm.id << "\""
               " n=\"" << proc.id << "\" s=\"" << proc.id << "\""
               " l=\"" << fl.second << "\""
               " f=\"" << udf.id << "\">\n";
      pre = ss.str();
      premetrics = true;
      post = "</PF>\n";
    } else {
      const File *file, *parent_file;
      if (s.type() == Scope::Type::point || s.type() == Scope::Type::call) {
        file = nullptr;
      } else {
        file = &(s.line_data()).first;
      }
      if (pty == Scope::Type::function || pty == Scope::Type::inlined_function) {
        parent_file = c.direct_parent()->scope().function_data().file;
      } else if (pty == Scope::Type::loop) {
        parent_file = &(c.direct_parent()->scope().line_data()).first;
      } else {
        parent_file = nullptr;
      }
      if (file && parent_file && file != parent_file) {
        if(proc.prep()) {  // We're in charge of the tag, and this is a tag we want.
          proc.setTag("inlined from " + file->path().filename().string(), mo.second, fancynames::unknown_proc.second);
        }
        auto& udm = mo.first ? mo.first->userdata[exml.ud] : exml.unknown_module;
        auto& udf = fl.first ? fl.first->userdata[exml.ud] : udm.unknown_file;
        udf.incr(exml);

        std::ostringstream ss;
        ss << "<Pr i=\"" << c.userdata[exml.src.identifier()] << "\""
              " n=\"" << proc.id << "\""
              " s=\"" << proc.id << "\""
              " f=\"" << udf.id << "\""
              " a=\"1\">\n";
        pre = ss.str();
        post = "</Pr>\n";
      }
    }

    open = "<";
    std::ostringstream ss;
    ss << " i=\"" << c.userdata[exml.src.identifier()] << "\""
          " s=\"" << proc.id << "\""
          " l=\"" << fl.second << "\"";
    attr = ss.str();
    if(mo.first) mo.first->userdata[exml.ud].incr(*mo.first, exml);
    break;
  }
  case Scope::Type::inlined_function: {
    auto fl = s.line_data();
    std::ostringstream ss;
    ss << "<C i=\"" << c.userdata[exml.src.identifier()] << "\""
            " s=\"" << proc.id << "\""
            " v=\"0\""
            " l=\"" << fl.second << "\">\n";
    pre = ss.str();
    premetrics = true;
    post = "</C>\n";
    (s.line_data().first).userdata[exml.ud].incr(exml);
    // fallthrough
  }
  case Scope::Type::function: {
    const auto& f = s.function_data();
    if(proc.prep()) {  // Our job to do the tag
      if(f.name.empty()) { // Anonymous function, write as an <unknown proc>
        std::ostringstream ss;
        ss << fancynames::unknown_proc.first << " "
              "0x" << std::hex << f.offset << " "
              "[" << f.module().path().string() << "]";
        proc.setTag(ss.str(), f.offset, fancynames::unknown_proc.second);
      } else {  // Normal function, but might have a name translation
        auto it = nametrans.find(f.name);
        if(it != nametrans.end()) proc.setTag(it->second.first, 0, it->second.second);
        else proc.setTag(f.name, f.offset, 0);
      }
    }
    auto& udm = f.module().userdata[exml.ud];
    auto& udf = f.file ? f.file->userdata[exml.ud] : udm.unknown_file;
    udm.incr(f.module(), exml);
    udf.incr(exml);
    std::ostringstream ss;
    ss << "<PF i=\"" << c.userdata[exml.src.identifier()] << "\""
             " l=\"" << f.line << "\""
             " f=\"" << udf.id << "\""
             " n=\"" << proc.id << "\" s=\"" << proc.id << "\""
             " lm=\"" << udm.id << "\"";
    open = ss.str();
    close = "</PF>\n";
    partial = false;  // If we have a Function, we're not lost. Probably.
    break;
  }
  }
}

// ExperimentXML4 bits

ExperimentXML4::ExperimentXML4(const fs::path& out, bool srcs, HPCTraceDB2* db)
  : ProfileSink(), dir(out), of(), next_id(0x7FFFFFFF), tracedb(db),
    include_sources(srcs), file_unknown(*this), next_procid(2),
    proc_unknown_proc(0), proc_partial_proc(1), unknown_module(*this),
    next_cid(0x7FFFFFFF) {
  if(dir.empty()) {  // Dry run
    util::log::info() << "ExperimentXML4 issuing a dry run!";
  } else {
    stdshim::filesystem::create_directory(dir);
    of.open((dir / "experiment.xml").native());
  }
}

ExperimentXML4::Proc& ExperimentXML4::getProc(const Scope& k) {
  return procs.emplace(k, next_procid.fetch_add(1, std::memory_order_relaxed)).first;
}

const ExperimentXML4::Proc& ExperimentXML4::proc_unknown() {
  proc_unknown_flag.call_nowait([&](){
    proc_unknown_proc.setTag(fancynames::unknown_proc.first, 0, fancynames::unknown_proc.second);
  });
  return proc_unknown_proc;
}

const ExperimentXML4::Proc& ExperimentXML4::proc_partial() {
  proc_partial_flag.call_nowait([&](){
    proc_partial_proc.setTag(fancynames::partial_unwind.first, 0, fancynames::partial_unwind.second);
  });
  return proc_partial_proc;
}

void ExperimentXML4::Proc::setTag(std::string n, std::size_t v, int fake) {
  std::ostringstream ss;
  ss << "<Procedure i=\"" << id << "\" n=" << util::xmlquoted(n)
     << " v=\"" << std::hex << (v == 0 ? "" : "0x") << v << "\"";
  if(fake > 0) ss << " f=\"" << fake << "\"";
  ss << "/>\n";
  tag = ss.str();
}

bool ExperimentXML4::Proc::prep() {
  bool x = false;
  return done.compare_exchange_strong(x, true, std::memory_order_relaxed);
}

void ExperimentXML4::notifyPipeline() noexcept {
  auto& ss = src.structs();
  ud.file = ss.file.add<udFile>(std::ref(*this));
  ud.context = ss.context.add<udContext>(std::ref(*this));
  ud.module = ss.module.add<udModule>(std::ref(*this));
  ud.metric = ss.metric.add<udMetric>(std::ref(*this));
}

void ExperimentXML4::write() {
  const auto& name = src.attributes().name().value();
  of << "<?xml version=\"1.0\"?>\n"
        "<HPCToolkitExperiment version=\"4.0\">\n"
        "<Header n=" << util::xmlquoted(name) << ">\n"
        "<Info/>\n"
        "</Header>\n"
        "<SecCallPathProfile i=\"0\" n=" << util::xmlquoted(name) << ">\n"
        "<SecHeader>\n";

  // MetricTable: from the Metrics
  of << "<MetricTable>\n";
  unsigned int id = 0;
  for(const auto& m: src.metrics().iterate()) {
    const auto& udm = m().userdata[ud];
    of << udm.metric_tags;
    id = std::max(id, udm.maxId);
  }
  for(const auto& es: src.extraStatistics().iterate()) {
    of << eStatMetricTags(es, id);
  }
  of << "</MetricTable>\n";

  of << "<MetricDBTable>\n";
  for(const auto& m: src.metrics().iterate()) of << m().userdata[ud].metricdb_tags;
  of << "</MetricDBTable>\n";
  if(tracedb != nullptr)
    of << "<TraceDBTable>\n" << tracedb->exmlTag() << "</TraceDBTable>\n";
  of << "<LoadModuleTable>\n";
  // LoadModuleTable: from the Modules
  if(unknown_module) of << unknown_module.tag;
  for(const auto& m: src.modules().iterate()) {
    auto& udm = m().userdata[ud];
    if(!udm) continue;
    of << udm.tag;
  }
  of << "</LoadModuleTable>\n"
        "<FileTable>\n";
  // FileTable: from the Files
  if(file_unknown) of << file_unknown.tag;
  for(const auto& f: src.files().iterate()) {
    auto& udf = f().userdata[ud];
    if(!udf) continue;
    of << udf.tag;
  }
  for(const auto& m: src.modules().iterate()) {
    auto& udm = m().userdata[ud].unknown_file;
    if(!udm) continue;
    of << udm.tag;
  }
  of << "</FileTable>\n"
        "<ProcedureTable>\n";
  // ProcedureTable: from the Functions for each Module.
  for(const auto& sp: procs.iterate()) of << sp.second.tag;
  if(proc_unknown_flag.query()) of << proc_unknown_proc.tag;
  if(proc_partial_flag.query()) of << proc_partial_proc.tag;
  of << "</ProcedureTable>\n"

        "<Info/>\n"
        "</SecHeader>\n";

  // Early check: the global Context must have id 0
  assert(src.contexts().userdata[src.identifier()] == 0 && "Global Context must have id 0!");

  // Spit out the CCT
  src.contexts().citerate([&](const Context& c){
    auto& udc = c.userdata[ud];

    // First emit our tags, and whatever extensions are nessesary.
    of << udc.pre << udc.open;
    switch(c.scope().type()) {
    case Scope::Type::unknown:
    case Scope::Type::global:
    case Scope::Type::loop:
    case Scope::Type::inlined_function:
    case Scope::Type::function:
      break;
    case Scope::Type::point:
    case Scope::Type::classified_point:
    case Scope::Type::call:
    case Scope::Type::classified_call:
    case Scope::Type::line:
    case Scope::Type::concrete_line:
      of << (c.children().empty() ? 'S' : 'C') << udc.attr;
      break;
    }
    if(c.scope().type() != Scope::Type::global)
      of << " it=\"" << c.userdata[src.identifier()] << "\"";

    if ((c.scope().type() == hpctoolkit::Scope::Type::point) || (c.scope().type() == hpctoolkit::Scope::Type::call) ||
        (c.scope().type() == hpctoolkit::Scope::Type::classified_point) || (c.scope().type() == hpctoolkit::Scope::Type::classified_call) ||
        (c.scope().type() == hpctoolkit::Scope::Type::concrete_line)) {
      uint64_t offset = c.scope().point_data().second;
      const std::string latency_blame_metric_name = "GINS: LAT_BLAME(cycles)";
      const auto& stats = c.statistics();

      for(const auto& mx: stats.citerate()) {
        const auto& m = mx.first;
        if(!m->scopes().has(MetricScope::function) || !m->scopes().has(MetricScope::execution))
          util::log::fatal{} << "Metric isn't function/execution!";
        const auto& vv = mx.second;

        if (m->name().find(latency_blame_metric_name) != std::string::npos) {
          int latency = *(vv.get(m->partials()[0]).get(MetricScope::point));
          // std::cout << "LATENCY_BLAME:: module: " << c.scope() << ", parent module: " << c.direct_parent()->scope() << ", offset: " << offset << ", val: " << latency << std::endl;
        }
      }
    }
    // If this is an empty tag, use the shorter form, otherwise close the tag.
    if(c.children().empty()) {
      of << "/>\n" << udc.post;
      return;
    }
    of << ">\n";
  }, [&](const Context& c){
    // If this is the shorter form, we have no ending tag
    if(c.children().empty()) return;

    auto& udc = c.userdata[ud];

    // Close off this tag.
    switch(c.scope().type()) {
    case Scope::Type::unknown:
    case Scope::Type::global:
    case Scope::Type::function:
    case Scope::Type::inlined_function:
    case Scope::Type::loop:
      break;
    case Scope::Type::point:
    case Scope::Type::classified_point:
    case Scope::Type::call:
    case Scope::Type::classified_call:
    case Scope::Type::line:
    case Scope::Type::concrete_line:
      of << "</" << (c.children().empty() ? 'S' : 'C') << ">\n";
      break;
    }
    of << udc.close << udc.post;
  });

  of << "</SecCallPathProfile>\n"
        "</HPCToolkitExperiment>\n" << std::flush;
}
