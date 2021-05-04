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

#ifndef HPCTOOLKIT_PROFILE_ANALYZE_H
#define HPCTOOLKIT_PROFILE_ANALYZE_H

#include <iostream>   //cout
#include "pipeline.hpp"

namespace hpctoolkit {

class ProfileAnalyzer {
public:
  virtual ~ProfileAnalyzer() = default;

  void bindPipeline(ProfilePipeline::Source&& se) noexcept {
    sink = std::move(se);
  }

  /// Query what Classes this Source can actually provide to the Pipeline.
  // MT: Safe (const)
  DataClass provides() const noexcept {
    return DataClass::metrics + DataClass::attributes;
  }

  virtual void context(Context& c) {}
  virtual void analysisMetricsFor(const Metric& m) {}
  virtual void analyze(Thread::Temporary& t) {}

protected:
  // Use a subclass to implement the bits.
  ProfileAnalyzer() = default;

  // Source of the Pipeline to use for inserting data: our sink.
  ProfilePipeline::Source sink;
};

class context_map_compare {
  public:
    size_t operator()(const std::pair<uint64_t, ContextRef>& p) const {
      Context& ctx = std::get<Context>(p.second);
      std::size_t hash_contextref = std::hash<const Context*>{}(&ctx);
      return (p.first^hash_contextref);
    }
};

struct LatencyBlameAnalyzer : public ProfileAnalyzer {
  LatencyBlameAnalyzer() = default;
  ~LatencyBlameAnalyzer() = default;

  void context(Context& ctx) noexcept override {

    const Scope& s = ctx.scope();
    if(s.type() == Scope::Type::point || s.type() == Scope::Type::call ||
        s.type() == Scope::Type::classified_point || s.type() == Scope::Type::classified_call) {
      //std::cout << "context: " << ctx.scope() << std::endl;
      std::string scopeFile = s.point_data().first.path().filename().string();
      size_t scopeHash = std::hash<std::string>{}(scopeFile);
      uint64_t offset = ctx.scope().point_data().second;
      auto mo = s.point_data();
      const auto& c = mo.first.userdata[sink.classification()];
      const std::map<uint64_t, uint32_t> empty_edges = {};
      auto iter = c._def_use_graph.find(offset);
      const std::map<uint64_t, uint32_t> &incoming_edges = (iter != c._def_use_graph.end()) ?
        iter->second: empty_edges;
      if (def_use_graph[scopeHash].size() == 0 && c._def_use_graph.size() > 0) {
        // std::cout << "scope: " << scopeFile << ", incoming edges: " << incoming_edges.size() << ", graph size: " << c._def_use_graph.size() << std::endl;
        def_use_graph[scopeHash] = c._def_use_graph;
      }

      for (auto edge: incoming_edges) {
        uint64_t from = edge.first;
        auto parent = ctx.direct_parent();
        while (parent) {
          const Scope& s = parent->scope();
          if(s.type() == Scope::Type::point || s.type() == Scope::Type::call || s.type() == Scope::Type::classified_point || s.type() == Scope::Type::classified_call) {
            break;
          }
          parent = parent->direct_parent();
        }
        // std::cout << "incoming edges" << std::endl;
        if (parent) {
          ContextRef cr = sink.context(*parent, {ctx.scope().point_data().first, from}, false);
          context_map[{offset, ContextRef(ctx)}].insert({from, cr});
        }
      }
    }
  }

  void analysisMetricsFor(const Metric& m) noexcept override {
    const std::string latency_metric_name = "GINS: LAT(cycles)";
    const std::string frequency_metric_name = "GINS: EXC_CNT";
    const std::string name = "GINS: LAT_BLAME(cycles)";
    const std::string desc = "Accumulates the latency blame for a given context";
    // std::cout << "metric name: " << m.name() << std::endl;
    if (m.name().find(latency_metric_name) != std::string::npos) {
      latency_metric = &m;
      latency_blame_metric = &(sink.metric(Metric::Settings(name, desc)));
      sink.metricFreeze(*latency_blame_metric);
    } else if (m.name().find(frequency_metric_name) != std::string::npos) {
      frequency_metric = &m;
    }
  }

  void analyze(Thread::Temporary& t) noexcept override {
    std::map<size_t, std::map<uint64_t, double>> denominator_map;

    // std::cout << "analyze" << std::endl;
    for (auto iter1: context_map) {
      // std::cout << "iter1" << std::endl;
      const std::pair<uint64_t, ContextRef> to_obj = iter1.first;
      uint64_t to = to_obj.first;
      ContextRef to_context = to_obj.second;
      const Scope& s = std::get<Context>(to_context).scope();
      std::string scopeFile = s.point_data().first.path().filename().string();
      size_t scopeHash = std::hash<std::string>{}(scopeFile);

      double denominator = 0;
      for (auto iter2: iter1.second) {
        // std::cout << "iter2" << std::endl;
        uint64_t from = iter2.first;
        ContextRef cr = iter2.second;
        auto from_ctx = t.accumulators().find(std::get<Context>(cr));
        if (!from_ctx) {
          // std::cout << "from ctx1 null" << std::endl;
          continue;
        }
        // std::cout << "from ctx1 not null" << std::endl;
        auto from_freq_metric = from_ctx->find(*frequency_metric);
        if (!from_freq_metric) {
          continue;
        }
        const std::optional<double> ef_ptr = from_freq_metric->get(MetricScope::point);
        if (ef_ptr) {
          double execution_frequency = *ef_ptr;
          double path_length_inv = (double) 1 / (def_use_graph[scopeHash][to][from]);
          denominator += execution_frequency * path_length_inv;
        }
      }
      denominator_map[scopeHash][to] = denominator;
    }
    for (auto iter1: context_map) {
      std::pair<uint64_t, ContextRef> to_obj = iter1.first;
      uint64_t to = to_obj.first;
      ContextRef to_context = to_obj.second;
      const Scope& s = std::get<Context>(to_context).scope();
      std::string scopeFile = s.point_data().first.path().filename().string();
      size_t scopeHash = std::hash<std::string>{}(scopeFile);

      int latency;
      auto to_ctx = t.accumulators().find(std::get<Context>(to_context));
      if (!to_ctx) {
        // std::cout << "to ctx null" << std::endl;
        continue;
      }
      // std::cout << "to ctx not null" << std::endl;
      auto to_lat_metric = to_ctx->find(*latency_metric);
      if (!to_lat_metric) {
        continue;
      }
      const std::optional<double> l_ptr = to_lat_metric->get(MetricScope::point);
      if (l_ptr) {
        latency = *l_ptr;
      }
      for (auto iter2: iter1.second) {
        uint64_t from = iter2.first;
        ContextRef cr = iter2.second;
        auto from_ctx = t.accumulators().find(std::get<Context>(cr));
        if (!from_ctx) {
          // std::cout << "from ctx2 null" << std::endl;
          continue;
        }
        // std::cout << "from ctx2 not null" << std::endl;
        auto from_freq_metric = from_ctx->find(*frequency_metric);
        if (!from_freq_metric) {
          continue;
        }
        const std::optional<double> ef_ptr = from_freq_metric->get(MetricScope::point);
        if (ef_ptr) {
          double execution_frequency = *ef_ptr;
          double path_length_inv = (double) 1 / (def_use_graph[scopeHash][to][from]);
          double latency_blame = execution_frequency * path_length_inv / denominator_map[scopeHash][to] * latency;
          // std::cout << "LAT_BLAME (analyze):: scope: " << scopeFile << ", graph size: " << def_use_graph[scopeHash].size() << ", offset: " << from << ", val: " << latency_blame << std::endl;
          sink.accumulateTo(cr, t).add(*latency_blame_metric, latency_blame);
        }
      }
    }
  }

private:
  // offset -> ContextRef
  std::unordered_map <std::pair<uint64_t, ContextRef>, std::unordered_map<uint64_t, ContextRef>, context_map_compare>context_map;

  const Metric *latency_metric;
  const Metric *frequency_metric;
  Metric *latency_blame_metric;

  std::map<size_t, std::map<uint64_t , std::map<uint64_t, uint32_t>>> def_use_graph;
};

}

#endif  // HPCTOOLKIT_PROFILE_ANALYZE_H
