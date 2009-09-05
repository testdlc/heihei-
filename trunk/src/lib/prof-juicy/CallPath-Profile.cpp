// -*-Mode: C++;-*-

// * BeginRiceCopyright *****************************************************
//
// $HeadURL$
// $Id$
//
// -----------------------------------
// Part of HPCToolkit (hpctoolkit.org)
// -----------------------------------
// 
// Copyright ((c)) 2002-2009, Rice University 
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
//   $HeadURL$
//
// Purpose:
//   [The purpose of this file]
//
// Description:
//   [The set of functions, macros, etc. defined in the file]
//
//***************************************************************************

//************************* System Include Files ****************************

#include <iostream>
using std::hex;
using std::dec;

#include <string>
using std::string;

#include <map>

#include <typeinfo>

#include <cstdio>

#include <alloca.h>


//*************************** User Include Files ****************************

#include <include/uint.h>

#include "CallPath-Profile.hpp"
#include "Struct-Tree.hpp"

#include <lib/xml/xml.hpp>
using namespace xml;

#include <lib/prof-lean/hpcfmt.h>
#include <lib/prof-lean/hpcrun-fmt.h>

#include <lib/support/diagnostics.h>
#include <lib/support/RealPathMgr.hpp>
#include <lib/support/StrUtil.hpp>

//*************************** Forward Declarations **************************

#define DBG 0

//***************************************************************************


//***************************************************************************
// Profile
//***************************************************************************

namespace Prof {

namespace CallPath {


Profile::Profile(const std::string name, uint numMetrics)
{
  m_name = name;
  m_metricdesc.resize(numMetrics);
  for (uint i = 0; i < m_metricdesc.size(); ++i) {
    m_metricdesc[i] = new SampledMetricDesc();
  }
  m_loadmapMgr = new LoadMapMgr;
  m_cct = new CCT::Tree(this);
  m_structure = NULL;
}


Profile::~Profile()
{
  for (uint i = 0; i < m_metricdesc.size(); ++i) {
    delete m_metricdesc[i];
  }
  delete m_loadmapMgr;
  delete m_cct;
  delete m_structure;
}


void 
Profile::merge(Profile& y, bool isSameThread)
{
  DIAG_Assert(!m_structure && !y.m_structure, "Profile::merge: profiles should not have structure yet!");

  // -------------------------------------------------------
  // merge name, flags
  // -------------------------------------------------------
  DIAG_WMsgIf(m_flags.bits != y.m_flags.bits, "Prof::Profile::merge(): ignoring incompatible flags");

  // -------------------------------------------------------
  // merge metrics
  // -------------------------------------------------------
  uint x_numMetrics = numMetrics();
  uint x_newMetricBegIdx = 0;
  uint y_newMetrics   = 0;

  if (!isSameThread) {
    // new metrics columns
    x_newMetricBegIdx = x_numMetrics;
    y_newMetrics   = y.numMetrics();

    for (uint i = 0; i < y.numMetrics(); ++i) {
      const SampledMetricDesc* m = y.metric(i);
      addMetric(new SampledMetricDesc(*m));
    }
  }
  
  // -------------------------------------------------------
  // merge LoadMaps
  //
  // Post-INVARIANT: y's cct refers to x's LoadMapMgr
  // -------------------------------------------------------
  std::vector<LoadMap::MergeChange> mergeChg = 
    m_loadmapMgr->merge(*y.loadMapMgr());
  y.merge_fixCCT(mergeChg);

  // -------------------------------------------------------
  // merge CCTs
  // -------------------------------------------------------
  m_cct->merge(y.cct(), &m_metricdesc, x_newMetricBegIdx, y_newMetrics);
}


void 
Profile::merge_fixCCT(std::vector<LoadMap::MergeChange>& mergeChg)
{
  CCT::ANode* root = cct()->root();
  
  for (CCT::ANodeIterator it(root); it.CurNode(); ++it) {
    CCT::ANode* n = it.CurNode();
    
    CCT::ADynNode* n_dyn = dynamic_cast<CCT::ADynNode*>(n);
    if (n_dyn) {
      lush_lip_t* lip = n_dyn->lip();

      LoadMap::LM_id_t lmId1, lmId2;
      lmId1 = n_dyn->lmId_real();
      lmId2 = (lip) ? lush_lip_getLMId(lip) : LoadMap::LM_id_NULL;
      
      for (uint i = 0; i < mergeChg.size(); ++i) {
	const LoadMap::MergeChange& chg = mergeChg[i];
	if (chg.old_id == lmId1) {
	  n_dyn->lmId_real(chg.new_id);
	  if (lmId2 == LoadMap::LM_id_NULL) {
	    break; // quick exit in the common case
	  }
	}
	if (chg.old_id == lmId2) {
	  lush_lip_setLMId(lip, chg.new_id);
	}
      }
    }
  }
}


void 
writeXML_help(std::ostream& os, const char* entry_nm, 
	      Struct::Tree* structure, const Struct::ANodeFilter* filter,
	      int type)
{
  Struct::ANode* root = structure ? structure->root() : NULL;
  if (!root) {
    return;
  }

  for (Struct::ANodeIterator it(root, filter); it.Current(); ++it) {
    Struct::ANode* strct = it.CurNode();
    
    uint id = strct->id();
    const char* nm = NULL;
    
    if (type == 1) { // LoadModule
      nm = strct->name().c_str();
    }
    else if (type == 2) { // File
      nm = ((typeid(*strct) == typeid(Struct::Alien)) ? 
	    dynamic_cast<Struct::Alien*>(strct)->fileName().c_str() :
	    dynamic_cast<Struct::File*>(strct)->name().c_str());
    }
    else if (type == 3) { // Proc
      nm = strct->name().c_str();
    }
    else {
      DIAG_Die(DIAG_UnexpectedInput);
    }
    
    os << "    <" << entry_nm << " i" << MakeAttrNum(id) 
       << " n" << MakeAttrStr(nm) << "/>\n";
  }
}


static bool 
writeXML_FileFilter(const Struct::ANode& x, long type)
{
  return (typeid(x) == typeid(Struct::File) || typeid(x) == typeid(Struct::Alien));
}


static bool 
writeXML_ProcFilter(const Struct::ANode& x, long type)
{
  return (typeid(x) == typeid(Struct::Proc) || typeid(x) == typeid(Struct::Alien));
}


std::ostream& 
Profile::writeXML_hdr(std::ostream& os, int oFlags, const char* pre) const
{
  os << "  <MetricTable>\n";
  uint n_metrics = numMetrics();
  for (uint i = 0; i < n_metrics; i++) {
    const SampledMetricDesc* m = metric(i);
    os << "    <Metric i" << MakeAttrNum(i) 
       << " n" << MakeAttrStr(m->name()) << ">\n";
    os << "      <Info>" 
       << "<NV n=\"period\" v" << MakeAttrNum(m->period()) << "/>"
       << "<NV n=\"flags\" v" << MakeAttrNum(m->flags(), 16) << "/>"
       << "</Info>\n";
    os << "    </Metric>\n";
  }
  os << "  </MetricTable>\n";

  os << "  <LoadModuleTable>\n";
  writeXML_help(os, "LoadModule", m_structure, &Struct::ANodeTyFilter[Struct::ANode::TyLM], 1);
  os << "  </LoadModuleTable>\n";

  os << "  <FileTable>\n";
  Struct::ANodeFilter filt1(writeXML_FileFilter, "FileTable", 0);
  writeXML_help(os, "File", m_structure, &filt1, 2);
  os << "  </FileTable>\n";

  if ( !(oFlags & CCT::Tree::OFlg_Debug) ) {
    os << "  <ProcedureTable>\n";
    Struct::ANodeFilter filt2(writeXML_ProcFilter, "ProcTable", 0);
    writeXML_help(os, "Procedure", m_structure, &filt2, 3);
    os << "  </ProcedureTable>\n";
  }

  return os;
}


std::ostream&
Profile::dump(std::ostream& os) const
{
  os << m_name << std::endl;

  //m_metricdesc.dump(os);

  if (m_loadmapMgr) {
    m_loadmapMgr->dump(os);
  }

  if (m_cct) {
    m_cct->dump(os);
  }
  return os;
}


void 
Profile::ddump() const
{
  dump();
}


} // namespace CallPath

} // namespace Prof


//***************************************************************************
// 
//***************************************************************************

static std::pair<Prof::CCT::ANode*, Prof::CCT::ANode*>
cct_makeNode(const Prof::CCT::Tree& cct, 
	     const hpcrun_fmt_cct_node_t& nodeFmt,
	     Prof::CallPath::Profile& prof,
	     Prof::LoadMap* loadmap);

static void
fmt_cct_makeNode(hpcrun_fmt_cct_node_t& n_fmt, 
		 const Prof::CCT::ADynNode& n_dyn,
		 epoch_flags_t flags);


//***************************************************************************

namespace Prof {

namespace CallPath {

Profile* 
Profile::make(const char* fnm, FILE* outfs) 
{
  int ret;

  FILE* fs = hpcio_open_r(fnm);
  if (!fs) {
    DIAG_Throw("error opening file");
  }

  Profile* prof = NULL;
  ret = fmt_fread(prof, fs, fnm, outfs);
  
  hpcio_close(fs);

  return prof;
}


int
Profile::fmt_fread(Profile* &prof, FILE* infs, 
		   std::string ctxtStr, FILE* outfs)
{
  int ret;

  // ------------------------------------------------------------
  // hdr
  // ------------------------------------------------------------
  hpcrun_fmt_hdr_t hdr;
  ret = hpcrun_fmt_hdr_fread(&hdr, infs, malloc);
  if (ret != HPCFMT_OK) {
    DIAG_Throw("error reading 'fmt-hdr'");
  }
  if (outfs) {
    hpcrun_fmt_hdr_fprint(&hdr, outfs);
  }


  // ------------------------------------------------------------
  // epoch: Read each epoch and merge them to form one Profile
  // ------------------------------------------------------------
  
  prof = NULL;

  uint num_epochs = 0;
  while ( !feof(infs) ) {

    Profile* myprof = NULL;

    try {
      ctxtStr += ": epoch " + StrUtil::toStr(num_epochs + 1);
      ret = fmt_epoch_fread(myprof, infs, &hdr.nvps, ctxtStr, outfs);
      if (ret == HPCFMT_EOF) {
	break;
      }
    }
    catch (const Diagnostics::Exception& x) {
      delete myprof;
      DIAG_Throw("error reading 'epoch': " << x.what());
    }

    if (! prof) {
      prof = myprof;
    }
    else {
      prof->merge(*myprof, /*isSameThread*/true);
    }

    num_epochs++;
  }

  if (! prof) {
    prof = new Profile("[program-name]", 0);
    prof->cct_canonicalize();
  }

  if (outfs) {
    fprintf(outfs, "\n[You look fine today! (num-epochs: %d)]\n", num_epochs);
  }

  hpcrun_fmt_hdr_free(&hdr, free);

  return HPCFMT_OK;
}


int
Profile::fmt_epoch_fread(Profile* &prof, FILE* infs, 
			 HPCFMT_List(hpcfmt_nvpair_t)* hdrNVPairs,
			 std::string ctxtStr, FILE* outfs)
{
  using namespace Prof;

  int ret;

  // ------------------------------------------------------------
  // Read epoch data
  // ------------------------------------------------------------

  // ----------------------------------------
  // epoch-hdr
  // ----------------------------------------
  hpcrun_fmt_epoch_hdr_t ehdr;
  ret = hpcrun_fmt_epoch_hdr_fread(&ehdr, infs, malloc);
  if (ret == HPCFMT_EOF) {
    return HPCFMT_EOF;
  }
  if (ret != HPCFMT_OK) {
    DIAG_Throw("error reading 'epoch-hdr'");
  }
  if (outfs) {
    hpcrun_fmt_epoch_hdr_fprint(&ehdr, outfs);
  }

  // ----------------------------------------
  // metric-tbl
  // ----------------------------------------
  metric_tbl_t metric_tbl;
  ret = hpcrun_fmt_metricTbl_fread(&metric_tbl, infs, malloc);
  if (ret != HPCFMT_OK) {
    DIAG_Throw("error reading 'metric-tbl'");
  }
  if (outfs) {
    hpcrun_fmt_metricTbl_fprint(&metric_tbl, outfs);
  }

  uint num_metrics = metric_tbl.len;
  
  // ----------------------------------------
  // loadmap
  // ----------------------------------------
  loadmap_t loadmap_tbl;
  ret = hpcrun_fmt_loadmap_fread(&loadmap_tbl, infs, malloc);
  if (ret != HPCFMT_OK) {
    DIAG_Throw("error reading 'loadmap'");
  }
  if (outfs) {
    hpcrun_fmt_loadmap_fprint(&loadmap_tbl, outfs);
  }

  // ------------------------------------------------------------
  // Create Profile
  // ------------------------------------------------------------

  // ----------------------------------------
  // obtain meta information
  // ----------------------------------------

  const char* val;

  string progNm;
  val = hpcfmt_nvpair_search(hdrNVPairs, HPCRUN_FMT_NV_prog);
  if (val && strlen(val) > 0) {
    progNm = val;
  }

  string mpiRank, tid;
  //const char* jobid = hpcfmt_nvpair_search(hdrNVPairs, HPCRUN_FMT_NV_jobId);
  val = hpcfmt_nvpair_search(hdrNVPairs, HPCRUN_FMT_NV_mpiRank);
  if (val) { mpiRank = val; }
  val = hpcfmt_nvpair_search(hdrNVPairs, HPCRUN_FMT_NV_tid);
  if (val) { tid = val; }

  // FIXME: temporary for dual-interpretations
  bool isNewFormat = true; 
  val = hpcfmt_nvpair_search(hdrNVPairs, "nasty-message");
  if (val) { isNewFormat = false; }

  //val = hpcfmt_nvpair_search(ehdr.&nvps, "to-find");

  // ----------------------------------------
  // 
  // ----------------------------------------
  
  prof = new Profile(progNm, num_metrics);
  prof->m_flags = ehdr.flags;
  
  // ----------------------------------------
  // add metrics
  // ----------------------------------------

  string m_sfx;
  if (!mpiRank.empty() && !tid.empty()) {
    m_sfx = " [" + mpiRank + "," + tid + "]";
  }
  else if (!mpiRank.empty()) {
    m_sfx = " [" + mpiRank + "]";
  }
  else if (!tid.empty()) {
    m_sfx = " [" + tid + "]";    
  }

  metric_desc_t* m_lst = metric_tbl.lst;
  for (uint i = 0; i < num_metrics; i++) {
    SampledMetricDesc* metric = prof->metric(i);
    string m_nm = m_lst[i].name + m_sfx;
    metric->name(m_nm);
    metric->flags(m_lst[i].flags);
    metric->period(m_lst[i].period);
  }

  hpcrun_fmt_metricTbl_free(&metric_tbl, free);

  // ----------------------------------------
  // add loadmap
  // ----------------------------------------
  uint num_lm = loadmap_tbl.len;

  LoadMap loadmap(num_lm);

  for (uint i = 0; i < num_lm; ++i) { 
    string nm = loadmap_tbl.lst[i].name;
    RealPathMgr::singleton().realpath(nm);
    VMA loadAddr = loadmap_tbl.lst[i].mapaddr;
    size_t sz = 0; //loadmap_tbl->epoch_modlist[loadmap_id].loadmodule[i].size;

    LoadMap::LM* lm = new LoadMap::LM(nm, loadAddr, sz);
    loadmap.lm_insert(lm);
    
    DIAG_Assert(lm->id() == i + 1, "FIXME: Profile::fmt_epoch_fread: Expect lm id's to be in order to support dual-interpretations.");
  }

  DIAG_MsgIf(DBG, loadmap.toString());

  try {
    loadmap.compute_relocAmt();
  }
  catch (const Diagnostics::Exception& x) {
    DIAG_EMsg(ctxtStr << ": Cannot fully process samples from unavailable load modules:\n" << x.what());
  }

  std::vector<ALoadMap::MergeChange> mergeChg = 
    prof->loadMapMgr()->merge(loadmap);
  DIAG_Assert(mergeChg.empty(), "Profile::fmt_epoch_fread: " << DIAG_UnexpectedInput);


  hpcrun_fmt_loadmap_free(&loadmap_tbl, free);


  // ------------------------------------------------------------
  // cct
  // ------------------------------------------------------------
  LoadMap* loadmap_p = (isNewFormat) ? NULL : &loadmap; // FIXME:temporary
  fmt_cct_fread(*prof, infs, loadmap_p, outfs);

  prof->cct_canonicalize();


  hpcrun_fmt_epoch_hdr_free(&ehdr, free);
  
  return HPCFMT_OK;
}


int
Profile::fmt_cct_fread(Profile& prof, FILE* infs, LoadMap* loadmap, FILE* outfs)
{
  typedef std::map<int, CCT::ANode*> CCTIdToCCTNodeMap;

  DIAG_Assert(infs, "Bad file descriptor!");
  
  CCTIdToCCTNodeMap cctNodeMap;

  int ret = HPCFMT_ERR;

  if (outfs) {
    fprintf(outfs, "{cct:\n"); 
  }

  // ------------------------------------------------------------
  // Read num cct nodes
  // ------------------------------------------------------------
  uint64_t num_nodes = 0;
  hpcfmt_byte8_fread(&num_nodes, infs);

  // ------------------------------------------------------------
  // Read each CCT node
  // ------------------------------------------------------------

  hpcrun_fmt_cct_node_t nodeFmt;
  nodeFmt.num_metrics = prof.numMetrics();
  nodeFmt.metrics = 
    (hpcrun_metricVal_t*)alloca(prof.numMetrics() * sizeof(hpcrun_metricVal_t));

  for (uint i = 0; i < num_nodes; ++i) {

    // ----------------------------------------------------------
    // Read the node
    // ----------------------------------------------------------
    ret = hpcrun_fmt_cct_node_fread(&nodeFmt, prof.m_flags, infs);
    if (ret != HPCFMT_OK) {
      DIAG_Throw("Error reading CCT node " << nodeFmt.id);
    }
    if (outfs) {
      hpcrun_fmt_cct_node_fprint(&nodeFmt, outfs, prof.m_flags, "  ");
    }

    // Find parent of node
    CCT::ANode* node_parent = NULL;
    if (nodeFmt.id_parent != HPCRUN_FMT_CCTNodeId_NULL) {
      CCTIdToCCTNodeMap::iterator it = cctNodeMap.find(nodeFmt.id_parent);
      if (it != cctNodeMap.end()) {
	node_parent = it->second;
      }
      else {
	DIAG_Throw("Cannot find parent for node " << nodeFmt.id);	
      }
    }

    if ( !(nodeFmt.id_parent < nodeFmt.id) ) {
      DIAG_Throw("Invalid parent " << nodeFmt.id_parent << " for node " << nodeFmt.id);
    }

    // ----------------------------------------------------------
    // Create node and link to parent
    // ----------------------------------------------------------

    CCT::Tree* cct = prof.cct();

    std::pair<CCT::ANode*, CCT::ANode*> n2 = cct_makeNode(*cct, nodeFmt, prof, loadmap);
    CCT::ANode* node = n2.first;
    CCT::ANode* node_sib = n2.second;

    DIAG_DevMsgIf(0, "fmt_cct_fread: " << hex << node << " -> " << node_parent << dec);

    if (node_parent) {
      node->Link(node_parent);
      if (node_sib) {
	node_sib->Link(node_parent);
      }
    }
    else {
      DIAG_Assert(cct->empty() && !node_sib, "Must only have one root node!");
      cct->root(node);
    }

    cctNodeMap.insert(std::make_pair(nodeFmt.id, node));
  }

  if (outfs) {
    fprintf(outfs, "}\n"); 
  }

  return HPCFMT_OK;
}


//***************************************************************************

int
Profile::fmt_fwrite(const Profile& prof, FILE* fs)
{
  // ------------------------------------------------------------
  // header
  // ------------------------------------------------------------
  hpcrun_fmt_hdr_fwrite(fs, 
			"TODO:hdr-name","TODO:hdr-value",
			NULL);

  // ------------------------------------------------------------
  // epoch
  // ------------------------------------------------------------
  fmt_epoch_fwrite(prof, fs);

  return HPCFMT_OK;
}


int
Profile::fmt_epoch_fwrite(const Profile& prof, FILE* fs)
{
  // ------------------------------------------------------------
  // epoch-hdr
  // ------------------------------------------------------------
  
  hpcrun_fmt_epoch_hdr_fwrite(fs, prof.m_flags,
			      0 /*TODO:default_ra_distance*/,
			      0 /*TODO:default_granularity*/,
			      "TODO:epoch-name","TODO:epoch-value",
			      NULL);

  // ------------------------------------------------------------
  // metric-tbl
  // ------------------------------------------------------------

  hpcfmt_byte4_fwrite(prof.numMetrics(), fs);
  for (uint i = 0; i < prof.numMetrics(); i++) {
    const SampledMetricDesc* m = prof.metric(i);

    metric_desc_t mdesc;
    mdesc.name = const_cast<char*>(m->name().c_str());
    mdesc.flags = m->flags();
    mdesc.period = m->period();

    hpcrun_fmt_metricDesc_fwrite(&mdesc, fs);
  }


  // ------------------------------------------------------------
  // loadmap
  // ------------------------------------------------------------

  LoadMapMgr* loadMapMgr = prof.loadMapMgr();

  hpcfmt_byte4_fwrite(loadMapMgr->size(), fs);
  for (ALoadMap::LM_id_t i = 1; i <= loadMapMgr->size(); i++) {
    const ALoadMap::LM* lm = loadMapMgr->lm(i);

    loadmap_entry_t lm_entry;
    lm_entry.id = lm->id();
    lm_entry.name = const_cast<char*>(lm->name().c_str());
    lm_entry.vaddr = 0;
    lm_entry.mapaddr = 0;
    lm_entry.flags = 0; // TODO:flags
    
    hpcrun_fmt_loadmapEntry_fwrite(&lm_entry, fs);
  }

  // ------------------------------------------------------------
  // cct
  // ------------------------------------------------------------
  fmt_cct_fwrite(prof, fs);

  return HPCFMT_OK;
}


int
Profile::fmt_cct_fwrite(const Profile& prof, FILE* fs)
{
  uint numMetrics = prof.numMetrics();

  hpcrun_fmt_cct_node_t nodeFmt;
  nodeFmt.num_metrics = numMetrics;
  nodeFmt.metrics = 
    (hpcrun_metricVal_t*) alloca(numMetrics * sizeof(hpcrun_metricVal_t));

  CCT::ANode* root = prof.cct()->root(); // FIXME: find the original root...

  for (CCT::ANodeIterator it(root); it.CurNode(); ++it) {
    CCT::ANode* n = it.CurNode();

    CCT::ADynNode* n_dyn = dynamic_cast<CCT::ADynNode*>(n);
    DIAG_Assert(n_dyn, "Profile::fmt_cct_fwrite: " << DIAG_UnexpectedInput);
    
    if (n_dyn) {
      fmt_cct_makeNode(nodeFmt, *n_dyn, prof.m_flags);
      int ret = hpcrun_fmt_cct_node_fwrite(&nodeFmt, prof.m_flags, fs);
      if (ret != HPCFMT_OK) {
	return HPCFMT_ERR;
      }
    }
  }

  return HPCFMT_OK;
}


//***************************************************************************

// 1. Create a (PGM) root for the CCT
// 2. Remove the two outermost frames: 
//      "synthetic-root -> monitor_main"
void
Profile::cct_canonicalize()
{
  using namespace Prof;

  CCT::ANode* root = m_cct->root();

  // idempotent
  if (root && typeid(*root) == typeid(CCT::Root)) {
    return;
  }

  CCT::ANode* newRoot = new CCT::Root(m_name);

  // 1. find the splice point
  CCT::ANode* spliceRoot = root;
  if (root && root->ChildCount() == 1) {
    spliceRoot = root->firstChild();
  }
  
  // 2. splice: move all children of 'spliceRoot' to 'newRoot'
  if (spliceRoot) {
    for (CCT::ANodeChildIterator it(spliceRoot); it.Current(); /* */) {
      CCT::ANode* n = it.CurNode();
      it++; // advance iterator -- it is pointing at 'n'
      n->Unlink();
      n->Link(newRoot);
    }
    
    delete root; // N.B.: also deletes 'spliceRoot'
  }
  
  m_cct->root(newRoot);
}


} // namespace CallPath

} // namespace Prof


//***************************************************************************


static std::pair<Prof::CCT::ANode*, Prof::CCT::ANode*>
cct_makeNode(const Prof::CCT::Tree& cct, const hpcrun_fmt_cct_node_t& nodeFmt,
	     Prof::CallPath::Profile& prof, /*FIXME:temp*/
	     Prof::LoadMap* loadmap /*FIXME:temp*/)
{
  using namespace Prof;

  // ----------------------------------------------------------
  // Gather node parameters
  // ----------------------------------------------------------
  bool isLeaf = false;

  // ----------------------------------------
  // cpId
  // ----------------------------------------
  uint cpId = 0;
  int id_tmp = (int)nodeFmt.id;
  if (id_tmp < 0) {
    isLeaf = true;
    id_tmp = -id_tmp;
  }
  if (hpcrun_fmt_doRetainId(nodeFmt.id)) {
    cpId = id_tmp;
  }

  // ----------------------------------------
  // lmId and ip
  // ----------------------------------------
  ALoadMap::LM_id_t lmId = nodeFmt.lm_id;

  VMA ip = (VMA)nodeFmt.ip; // FIXME:tallent: Use ISA::ConvertVMAToOpVMA
  ushort opIdx = 0;

  if (loadmap) {
    VMA ip_orig = ip;
    LoadMap::LM* lm = loadmap->lm_find(ip_orig);

    ip = ip_orig - lm->relocAmt(); // unrelocated ip
    lmId = lm->id();
  }

  if (lmId != ALoadMap::LM_id_NULL) {
    prof.loadMapMgr()->lm(lmId)->isUsed(true);
  }

  DIAG_MsgIf(0, "cct_makeNode(: " << hex << ip << dec << ", " << lmId << ")");

  // ----------------------------------------  
  // lip
  // ----------------------------------------
  lush_lip_t* lip = NULL;
  if (!lush_lip_eq(&nodeFmt.lip, &lush_lip_NULL)) {
    lip = new lush_lip_t;
    memcpy(lip, &nodeFmt.lip, sizeof(lush_lip_t));
  }

  if (lip) {
    if (loadmap) {
      VMA lip_ip = lush_lip_getIP(lip);

      LoadMap::LM* lm = loadmap->lm_find(lip_ip);
      
      lush_lip_setLMId(lip, lm->id());
      lush_lip_setIP(lip, lip_ip - lm->relocAmt()); // unrelocated ip
    }

    ALoadMap::LM_id_t lip_lmId = lush_lip_getLMId(lip);
    if (lip_lmId != ALoadMap::LM_id_NULL) {
      prof.loadMapMgr()->lm(lip_lmId)->isUsed(true);
    }
  }

  // ----------------------------------------  
  // metrics
  // ----------------------------------------  
  bool hasMetrics = false;
  std::vector<hpcrun_metricVal_t> metricVec(nodeFmt.num_metrics);
  for (uint i = 0; i < nodeFmt.num_metrics; i++) {
    hpcrun_metricVal_t m = nodeFmt.metrics[i];
    metricVec[i] = m;
    if (!hpcrun_metricVal_isZero(m)) {
      hasMetrics = true;
    }
  }


  // ----------------------------------------------------------
  // Create nodes.  
  //
  // Note that it is possible for an interior node to have
  // a non-zero metric count.  If this is the case, the node should be
  // split into two sibling nodes: 1) an interior node with metrics
  // == 0 (that has cpId == 0 *and* that is the primary return node);
  // and 2) a leaf node with the metrics and the cpId.
  // ----------------------------------------------------------
  Prof::CCT::ANode* n = NULL;
  Prof::CCT::ANode* n_leaf = NULL;

  if (hasMetrics || isLeaf) {
    n = new CCT::Stmt(NULL, cpId, nodeFmt.as_info, lmId, ip, opIdx, lip,
		      &(cct.metadata()->metricDesc()), metricVec);
  }

  if (!isLeaf) {
    if (hasMetrics) {
      n_leaf = n;

      std::vector<hpcrun_metricVal_t> metricVec0(nodeFmt.num_metrics);
      n = new CCT::Call(NULL, 0, nodeFmt.as_info, lmId, ip, opIdx, lip,
			&(cct.metadata()->metricDesc()), metricVec0);
    }
    else {
      n = new CCT::Call(NULL, cpId, nodeFmt.as_info, lmId, ip, opIdx, lip,
			&(cct.metadata()->metricDesc()), metricVec);
    }
  }

  return std::make_pair(n, n_leaf);
}


static void
fmt_cct_makeNode(hpcrun_fmt_cct_node_t& n_fmt, 
		 const Prof::CCT::ADynNode& n_dyn,
		 epoch_flags_t flags)
{
  n_fmt.id = (n_dyn.isLeaf()) ? -(n_dyn.id()) : n_dyn.id();

  n_fmt.id_parent = (n_dyn.parent()) ? n_dyn.parent()->id() : 0;

  if (flags.flags.isLogicalUnwind) {
    n_fmt.as_info = n_dyn.assocInfo();
  }
      
  n_fmt.lm_id = n_dyn.lmId();
  
  n_fmt.ip = n_dyn.Prof::CCT::ADynNode::ip();

  if (flags.flags.isLogicalUnwind) {
    lush_lip_init(&n_fmt.lip);
    if (n_dyn.lip()) {
      memcpy(&n_fmt.lip, n_dyn.lip(), sizeof(lush_lip_t));
    }
  }

  for (uint i = 0; i < n_dyn.numMetrics(); ++i) {
    n_fmt.metrics[i] = n_dyn.metric(i);
  }
}

