// $Id$
// -*-C++-*-
// * BeginRiceCopyright *****************************************************
// 
// Copyright ((c)) 2002, Rice University 
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

//************************ System Include Files ******************************

#include <iostream> 
using std::cout;
using std::cerr;
using std::endl;

#include <string>
using std::string;

#include <time.h> 
#include <sys/time.h>
#include <sys/types.h> 
#include <sys/stat.h> 

//************************* User Include Files *******************************

#include "Args.hpp"
#include "Driver.hpp"
#include "HTMLDriver.hpp"
#include "StaticFiles.hpp"
#include "HTMLFile.hpp"
#include "HTMLTable.hpp"
#include "HTMLScopes.hpp"
#include "HTMLSrcFiles.hpp"

#include <lib/prof-juicy/PgmScopeTree.hpp>

#include <lib/support/IntVector.hpp>
#include <lib/support/StrUtil.hpp>
#include <lib/support/String.hpp> // FIXME
#include <lib/support/Files.hpp>
#include <lib/support/Nan.h>
#include <lib/support/Assertion.h>
#include <lib/support/Trace.hpp>

//************************ Forward Declarations ******************************

//****************************************************************************

// ----------------------------------------------------------------------------
// the javascript code in utils.js relays on these names for frames 
// also StaticFiles  assumes that there is a scopes subdirectory in the 
// generated html database that it can copy the SCOPES_UTILS_HERE file to, 
// which is meant to be used with files generated by HTMLScopes
const char* SOURCE       = "source"; 
const char* SCOPES_SELF  = "scopes_self"; 
const char* SCOPES_KIDS  = "scopes_kids"; 
const char* ScopesDir    = "scopes"; 

// ----------------------------------------------------------------------------

// javascript also relieds on the fact that the performance table frame 
// is called PerfTable.ANY 

// some other constants  for frame names 
static const char* EMPTY   = "empty"; 
static const char* HEADER  = "header"; 

const DataDisplayInfo 
HTMLDriver::NameDisplayInfo = DataDisplayInfo("Location", "black", 25, false); 


static String
UniqueNameFilePrefix(const FileScope *f) 
{
  String nm; 
  nm = String("F") + StrUtil::toStr(f->UniqueId()).c_str(); 
  return nm; 
}
 
String 
HTMLDriver::UniqueNameForSelf(const ScopeInfo *s) 
{
  String nm; 
  nm = ScopeInfo::ScopeTypeToName(s->Type()).c_str()[0]; 
  nm += StrUtil::toStr(s->UniqueId()).c_str(); 
  return nm; 
}

String
HTMLDriver::UniqueName(const ScopeInfo *c, int pIndex, int flattenDepth) 
{
  // must generate names such that: 
  // if (c->File() != NULL && if c->File() != c
  //     UniqueName(c) == UniqueName(c->File()) + "_" + <non empty whateveer> 
  // Why ? because javascript code relies on this naming convention 
  //
  BriefAssertion(c != NULL); 
  String nm; 
  ScopeInfo::ScopeType t = c->Type();
  if (t == ScopeInfo::PGM || t == ScopeInfo::GROUP || t == ScopeInfo::LM) {
    nm = UniqueNameForSelf(c);
  } else {
    const FileScope *f = c->File(); 
    BriefAssertion(f != NULL); 
    nm = UniqueNameFilePrefix(f); 
    if (f != c) {
      nm += "_" + UniqueNameForSelf(c); 
    }
  }
  if (pIndex != NO_PERF_INDEX) {
    nm += String(".") + IndexToPerfDataInfo(pIndex).Name().c_str();
  }

  if (flattenDepth != NO_FLATTEN_DEPTH) {
    nm += String(".") + StrUtil::toStr(flattenDepth).c_str();
  }
  return nm; 
}

HTMLDriver::HTMLDriver(const PgmScopeTree& scps, 
		       const char* fHm, 
		       const char* htmlDr,
		       const Args& pgmArgs)
  : Unique("HTMLDriver"), scopes(scps), fileHome(fHm), 
    htmlDir(htmlDr), args(pgmArgs)
{
  string scpsDir = string(htmlDir) + "/" + ScopesDir; 
  if (MakeDir(scpsDir.c_str()) != 0 ||
      StaticFiles(fileHome, htmlDir).CopyAllFiles(args.OldStyleHTML) != 0) {
    exit(1);
  }
  IFTRACE << "HTMLDriver::" << ToString() << endl; 
}

HTMLDriver::~HTMLDriver() 
{
  IFTRACE << "~HTMLDriver::" << ToString() << endl; 
}

string
HTMLDriver::ToString() const 
{
  return string("HTMLDriver: ")  + 
    string("fileHome=") + fileHome +   " "  + 
    string("htmlDir=") + htmlDir +   " "  + 
    string("&scopes=") + StrUtil::toStr((void*)&scopes); 
}

bool
HTMLDriver::Write(const Driver& driver)  const
{
  IFTRACE << "HTMLDriver::Write " << endl 
	  << "   this=" << ToString()  << endl
	  << "   driver=" << driver.ToString() << endl; 
  // here we rely on the fact that PerfMetrics and MetricInfos 
  // have the same indexes 
  IntVector displayMetrics; 
  int firstSorted = (-1);
  for (unsigned int i = 0; i < driver.NumberOfMetrics(); i++) {
    if (driver.PerfDataSrc(i).Display()) {
       displayMetrics.push_back(i); 
       if (driver.PerfDataSrc(i).SortBy() && (firstSorted==(-1))) {
          firstSorted = i;
       }
    } 
  } 
  if (displayMetrics.size() == 0) {
    cerr << "ERROR: No valid metric to display." << endl; 
    return false; 
  } 
  if (firstSorted == -1) {
    cerr << "hpcview warning: CONFIGURATION file does not specify a sortBy attribute for any METRIC."  << endl
	 << "        Sorting on the first displayed metric by default." << endl
         << "        Avoid this warning by adding the attribute "
	 << "sortBy=\"true\" to one or more METRIC definitions." << endl;
    firstSorted = displayMetrics[0];
    IndexToPerfDataInfo(firstSorted).setSortBy();
  }
  displayMetrics.push_back(-1); 

  HTMLFile hf(htmlDir, EMPTY, NULL);   
  hf.SetBgColor("white");
  
  HTMLSrcFiles srcFiles(scopes, args, ScopeTypeFilter[ScopeInfo::ANY], 
			/*leavesOnly=*/ true); 

  srcFiles.WriteLabels(htmlDir, "", "#e0e0e0"); 
  srcFiles.WriteSrcFiles(htmlDir); 
  
  // make a table of all leaves in scopes.Root() 
  // showing their performance numbers
  HTMLTable table("PerfTable", scopes, ScopeTypeFilter[ScopeInfo::ANY], 
		  /*leavesOnly=*/ true, &displayMetrics, args); 
  table.Write(htmlDir, "#e0e0e0", "white"); 

  // make quantify like scope tables 
  HTMLScopes scopeTables(scopes, &displayMetrics, args); 
  scopeTables.Write(htmlDir, "white", "#e0e0e0"); 

  // miscellaneous files
  WriteFiles("files"); 
  WriteHeader(HEADER, driver.Title()); 
  WriteIndexFile(scopes.GetRoot(), table, firstSorted, scopeTables, 
		 driver.Title(), HEADER);
  return true; 
} 

static void 
WriteIndexFile(const char* htmlDir, 
	       PgmScope* pgmScope, 
	       HTMLTable &table, int perfIndex, 
	       HTMLScopes &scopes, 
	       const string& title, 
	       const char* header, 
	       bool debug, bool oldStyleHtml)
{
  BriefAssertion(table.Name().Length() > 0); 
  
  struct timeval t; 
  gettimeofday(&t,0); 
  char date[50]; 
  strftime(date, 50, "%x %X", localtime(&t.tv_sec)); 
  string tit = title + " &nbsp; (" + date + ")"; 
  
  string fname = "index"; 
  if (debug) {
    fname += ".debug"; 
  } 
  int flattening = NO_FLATTEN_DEPTH;
  if (oldStyleHtml)
    flattening = 0;
  
  HTMLFile hf(htmlDir, fname.c_str(), tit.c_str(), /*frameset*/ true); 
  hf.PrintNoBodyOrFrameset(); 
  hf.JSFileInclude(DETECTBS); 
  hf.JSFileInclude(GLOBAL); 
  hf.StopHead();  

  BriefAssertion(strcmp("PerfTable.ANY", table.Name()) == 0); 
  // the javascript code relies on the frame name to be PerfTable.ANY

  hf << "<frameset onLoad='detect_bs()'>" << endl
     << "  <frameset rows='50,70%,*'>" << endl
     << endl
     << "    <!-- Frameset 1: Frame: Title -->" << endl
     << "    <frame name='header' frameborder='1' scrolling='no'" << endl
     << "       src='" << header << ((debug) ? ".debug.html" : ".html")
     <<         "'>" << endl
     << "    <frameset cols='20%,*'>" << endl
     << endl
     << "      <!-- Frameset 2: Frame: Source File List -->" << endl    
     << "      <frame name='files' frameborder='1' src='files.html'>" << endl
     << "      <frameset rows='25,*,50'>" << endl
     << endl
     << "        <!-- Frameset 2: Frame: Source File Name -->" << endl
     << "        <frame name='srclabel' frameborder='1' scrolling='no'" << endl
     << "          title='name of the currently displayed source file'" << endl
     << "          src='" << HTMLSrcFiles::LabelFileName("") <<".html'>"<< endl
     << endl
     << "        <!-- Frameset 2: Frame: The Source File -->" << endl
     << "        <frame name='" << SOURCE << "' frameborder='1'" << endl
     << "          title='a source file' src='empty.html'>" << endl
     << endl
     << "        <!-- Frameset 2: Frame: Performance Data -->" << endl
     << "        <frame name='tablehead' frameborder='1'" << endl
     << "          scrolling='no'" << endl
     << "          title='header for the performance data table'" << endl
     << "          src='" << table.HeadFileName(perfIndex) << ".html'>" << endl
#if 0
     << "        <frame name='" << table.Name() << "' frameborder='1'" << endl
     << "          title='table of performance data'" << endl
     << "          src='" << table.TableFileName(perfIndex) << ".html'>"<< endl
#endif
     << "      </frameset>" << endl
     << "    </frameset>" << endl
     << "    <frameset cols='20%,*'> " << endl
     << "       <frameset rows='50,*'> " << endl
     << endl
     << "          <!-- Frameset 3: Frame: Ancestor/Current Control -->" << endl
     << "          <frame name='scopes_hdr1' frameborder='1'" << endl
     << "            title='performance info of scopes'" << endl
     << "            scrolling='no'" << endl
     << "            src='" << scopes.SelfHeadFileName() << ".html'>" << endl
     << endl
     << "          <!-- Frameset 3: Frame: Descendants Control -->" << endl
     << "          <frame name='scopes_hdr2' frameborder='1'" << endl
     << "            title='performance info of scopes'" << endl
     << "            scrolling=no" << endl
     << "            src='" << scopes.KidsHeadFileName() << ".html'>" << endl
     << "       </frameset>" << endl
     << "       <frameset rows='50,*'> " << endl
     << endl
     << "          <!-- Frameset 3: Frame: Ancestor/Current List -->" << endl
     << "          <frame name='" << HTMLScopes::SelfFrameName() << "'" << endl
     << "            frameborder='1'" << endl
     << "            title='performance info of scopes:: node and parent'" << endl
     << "            scrolling='no'" << endl
     << "            src='" << scopes.SelfFileName(pgmScope, perfIndex, 
                            flattening) << ".html'>" << endl
     << endl
     << "          <!-- Frameset 3: Frame: Descendants List -->" << endl
     << "          <frame name='" << HTMLScopes::KidsFrameName() << "'" << endl
     << "            frameborder='1'" << endl
     << "            title='performance info of scopes:: kids'" << endl
     << "            src='" << scopes.KidsFileName(pgmScope, perfIndex, 
                            flattening) << ".html'>" << endl
     << "       </frameset> " << endl
     << "    </frameset>" << endl
     << "  </frameset>" << endl
     << "  <noframes> " << endl
     << "    Sorry! HPCView requires a browser that can display" << endl
     << "    frames.  Please download a new browser." << endl
     << "  </noframes> " << endl
     << "</frameset>" << endl;
}

void 
HTMLDriver::WriteIndexFile(PgmScope *pgmScope, 
			   HTMLTable &table, int perfIndex, 
			   HTMLScopes &htmlScopes, 
			   const string& title, const char* header) const
{
  ::WriteIndexFile(htmlDir, pgmScope, table, perfIndex, 
		   htmlScopes, title, header, true, args.OldStyleHTML); 
  ::WriteIndexFile(htmlDir, pgmScope, table, perfIndex, 
		   htmlScopes, title, header, false, args.OldStyleHTML); 
}

static void
WriteHeader(const char* htmlDir, const char* headerFileName, 
	    const string& title, bool debug)  
{
  HTMLFile hf(htmlDir, headerFileName, "HPCView header"); 
  hf.JSFileInclude(DETECTBS); 
  hf.JSFileInclude(GLOBAL); 
  hf.SetBgColor("#a0a0a0"); 
  hf.StartBodyOrFrameset();
  hf << "<table width='100%'>" << endl
     << "<tr valign='top'>" << endl;

  // Left side
  hf << "<td align='left' width='30%'> "
     << "<span class='titleText'>" << endl;
  hf.JavascriptHref("reset_all_frames()", "Reset"); 
  hf << "&nbsp;&nbsp;" << endl;
  if (debug) {
    hf.JavascriptHref("restart('index.debug.html')", "Restart"); 
  } else {
    hf.JavascriptHref("restart('index.html')", "Restart"); 
  }
  hf << "</span></td>" << endl;

  // Middle: title
  hf << "<td align='center' width='40%'><h1> "
     << title << " </h1></td>" << endl;

  // Right side
  hf << "<td align='right' width='30%'>"
     << "<span class='titleText'>" << endl;
  hf.JavascriptHref("open_man()", "Help");
  if (debug) {
     hf << "&nbsp; &nbsp;" << endl;    
     hf.JavascriptHref("showTrace()", "[ShowTrace]"); 
     hf << "&nbsp; &nbsp;" << endl;
     hf.JavascriptHref("clearTrace()", "[ClearTrace]"); 
     hf << "&nbsp; &nbsp;" << endl;
     hf.JavascriptHref("setDebug(1);", "[ActivateTrace]"); 
     hf << "&nbsp; &nbsp;" << endl;
     hf.JavascriptHref("setDebug(0);", "[DeactivateTrace]"); 
     hf << "&nbsp; &nbsp;" << endl;
     hf.JavascriptHref("alert_globals();", "[Globals]"); 
  }
  hf << "</span></td>" << endl;
  
  hf << "</tr> </table>" << endl; 
}

void
HTMLDriver::WriteHeader(const char* headerFileName, 
			const string& title) const
{
  ::WriteHeader(htmlDir, headerFileName, title.c_str(), false); 
  string fnm = string(headerFileName) + ".debug";
  ::WriteHeader(htmlDir, fnm.c_str(), title.c_str(), true); 
}

void
WriteFileList(HTMLFile &hf, const char* header, 
	      ScopeInfoIterator& it,
	      bool hasSource) 
{ 
  // note: 'it' should iterate over files

  int count = 0; 
  
  // see whether there is at least one file 
  it.Reset(); 
  for (; it.CurScope(); it++) {
    FileScope *f = dynamic_cast<FileScope*>(it.CurScope()); 
    BriefAssertion(f != NULL); 
    if (f->HasSourceFile() == hasSource) 
    {
      count = 1; 
      break; 
    } 
  }
  
  if (count != 0) {
    hf << "<h4> "  << header << ": </h4>" << endl; 
    it.Reset(); 
    hf << "<pre>"; 
    string lastPath; 
    for (; it.CurScope(); it++) {
      FileScope *f = dynamic_cast<FileScope*>(it.CurScope()); 
      BriefAssertion(f != NULL); 
      if (f->HasSourceFile() == hasSource) 
      {
	if (hasSource) {
	  string curPath = PathComponent(it.CurScope()->Name());
	  if (curPath != lastPath) {
	    hf << curPath << ":" << endl; 
	    lastPath = curPath; 
	  } 
	} 
	hf << "  "; 
	hf.GotoSrcHref(f->BaseName().c_str(), 
		       HTMLDriver::UniqueName(f,NO_PERF_INDEX,
					      NO_FLATTEN_DEPTH)); 
	hf << endl; 
      }
    }
    hf << "</pre>" << endl; 
  }
}

void
HTMLDriver::WriteFiles(const char* filesFileName) const
{
  HTMLFile hf(htmlDir, filesFileName, "HPCView files");
  hf.SetBgColor("#a0a0a0");
  hf.StartBodyOrFrameset();
  hf << "<br>" << endl;

  // Iterate over load modules and print source files for each 
  ScopeInfoIterator it1(scopes.GetRoot(), &ScopeTypeFilter[ScopeInfo::LM]);
  //NameSortedChildIterator it1(scopes.Root());
  for (; it1.Current(); it1++) {
    LoadModScope* lm = dynamic_cast<LoadModScope*>(it1.Current());
    BriefAssertion(lm != NULL);
    hf << "<h3> "  << HTMLEscapeStr(lm->Name().c_str()) << "</h3>" << endl;

    hf << "<div class='indent'>" << endl;
    ScopeInfoIterator it2(lm, &ScopeTypeFilter[ScopeInfo::FILE]); 
    WriteFileList(hf, "Source Files", it2, true);
    WriteFileList(hf, "Other Files", it2, false);
    hf << "</div>" << endl;
  }   
}

