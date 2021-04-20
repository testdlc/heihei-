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
// Copyright ((c)) 2002-2021, Rice University
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

#ifdef ENABLE_IGC

//******************************************************************************
// system includes
//******************************************************************************

#include <iostream>
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <libelf.h>

#include <Symtab.h>
#include <CodeSource.h>
#include <CodeObject.h>

#include <iga/kv.hpp>



//******************************************************************************
// local includes
//******************************************************************************

#include <lib/binutils/ElfHelper.hpp>
#include <lib/support/diagnostics.h>

#include "DotCFG.hpp"
#include "GPUCFGFactory.hpp"
#include "GPUFunction.hpp"
#include "GPUBlock.hpp"
#include "GPUCodeSource.hpp"
#include "ReadIntelCFG.hpp"

//******************************************************************************
// macros
//******************************************************************************

#define DEBUG 0

#define MAX_STR_SIZE 1024
#define INTEL_GPU_DEBUG_SECTION_NAME "Intel(R) OpenCL Device Debug"

using namespace Dyninst;
using namespace ParseAPI;
using namespace SymtabAPI;
using namespace InstructionAPI;


static std::string
getOpString(iga::Op op) {
  switch (op)
  {
    case iga::Op::ADD:      return "ADD";
    case iga::Op::ADDC:     return "ADDC";
    case iga::Op::AND:      return "AND";
    case iga::Op::ASR:      return "ASR";
    case iga::Op::AVG:      return "AVG";
    case iga::Op::BFE:      return "BFE";
    case iga::Op::BFI1:     return "BFI1";
    case iga::Op::BFI2:     return "BFI2";
    case iga::Op::BFREV:    return "BFREV";
    case iga::Op::BRC:      return "BRC";
    case iga::Op::BRD:      return "BRD";
    case iga::Op::BREAK:    return "BREAK";
    case iga::Op::CALL:     return "CALL";
    case iga::Op::CALLA:    return "CALLA";
    case iga::Op::CBIT:     return "CBIT";
    case iga::Op::CMP:      return "CMP";
    case iga::Op::CMPN:     return "CMPN";
    case iga::Op::CONT:     return "CONT";
    case iga::Op::CSEL:     return "CSEL";
    case iga::Op::DIM:      return "DIM";
    case iga::Op::DP2:      return "DP2";
    case iga::Op::DP3:      return "DP3";
    case iga::Op::DP4:      return "DP4";
    case iga::Op::DP4A:     return "DP4A";
    case iga::Op::DPH:      return "DPH";
    case iga::Op::ELSE:     return "ELSE";
    case iga::Op::ENDIF:    return "ENDIF";
    case iga::Op::F16TO32:  return "F16TO32";
    case iga::Op::F32TO16:  return "F32TO16";
    case iga::Op::FBH:      return "FBH";
    case iga::Op::FBL:      return "FBL";
    case iga::Op::FRC:      return "FRC";
    case iga::Op::GOTO:     return "GOTO";
    case iga::Op::HALT:     return "HALT";
    case iga::Op::IF:       return "IF";
    case iga::Op::ILLEGAL:  return "ILLEGAL";
    case iga::Op::JMPI:     return "JMPI";
    case iga::Op::JOIN:     return "JOIN";
    case iga::Op::LINE:     return "LINE";
    case iga::Op::LRP:      return "LRP";
    case iga::Op::LZD:      return "LZD";
    case iga::Op::MAC:      return "MAC";
    case iga::Op::MACH:     return "MACH";
    case iga::Op::MAD:      return "MAD";
    case iga::Op::MADM:     return "MADM";
    case iga::Op::MATH:     return "MATH";
    case iga::Op::MOV:      return "MOV";
    case iga::Op::MOVI:     return "MOVI";
    case iga::Op::MUL:      return "MUL";
    case iga::Op::NOP:      return "NOP";
    case iga::Op::NOT:      return "NOT";
    case iga::Op::OR:       return "OR";
    case iga::Op::PLN:      return "PLN";
    case iga::Op::RET:      return "RET";
    case iga::Op::RNDD:     return "RNDD";
    case iga::Op::RNDE:     return "RNDE";
    case iga::Op::RNDU:     return "RNDU";
    case iga::Op::RNDZ:     return "RNDZ";
    case iga::Op::ROL:      return "ROL";
    case iga::Op::ROR:      return "ROR";
    case iga::Op::SAD2:     return "SAD2";
    case iga::Op::SADA2:    return "SADA2";
    case iga::Op::SEL:      return "SEL";
    case iga::Op::SEND:     return "SEND";
    case iga::Op::SENDC:    return "SENDC";
    case iga::Op::SENDS:    return "SENDS";
    case iga::Op::SENDSC:   return "SENDSC";
    case iga::Op::SHL:      return "SHL";
    case iga::Op::SHR:      return "SHR";
    case iga::Op::SMOV:     return "SMOV";
    case iga::Op::SUBB:     return "SUBB";
    case iga::Op::SYNC:     return "SYNC";
    case iga::Op::WAIT:     return "WAIT";
    case iga::Op::WHILE:    return "WHILE";
    case iga::Op::XOR:      return "XOR";
    default:                return "INVALID";
  }
}

#if DEBUG
static std::string getKindString(iga::Kind kind) {
  switch (kind)
  {
    case iga::Kind::INVALID:    return "INVALID";   // an invalid or uninitialized operand
    case iga::Kind::DIRECT:     return "DIRECT";    // direct register reference
    case iga::Kind::MACRO:      return "MACRO";     // madm or math.invm or math.rsqrtm
    case iga::Kind::INDIRECT:   return "INDIRECT";  // register-indriect access
    case iga::Kind::IMMEDIATE:  return "IMMEDIATE"; // immediate value
    case iga::Kind::LABEL:      return "LABEL";     // block target (can be numeric label/i.e. imm value)
    default:                    return "UNKNOWN";
  }
}


static std::string getRegNameString(iga::RegName reg)
{
  switch (reg)
  {
    case iga::RegName::GRF_R:       return  "GRF"; 
    case iga::RegName::ARF_NULL:    return  "AREG_NULL"; 
    case iga::RegName::ARF_A:       return  "AREG_A";
    case iga::RegName::ARF_ACC:     return  "AREG_ACC";  
    case iga::RegName::ARF_CE:      return  "AREG_MASK0";
    case iga::RegName::ARF_MSG:     return  "AREG_MSG";
    case iga::RegName::ARF_DBG:     return  "AREG_DBG"; 
    case iga::RegName::ARF_SR:      return  "AREG_SR0"; 
    case iga::RegName::ARF_CR:      return  "AREG_CR0"; 
    case iga::RegName::ARF_N:       return  "AREG_N"; 
    case iga::RegName::ARF_IP:      return  "AREG_IP";   
    case iga::RegName::ARF_F:       return  "AREG_F";  
    case iga::RegName::ARF_TM:      return  "AREG_TM0";  
    case iga::RegName::ARF_TDR:     return  "AREG_TDR0"; 
    case iga::RegName::ARF_SP:      return  "AREG_SP";
    case iga::RegName::ARF_MME:     return  "AREG_MME";
    case iga::RegName::ARF_FC:      return  "AREG_FC";
    default: //iga::RegName::INVALID
                                    //assert(false && "illegal ARF");
                                    return "REG_INVALID";
  }
}


static std::string getIGATypeString(iga::Type type)
{
    switch (type)
    {
    case iga::Type::UB:   return "Type_UB";
    case iga::Type::B:    return "Type_B"; 
    case iga::Type::UW:   return "Type_UW";
    case iga::Type::W:    return "Type_W";
    case iga::Type::UD:   return "Type_UD";
    case iga::Type::D:    return "Type_D";
    case iga::Type::UQ:   return "Type_UQ";
    case iga::Type::Q:    return "Type_Q";
    case iga::Type::HF:   return "Type_HF";
    case iga::Type::F:    return "Type_F"; 
    case iga::Type::DF:   return "Type_DF";
    case iga::Type::UV:   return "Type_UV";
    case iga::Type::V:    return "Type_V";
    case iga::Type::VF:   return "Type_VF";
    case iga::Type::NF:   return "Type_NF";
    default: //iga::Type::INVALID
        //assert(false && "illegal type");
        return "Type_INVALID";
    }
}


static std::string getIGAPredCtrlString(iga::PredCtrl predCtrl)
{
    switch (predCtrl)
    {
      case iga::PredCtrl::SEQ:        return "PRED_DEFAULT";
      case iga::PredCtrl::ANY2H:      return "PRED_ANY2H";        
      case iga::PredCtrl::ANY4H:      return "PRED_ANY4H";
      case iga::PredCtrl::ANY8H:      return "PRED_ANY8H";       
      case iga::PredCtrl::ANY16H:     return "PRED_ANY16H";       
      case iga::PredCtrl::ANY32H:     return "PRED_ANY32H";       
      case iga::PredCtrl::ALL2H:      return "PRED_ALL2H";        
      case iga::PredCtrl::ALL4H:      return "PRED_ALL4H";        
      case iga::PredCtrl::ALL8H:      return "PRED_ALL8H";       
      case iga::PredCtrl::ALL16H:     return "PRED_ALL16H";       
      case iga::PredCtrl::ALL32H:     return "PRED_ALL32H";       
      case iga::PredCtrl::ANYV:       return "PRED_ANYV";         
      case iga::PredCtrl::ALLV:       return "PRED_ALLV";         
      default:   //iga::PredCtrl::NONE;
        //assert(false && "illegal predicate control");
        return "PRED_NONE";
    }
}
#endif


static int
getElementSize
(
 iga::Type dataType 
)
{
  // values for NF and INVALID not added
  if (dataType == iga::Type::UB || dataType == iga::Type::B) {
    return 1;
  } else if (dataType == iga::Type::UW || dataType == iga::Type::W) {
    return 2;
  } else if (dataType == iga::Type::UD || dataType == iga::Type::D || dataType == iga::Type::UV ||
      dataType == iga::Type::V || dataType == iga::Type::VF || dataType == iga::Type::F) {
    return 4;
  } else if (dataType == iga::Type::UQ || dataType == iga::Type::Q || dataType == iga::Type::DF) {
    return 8;
  } else {
    return 0;
  }
}


void 
addCustomFunctionObject
(
 const std::string &func_obj_name,
 Symtab *symtab
)
{
  Region *reg = NULL;
  bool status = symtab->findRegion(reg, ".text");
  assert(status == true);

  unsigned long reg_size = reg->getMemSize();
  Symbol *custom_symbol = new Symbol(
      func_obj_name, 
      SymtabAPI::Symbol::ST_FUNCTION, // SymbolType
      Symbol::SL_LOCAL, //SymbolLinkage
      SymtabAPI::Symbol::SV_DEFAULT, //SymbolVisibility
      0, //Offset,
      NULL, //Module *module 
      reg, //Region *r
      reg_size, //unsigned s
      false, //bool d
      false, //bool a
      -1, //int index
      -1, //int strindex
      false //bool cs
  );

  //adding the custom symbol into the symtab object
  status = symtab->addSymbol(custom_symbol); //(Symbol *newsym)
  assert(status == true);
}


static GPUParse::InstructionStat*
getIntelInstructionStat
(
 const KernelView &kv,
 int offset
)
{
  char inst_asm_text[MAX_STR_SIZE] = { 0 };
  size_t length = kv.getInstSyntax(offset, inst_asm_text, MAX_STR_SIZE);
  assert(length > 0);
  iga::Op opcode = kv.getOpcode(offset);
  std::string op = getOpString(opcode);

  int execSize = (int)kv.getExecutionSize(offset); // returns iga::ExecSize
  int32_t noSrcReg = kv.getNumberOfSources(offset);

#if DEBUG
  std::cout << "offset: " << offset << ". asm: " << inst_asm_text << std::endl;
  std::cout << "\nopcode:" << op << "\n";
  std::cout << "\nnumber of source registers: " << noSrcReg;
#endif

  std::vector<int> srcs;
  for (int i = 0;   i < noSrcReg; i++) {
    int32_t srcRegNo = kv.getSrcRegNumber(offset, i);
    int32_t srcSubRegNo = kv.getSrcSubRegNumber(offset, i);
    iga::Type srcDataType = kv.getSrcDataType(offset, i);
    iga::RegName srcRegType = kv.getSrcRegType(offset, i);

    if (srcRegType != iga::RegName::GRF_R) {
      continue;
    }
    uint32_t vertStride, width, horzStride;
    // Returns 0 if any of instruction's src operand region components
    // (Src RgnVt, RgnWi, RgnHz) are succesfully determined.
    // Otherwise returns -1.
    int32_t status = kv.getSrcRegion(offset, i, &vertStride, &width, &horzStride);
    assert(status == 0);

#if DEBUG
    iga::Kind srcRegKind = kv.getSrcRegKind(offset, i);
    std::cout << "\nSrcreg no: " << i << "\n  register: " << srcRegNo << ", subregister: " << srcSubRegNo
      << ", srcDataType: " << getIGATypeString(srcDataType)
      << ", srcRegType: " << getRegNameString(srcRegType)
      << ", srcRegKind: " << getKindString(srcRegKind)
      << ", execSize: " << execSize
      << "\n  stride fetch status: " << status << ", vert.stride: " << vertStride << ", width: "  << width << ", hor.stride: " << horzStride << std::endl;
#endif

    int elementSize = getElementSize(srcDataType);
    int height = execSize / width;
    int channel = 0;
    int base1 = (srcRegNo << 5) + srcSubRegNo * elementSize;
    std::vector<int> childSrc(execSize * elementSize);
    for (int x=0; x < height; x++) {
      int base2 = base1;
      for (uint32_t y=0; y < width; y++) {
        int addr_y = base2;
        for (int z=0;z<elementSize;z++) {
          childSrc[channel++] = addr_y + z;
        }
        base2 += horzStride*elementSize;
      }
      base1 +=vertStride*elementSize;
    }
    srcs.insert(srcs.end(), childSrc.begin(), childSrc.end());
  }

  int32_t dstRegNo = kv.getDstRegNumber(offset);
  int32_t dstSubRegNo = kv.getDstSubRegNumber(offset);
  iga::Type dstDataType = kv.getDstDataType(offset);
  iga::RegName dstRegType = kv.getDstRegType(offset);
  std::vector<int> dsts;
  if (dstRegType == iga::RegName::GRF_R) {
    uint32_t horzStride;
    // Returns 0 if instruction's destination operand horizontal stride
    // (DstRgnHz) is succesfully returned.
    // Otherwise returns -1.
    int32_t status = kv.getDstRegion(offset, &horzStride);
    assert(status == 0);
    int elementSize = getElementSize(dstDataType);

    dsts.resize(execSize * elementSize);
    int channel = 0;
    int base1 = (dstRegNo << 5) + dstSubRegNo * elementSize;
    for (int x=0; x < execSize; x++) {
      int addr_x = base1;
      for (int y=0;y<elementSize;y++) {
        dsts[channel++] = addr_x + y;
      }
      base1 += (horzStride * elementSize);
    }

#if DEBUG
  iga::Kind dstRegKind = kv.getDstRegKind(offset);
    std::cout << "\ndstRegNo: " << dstRegNo << ", subregister: " << dstSubRegNo
      << ", DataType: " << getIGATypeString(dstDataType)
      << ", RegType: " << getRegNameString(dstRegType)
      << ", RegKind: " << getKindString(dstRegKind)
      << ", execSize: " << execSize << std::endl;
#endif
  } else {
    // To be considered: How to deal with writes to ARF registers?
  }

  // barriers are executed using send instruction. example:
  // [324] (W)      send (1|M0)              null     r22     0x3         0x2000004  //  wr:1+?, rd:0,  barrier 
  // one can only have memory coherency and synchronization inside a 'work group' or 'thread group', depending on the nomenclature.
  // This instruction makes this thread wait until all threads in its group have entered the barrier.
  // AFAIK, all threads in a work-group share a single barrier
  // unlike CUDA, there is a single barrier register in intel instructions.
  // So is fine to send a vector of 1 entry when an instruction is a barrier and 
  // empty vector when the instruction is not a barrier?

  // commenting this section since synchronization is not factored in backward slicing 
#if 0
  bool instContainsBarrier = false;
  if (opcode == iga::Op::SEND || opcode == iga::Op::SENDC || opcode == iga::Op::SENDS || opcode == iga::Op::SENDSC) {
    char *output = strstr (inst_asm_text, "barrier");
    if (output) {
      instContainsBarrier = true;
    }
  }
#endif

  // intel instructions follow SIMD model.
  // when predication is on for an instruction, flag register is used to check which SIMD lanes should be used by the instruction
  // predication does not switch on/off an instruction, just some lanes
  // There are 3 more registers that affect lanes used (these registers affect all instructions, not just predicated inst.)
  // CE, ExecMask and DMask
  iga::PredCtrl pred = kv.getPredicate(offset);
  bool invPred = kv.isInversePredicate(offset);
  // int32_t flagReg = kv.getFlagReg(offset);
  // int32_t flagSubReg = kv.getFlagSubReg(offset);
  
  GPUParse::InstructionStat::PredicateFlag predFlag;
  if (pred == iga::PredCtrl::NONE) {
    predFlag = GPUParse::InstructionStat::PredicateFlag::PREDICATE_NONE;
  } else if (invPred) {
    predFlag = GPUParse::InstructionStat::PredicateFlag::PREDICATE_FALSE;
  } else {
    predFlag = GPUParse::InstructionStat::PredicateFlag::PREDICATE_TRUE;
  }

#if DEBUG
  std::cout << "\npred: " << getIGAPredCtrlString(pred) << ",invPred: " << invPred
    << ", flag register: " << flagReg << ", flag subregister: " << flagSubReg; 
#endif

  auto *inst_stat = new GPUParse::InstructionStat(op, offset, predFlag, dsts, srcs);
    //new InstructionStat(op, pc, pred, barrier_threshold, indirect, pred_flag, pred_assign_pcs, dsts, srcs,
    //    pdsts, psrcs, bdsts, bsrcs, udsts, usrcs, updsts, upsrcs, assign_pcs, passign_pcs,
    //    bassign_pcs, uassign_pcs, upassign_pcs, control);
  return inst_stat;
}


void
parseIntelCFG
(
 char *text_section,
 int text_section_size,
 GPUParse::Function &function
)
{
  KernelView kv(IGA_GEN9, text_section, text_section_size);
  std::map<int, GPUParse::Block *> block_offset_map;

  int offset = 0;
  int block_id = 0;

  // Construct basic blocks
  while (offset < text_section_size) {
    auto *block = new GPUParse::Block(block_id, offset, function.name + "_" + std::to_string(block_id)); 
    block_id++;

    function.blocks.push_back(block);
    block_offset_map[offset] = block;

    auto size = kv.getInstSize(offset);
    auto *inst_stat = getIntelInstructionStat(kv, offset);
    auto *inst = new GPUParse::IntelInst(offset, size, inst_stat);
    block->insts.push_back(inst);

    while (!kv.isInstTarget(offset + size) && (offset + size < text_section_size)) {
      offset += size;  
      size = kv.getInstSize(offset);
      if (size == 0) {
        // this is a weird edge case, what to do?
        break;
      }

      char inst_asm_text[MAX_STR_SIZE] = { 0 };
      size_t length;
      int32_t size = kv.getInstSize(offset);
      if (size == 0) {
        return;
      }

      length = kv.getInstSyntax(offset, inst_asm_text, MAX_STR_SIZE);
      assert(length > 0);
      auto *inst_stat = getIntelInstructionStat(kv, offset);
      inst = new GPUParse::IntelInst(offset, size, inst_stat);
      block->insts.push_back(inst);
    }

    if (kv.getOpcode(offset) == iga::Op::CALL || kv.getOpcode(offset) == iga::Op::CALLA) {
      inst->is_call = true;
    }
    offset += size;
  }
  
  using TargetType = Dyninst::ParseAPI::EdgeTypeEnum;

  // Construct targets
  std::array<int, KV_MAX_TARGETS_PER_INSTRUCTION + 1> jump_targets;
  for (size_t i = 0; i < function.blocks.size(); ++i) {
    auto *block = function.blocks[i];
    auto *inst = block->insts.back();
    size_t jump_targets_count = kv.getInstTargets(inst->offset, jump_targets.data());

    if (i != function.blocks.size() - 1) {
      // Add a fall through edge
      // The last block and the end of thread (EOT) block do not have a fall through
      int next_block_start_offset = function.blocks[i + 1]->insts.front()->offset;

      bool eot_inst = kv.getOpcodeGroup(inst->offset) == KV_OPGROUP_SEND_EOT;
      bool pred_inst = kv.getPredicate(inst->offset) != iga::PredCtrl::NONE;
      bool join_inst = kv.getOpcode(inst->offset) == iga::Op::JOIN;
      if ((pred_inst || jump_targets_count == 0) && !eot_inst) {
        jump_targets[jump_targets_count] = next_block_start_offset;
        jump_targets_count += 1;
      } else if (join_inst) {
        // Join is not a branch
        jump_targets[jump_targets_count - 1] = next_block_start_offset;
      }
    }

    for (size_t j = 0; j < jump_targets_count; j++) {
      auto *target_block = block_offset_map.at(jump_targets[j]);
      
      TargetType type = TargetType::COND_TAKEN;
      if (inst->is_call) {
        // XXX(Keren): since we parse each instruction individually,
        // we only see CALL_FT edges within a function
        type = TargetType::CALL_FT;
      } else if (target_block->insts.front()->offset == inst->offset + inst->size) {
        // Fallthrough
        type = TargetType::DIRECT;
      }

      // Jump
      bool added = false;
      for (auto *target : block->targets) {
        if (target->block == target_block) {
          added = true;
        }
      }
      if (!added) {
        block->targets.push_back(new GPUParse::Target(inst, target_block, type));
      }
    }
  }

  if (DEBUG) {
    // Instruction buffer
    char inst_str[MAX_STR_SIZE];

    for (auto *block : function.blocks) {
      std::cout << std::hex;
      std::cout << block->name << ": [" << block->insts.front()->offset << ", " << block->insts.back()->offset << "]" << std::endl;

      for (auto *inst : block->insts) {
        size_t n = kv.getInstSyntax(inst->offset, NULL, 0);
        assert(n < MAX_STR_SIZE);

        inst_str[n] = '\0';
        auto fmt_opts = IGA_FORMATTING_OPTS_DEFAULT; // see iga.h
        kv.getInstSyntax(inst->offset, inst_str, n, fmt_opts);

        std::cout << std::hex << inst->offset << std::dec << inst_str << std::endl;
      }

      for (auto *target : block->targets) {
        std::cout << "\t" << block->name << "->" << target->block->name << std::endl;
      }
      std::cout << std::dec;
    }
    std::cout << std::dec;
  }
}


bool
readIntelCFG
(
 const std::string &search_path,
 ElfFile *elfFile,
 Dyninst::SymtabAPI::Symtab *the_symtab, 
 bool cfg_wanted,
 Dyninst::ParseAPI::CodeSource **code_src, 
 Dyninst::ParseAPI::CodeObject **code_obj
)
{
  // An Intel GPU binary for a kernel does not contain a function symbol for the kernel
  // in its symbol table. Without a function symbol in the symbol table, Dyninst will not
  // associate line map entries with addresses in the kernel. To cope with this defect of
  // binaries for Intel GPU kernels, we add a function symbol for the kernel to its Dyninst
  // symbol table.	
  auto function_name = elfFile->getGPUKernelName();
  addCustomFunctionObject(function_name, the_symtab); //adds a dummy function object

  if (cfg_wanted) {
    char *text_section = NULL;
    auto text_section_size = elfFile->getTextSection(&text_section);
    if (text_section_size == 0) {
      *code_src = new SymtabCodeSource(the_symtab);
      *code_obj = new CodeObject(*code_src, NULL, NULL, false, true);

      return false;
    }

    GPUParse::Function function(0, function_name);
    parseIntelCFG(text_section, text_section_size, function);
    std::vector<GPUParse::Function *> functions = {&function};

    CFGFactory *cfg_fact = new GPUCFGFactory(functions);
    *code_src = new GPUCodeSource(functions, the_symtab); 
    *code_obj = new CodeObject(*code_src, cfg_fact);
    (*code_obj)->parse();

    return true;
  }

  *code_src = new SymtabCodeSource(the_symtab);
  *code_obj = new CodeObject(*code_src, NULL, NULL, false, true);

  return false;
}

#endif // ENABLE_IGC
