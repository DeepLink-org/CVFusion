//===-- elena/src/codegen/X86Codegen.cpp
// - Code generate for x86 code -------*- C++ -*-===//
//
// Part of the Elena Project.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration and implementation of the X86Codegen
/// and the TopologySorter class, which is used to generate x86 code.
///
//===----------------------------------------------------------------------===//

#include "CodeGen/DeviceCodegen.h"
#include "CodeGen/TextGen.h"
#include "IR/VisitorBase.h"
#include "api.h"

using namespace ir;  // NOLINT

class X86Codegen;

///
/// \brief Sort the topology of tensor var
class TopologySorter : public VisitorBase<TopologySorter> {
 public:
  explicit TopologySorter(X86Codegen *codegen) : codegen{codegen} {}

  void visit(TensorVar *);

  ///
  /// \brief Mark the current tensor var as be visited
  /// \param p
  void markVisited(TensorVar *p) { markVisitedByName(p->get_name()); }

  ///
  /// \brief Mark the current tensor var as be visited by name
  /// \param p
  void markVisitedByName(std::string n) { well_defined.insert(std::move(n)); }

  using VisitorBase::visit;

 private:
  std::set<std::string> well_defined;
  X86Codegen *codegen;
};

///
/// \brief Generate x86 code
class X86Codegen final : public VisitorBase<X86Codegen>, public TextGen {
 public:
  explicit X86Codegen(std::ostream &ostr) : TextGen(ostr), sorter(this) {}

  void visit(Allocate *);
  void visit(Provide *);
  void visit(For *);
  void visit(Store *);
  void visit(Attr *);
  void visit(TensorVar *);
  void visit(Binary *);
  void visit(Unary *);
  void visit(IterVar *);
  void visit(ScalarVar *);
  void visit(Let *);
  void visit(IfThenElse *);
  void visit(Logical *);
  void visit(Select *);
  void visit(Call *);
  void visit(Evaluate *);

  template <typename T>
  void visit(Const<T> *);

  // for raw code gen
  void visit(ComputeOp *);
  void visit(Reduce *);

  using VisitorBase::visit;

  TopologySorter sorter;
};

void X86Codegen::visit(Allocate *allocate_ptr) {
  CHECK_NODE_TYPE(allocate_ptr->var, TensorVar)
  auto tensor_ptr = ptr_cast<TensorVar>(allocate_ptr->var);
  sorter.markVisited(tensor_ptr.get());
  *this << TYPE_OF(tensor_ptr) << " ";
  visit(tensor_ptr);

  for (const auto &rg : allocate_ptr->bound->element) {
    *this << "[";
    visit(rg->extent);
    *this << "]";
  }

  *this << ";" << endl;

  visit(allocate_ptr->body);
}

void X86Codegen::visit(Provide *provide_ptr) {
  visit(provide_ptr->var);
  visit(provide_ptr->index);
  *this << " = ";
  visit(provide_ptr->value);
}

void X86Codegen::visit(For *for_stmt_ptr) {
  if (for_stmt_ptr->it->iter_type == ir::IterAttrType::Unrolled) {
    *this << "#pragma unroll" << endl;
  }
  *this << "for (" << TYPE_OF(for_stmt_ptr->it) << " ";
  visit(for_stmt_ptr->it);
  *this << " = ";
  visit(for_stmt_ptr->init);
  *this << "; ";
  visit(for_stmt_ptr->it);
  *this << " < ";
  visit(for_stmt_ptr->init);
  *this << " + ";
  visit(for_stmt_ptr->extent);
  *this << "; ++";
  visit(for_stmt_ptr->it);
  *this << ") " << block_begin;
  visit(for_stmt_ptr->body);
  *this << block_end;
}

void X86Codegen::visit(Store *store_ptr) {
  visit(store_ptr->var);
  for (const auto &index : store_ptr->index->element) {
    *this << "[";
    visit(index);
    *this << "]";
  }
  *this << " = ";
  visit(store_ptr->value);
  *this << ";" << endl;
}

void X86Codegen::visit(TensorVar *tensor_ptr) {
  *this << makeIdentifier(tensor_ptr->get_name());
}

void X86Codegen::visit(IterVar *iter_ptr) {
  // DISCUSS: Unlike in CUDA, all variable names are fed into
  // makeIdentifier, should some special variables be preserved?
  *this << makeIdentifier(iter_ptr->get_name());
}

void X86Codegen::visit(ScalarVar *scalar_ptr) {
  if (!scalar_ptr->is_placeholder()) {
    visit(scalar_ptr->tensor);
    for (const auto &index : scalar_ptr->indices->element) {
      *this << "[";
      visit(index);
      *this << "]";
    }
  } else {
    *this << scalar_ptr->get_name();
  }
}

void X86Codegen::visit(Attr *attr_ptr) {
  // expand a threadIdx/blockIdx into a loop
  if (attr_ptr->key == AttrType::ThreadExtent) {
    const auto iter = ptr_cast<Expr>(attr_ptr->node);
    *this << "for (" << TYPE_OF(iter) << " ";
    visit(iter);
    *this << " = 0; ";
    visit(iter);
    *this << " < ";
    visit(attr_ptr->value);
    *this << "; ++";
    visit(iter);
    *this << ")" << block_begin;
  }
  visit(attr_ptr->body);
  if (attr_ptr->key == AttrType::ThreadExtent) {
    *this << block_end;
  }
}

void X86Codegen::visit(Logical *logical_ptr) {
  *this << "(";
  visit(logical_ptr->lhs);
  *this << " " << LOGICALTYPE_SYMBOL(logical_ptr->operation_type) << " ";
  visit(logical_ptr->rhs);
  *this << ")";
}

void X86Codegen::visit(Unary *unary_ptr) {
  const auto type = unary_ptr->get_dtype();
  if (type == ir::ScalarType::Float32 &&
      unary_ptr->operation_type == UnaryType::Abs) {
    *this << "(fabs(";
  } else {
    *this << "(" << UOP_DEVICE_NAME(unary_ptr->operation_type) << "(";
  }
  visit(unary_ptr->operand);
  *this << "))";
}

void X86Codegen::visit(Binary *binary_ptr) {
  if (binary_ptr->operation_type == BinaryType::Max) {
    *this << "max(";
    visit(binary_ptr->lhs);
    *this << ", ";
    visit(binary_ptr->rhs);
    *this << ")";
  } else if (binary_ptr->operation_type == BinaryType::Min) {
    *this << "min(";
    visit(binary_ptr->lhs);
    *this << ", ";
    visit(binary_ptr->rhs);
    *this << ")";
  } else {
    *this << "(";
    visit(binary_ptr->lhs);
    *this << " " << BINARYTYPE_SYMBOL(binary_ptr->operation_type) << " ";
    visit(binary_ptr->rhs);
    *this << ")";
  }
}

template <typename T>
void X86Codegen::visit(Const<T> *const_ptr) {
  *this << std::boolalpha << const_ptr->get_value();
}

void X86Codegen::visit(Let *let_ptr) {
  // DISCUSS: see cuda_codegen::visit_Let for detail
  // *this << block_begin;
  *this << "const " << TYPE_OF(let_ptr->var) << " ";
  visit(let_ptr->var);
  *this << " = ";
  visit(let_ptr->value);
  *this << ";" << endl;
  visit(let_ptr->body);
  // *this << block_end;
}

void X86Codegen::visit(IfThenElse *if_then_else_ptr) {
  *this << "if (";
  visit(if_then_else_ptr->condition);
  *this << ") " << block_begin;
  visit(if_then_else_ptr->then_case);
  *this << block_end;
  if (if_then_else_ptr->else_case) {
    *this << " else " << block_begin;
    visit(if_then_else_ptr->else_case);
    *this << block_end;
  }
}

void X86Codegen::visit(Select *select_ptr) {
  *this << "(";
  visit(select_ptr->cond);
  *this << " ? ";
  visit(select_ptr->tBranch);
  *this << " : ";
  visit(select_ptr->fBranch);
  *this << ")";
}

void X86Codegen::visit(Call *call_ptr) {
  if (call_ptr->func == CallFunction::Sync) {
    // DISCUSS: Seems that no 'sync' is needed
    *this << "/* sync() */";
  } else {
    throw std::runtime_error(
        "Calling function other than 'Sync' and 'Select' is not supported!");
  }
}

void X86Codegen::visit(Evaluate *evaluate_ptr) {
  visit(evaluate_ptr->value);
  *this << ";" << endl;
}

void X86Codegen::visit(ComputeOp *compute) {
  for (auto &i : compute->iter_vars->element) {
    *this << "for (" << TYPE_OF(i) << " ";
    visit(i);
    *this << " = ";
    visit(i->range->init);
    *this << "; ";
    visit(i);
    *this << " < ";
    visit(i->range->extent);
    *this << "; ++";
    visit(i);
    *this << ") " << block_begin;
  }
  auto print_assignment_head = [&] {
    visit(compute->output(0));
    for (auto &i : compute->iter_vars->element) {
      *this << '[';
      visit(i);
      *this << ']';
    }
    *this << " = ";
  };
  if (compute->fcompute->get_type() == IRNodeType::Reduce) {
    auto reduce = static_cast<Reduce *>(compute->fcompute.get());
    visit(reduce);
    print_assignment_head();
    visit(reduce->accumulate);
    *this << ";" << endl;
  } else {
    print_assignment_head();
    visit(compute->fcompute);
    *this << ";" << endl;
  }
  for (auto &i : compute->iter_vars->element) {
    *this << block_end;
    // 'i' is deliberately not used.
    static_cast<void>(i);
  }
}

void X86Codegen::visit(Reduce *reduce) {
  *this << TYPE_OF(reduce->accumulate) << " ";
  visit(reduce->accumulate);
  *this << " = ";
  visit(reduce->init);
  *this << ";" << endl;
  for (auto &i : reduce->reduce_axis->element) {
    *this << "for (" << TYPE_OF(i) << " ";
    visit(i);
    *this << " = ";
    visit(i->range->init);
    *this << "; ";
    visit(i);
    *this << " < ";
    visit(i->range->extent);
    *this << "; ++";
    visit(i);
    *this << ") " << block_begin;
  }
  visit(reduce->accumulate);
  *this << " = ";
  visit(reduce->combiner);
  *this << ";" << endl;
  for (auto &i : reduce->reduce_axis->element) {
    *this << block_end;
    // 'i' is deliberately not used.
    static_cast<void>(i);
  }
}
