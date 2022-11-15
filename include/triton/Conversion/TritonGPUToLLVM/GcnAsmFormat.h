#ifndef TRITON_CONVERSION_TRITON_GPU_TO_LLVM_GCN_FORMAT_H_
#define TRITON_CONVERSION_TRITON_GPU_TO_LLVM_GCN_FORMAT_H_

#include "mlir/IR/Value.h"
#include "triton/Conversion/TritonGPUToLLVM/AsmFormat.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <string>

namespace mlir {
class ConversionPatternRewriter;
class Location;

namespace triton {
using llvm::StringRef;

class GCNInstr;
class GCNInstrCommon;
class GCNInstrExecution;

struct GCNBuilder {
  struct Operand {
    std::string constraint;
    Value value;
    int idx{-1};
    llvm::SmallVector<Operand *> list;
    std::function<std::string(int idx)> repr;

    // for list
    Operand() = default;
    Operand(const Operation &) = delete;
    Operand(Value value, StringRef constraint)
        : value(value), constraint(constraint) {}

    bool isList() const { return !value && constraint.empty(); }

    Operand *listAppend(Operand *arg) {
      list.push_back(arg);
      return this;
    }

    Operand *listGet(size_t nth) const {
      assert(nth < list.size());
      return list[nth];
    }

    std::string dump() const;
  };

  template <typename INSTR = GCNInstr, typename... Args>
  INSTR *create(Args &&...args) {
    instrs.emplace_back(std::make_unique<INSTR>(this, args...));
    return static_cast<INSTR *>(instrs.back().get());
  }

  // Create a list of operands.
  Operand *newListOperand() { return newOperand(); }

  Operand *newListOperand(ArrayRef<std::pair<mlir::Value, std::string>> items) {
    auto *list = newOperand();
    for (auto &item : items) {
      list->listAppend(newOperand(item.first, item.second));
    }
    return list;
  }

  Operand *newListOperand(unsigned count, mlir::Value val,
                          const std::string &constraint) {
    auto *list = newOperand();
    for (int i = 0; i < count; ++i) {
      list->listAppend(newOperand(val, constraint));
    }
    return list;
  }

  Operand *newListOperand(unsigned count, const std::string &constraint) {
    auto *list = newOperand();
    for (int i = 0; i < count; ++i) {
      list->listAppend(newOperand(constraint));
    }
    return list;
  }

  // Create a new operand. It will not add to operand list.
  // @value: the MLIR value bind to this operand.
  // @constraint: ASM operand constraint, .e.g. "=r"
  // @formatter: extra format to represent this operand in ASM code, default is
  //             "%{0}".format(operand.idx).
  Operand *newOperand(mlir::Value value, StringRef constraint,
                      std::function<std::string(int idx)> formatter = nullptr);

  // Create a new operand which is written to, that is, the constraint starts
  // with "=", e.g. "=r".
  Operand *newOperand(StringRef constraint);

  // Create a constant integer operand.
  Operand *newConstantOperand(int v);
  // Create a constant operand with explicit code specified.
  Operand *newConstantOperand(const std::string &v);

  Operand *newAddrOperand(mlir::Value addr, StringRef constraint, int off = 0);

  llvm::SmallVector<Operand *, 4> getAllArgs() const;

  llvm::SmallVector<Value, 4> getAllMLIRArgs() const;

  std::string getConstraints() const;

  std::string dump() const;

  mlir::Value launch(ConversionPatternRewriter &rewriter, Location loc,
                     Type resTy, bool hasSideEffect = true,
                     bool isAlignStack = false,
                     ArrayRef<Attribute> attrs = {}) const;

private:
  Operand *newOperand() {
    argArchive.emplace_back(std::make_unique<Operand>());
    return argArchive.back().get();
  }

  friend class GCNInstr;
  friend class GCNInstrCommon;

protected:
  llvm::SmallVector<std::unique_ptr<Operand>, 6> argArchive;
  llvm::SmallVector<std::unique_ptr<GCNInstrCommon>, 2> instrs;
  llvm::SmallVector<std::unique_ptr<GCNInstrExecution>, 4> executions;
  int oprCounter{};
};

// GCN instruction common interface.
// Put the generic logic for all the instructions here.
struct GCNInstrCommon {
  explicit GCNInstrCommon(GCNBuilder *builder) : builder(builder) {}

  using Operand = GCNBuilder::Operand;

  // clang-format off
  GCNInstrExecution& operator()() { return call({}); }
  GCNInstrExecution& operator()(Operand* a) { return call({a}); }
  GCNInstrExecution& operator()(Operand* a, Operand* b) { return call({a, b}); }
  GCNInstrExecution& operator()(Operand* a, Operand* b, Operand* c) { return call({a, b, c}); }
  GCNInstrExecution& operator()(Operand* a, Operand* b, Operand* c, Operand* d) { return call({a, b, c, d}); }
  GCNInstrExecution& operator()(Operand* a, Operand* b, Operand* c, Operand* d, Operand * e) { return call({a, b, c, d, e}); }
  GCNInstrExecution& operator()(Operand* a, Operand* b, Operand* c, Operand* d, Operand * e, Operand* f) { return call({a, b, c, d, e, f}); }
  GCNInstrExecution& operator()(Operand* a, Operand* b, Operand* c, Operand* d, Operand * e, Operand* f, Operand* g) { return call({a, b, c, d, e, f, g}); }
  // clang-format on

  // Set operands of this instruction.
  GCNInstrExecution &operator()(llvm::ArrayRef<Operand *> oprs);

protected:
  GCNInstrExecution &call(llvm::ArrayRef<Operand *> oprs);

  GCNBuilder *builder{};
  llvm::SmallVector<std::string, 4> instrParts;

  friend class GCNInstrExecution;
};

template <class ConcreteT> struct GCNInstrBase : public GCNInstrCommon {
  using Operand = GCNBuilder::Operand;

  explicit GCNInstrBase(GCNBuilder *builder, const std::string &name)
      : GCNInstrCommon(builder) {
    o(name);
  }

  ConcreteT &o(const std::string &suffix, bool predicate = true) {
    if (predicate)
      instrParts.push_back(suffix);
    return *static_cast<ConcreteT *>(this);
  }
};

struct GCNInstr : public GCNInstrBase<GCNInstr> {
  using GCNInstrBase<GCNInstr>::GCNInstrBase;
};

struct GCNInstrExecution {
  using Operand = GCNBuilder::Operand;

  llvm::SmallVector<Operand *> argsInOrder;

  GCNInstrExecution() = default;
  explicit GCNInstrExecution(GCNInstrCommon *instr,
                             llvm::ArrayRef<Operand *> oprs)
      : instr(instr), argsInOrder(oprs.begin(), oprs.end()) {}

  std::string dump() const;

  SmallVector<Operand *> getArgList() const;

  GCNInstrCommon *instr{};
};





} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TRITON_GPU_TO_LLVM_ASM_FORMAT_H_
