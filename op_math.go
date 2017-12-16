package gorgonia

/*
This file holds all the Ops that are related to doing math-related work. Due to the numerousness of
mathematical operations, they're classified into 3 main types:
	elemBinOp - a representation of a binary mathematical operation that is performed elementwise (example: +, *, -, or >, <)
	elemUnaryOp - a representation of a mathematical operation that is performed elmentwise
	linAlgBinOp - a representation of a binary mathematical operation that is performed on matrices

The individual operators are further exanded on operator*.go files. Their datatypes are often embedded in the datatypes here.

For all data type, the methods are standardized by arrangement in the order the Op interface is defined.
Any additional interfaces that the data type fulfils will be declared AFTER the Op interface methods.
*/

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"hash"
	"hash/fnv"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

/* ELEMENTWISE BINARY OPERATION */

// elemBinOp is the representation of an operation that is to be performed elementwise
type elemBinOp struct {
	ʘBinaryOperator
	arg0, arg1 hm.Type // pruned types only plz
	retSame    bool    // for comparison ops, return same type?
}

func newEBOByType(ot ʘBinaryOperatorType, at, bt hm.Type) elemBinOp {
	var binOp ʘBinaryOperator
	switch att := at.(type) {
	case tensor.Dtype:
		switch bt.(type) {
		case tensor.Dtype:
			binOp = scalarBinOp{
				ʘBinaryOperatorType: ot,
				t:                   att,
			}
		case TensorType:
			binOp = tBinOp{
				ʘBinaryOperatorType: ot,
				tensorLeft:          false,
			}
		default:
			panic(fmt.Sprintf("Unsupported type of b %v!", bt))
		}
	case TensorType:
		binOp = tBinOp{
			ʘBinaryOperatorType: ot,
			tensorLeft:          true,
		}
	default:
		panic(fmt.Sprintf("Unsupported type of a %v!", at))
	}
	return elemBinOp{
		ʘBinaryOperator: binOp,
		arg0:            at,
		arg1:            bt,
	}
}

func newElemBinOp(ot ʘBinaryOperatorType, a, b *Node) elemBinOp {
	// at := hm.Prune(a.t)
	// bt := hm.Prune(b.t)

	return newEBOByType(ot, a.t, b.t)
}

func (op elemBinOp) Arity() int { return 2 }

// elemBinOp has either of these types:
// 		elemBinOp :: (Floats a) ⇒ Tensor a → Tensor a → Tensor a
// 		elemBinOp :: (Floats a) ⇒ Tensor a → a → Tensor a
//		elemBinOp :: (Floats a) ⇒ a → Tensor a → a
//		elemBinOp :: (Floats a) ⇒ a → a → a
//		elemBinOp :: (Floats a) ⇒ a → a → Bool
// 		elemBinOp :: (Floats a) ⇒ Tensor a → Tensor a → Tensor Bool
// 		elemBinOp :: (Floats a) ⇒ Tensor a → a → Tensor Bool
//		elemBinOp :: (Floats a) ⇒ a → Tensor a → Bool
//
// To make things clearer, it helps to consider elemBinOp to be the representation of
// a dispatch table for different functions. In a sense it's "overloading" functions.
//
// At the moment, due to my refusal to create a sum type (which requires more finnicking with data constructors)
// Type() happens pretty much at close to run time
func (op elemBinOp) Type() hm.Type {
	a := hm.TypeVariable('a')

	var a0, a1, retType hm.Type
	switch arg0 := op.arg0.(type) {
	case TensorType:
		a0 = makeFromTensorType(arg0, a)
		retType = makeFromTensorType(arg0, a)
	default:
		a0 = a
		retType = a
	}

	switch arg1 := op.arg1.(type) {
	case TensorType:
		a1 = makeFromTensorType(arg1, a)
		retType = makeFromTensorType(arg1, a)
	default:
		a1 = a
	}

	if op.isArith() || (!op.isArith() && op.retSame) {
		return hm.NewFnType(a0, a1, retType)
	}

	switch rt := retType.(type) {
	case TensorType:
		rt.Of = Bool
		retType = rt
	default:
		retType = Bool
	}

	return hm.NewFnType(a0, a1, retType)
}

// elemBinOp has these allowed shapes:
// 		op :: () → () → ()
//		op :: () → (...) → (...)
//		op :: (...) → () → (...)
func (op elemBinOp) InferShape(inputs ...DimSizer) (retVal tensor.Shape, err error) {
	shapeLogf("Inferring shape of %v", op)
	enterLogScope()
	defer leaveLogScope()

	if inputs[0] == nil || inputs[1] == nil {
		return nil, errors.Errorf(nyiFail, "elemBinOp.inferShape", "runtime impl")
	}

	switch x := inputs[0].(type) {
	case tensor.Shape:
		switch y := inputs[1].(type) {
		case tensor.Shape:
			switch {
			case x.IsScalar() && y.IsScalar():
				retVal = scalarShape
			case x.IsScalar() && !y.IsScalar():
				retVal = y
			case !x.IsScalar() && y.IsScalar():
				retVal = x
			case !x.IsScalar() && !y.IsScalar():
				if !x.Eq(y) {
					return nil, errors.Errorf("Shape mismatch: %v and %v", x, y)
				}
				if x.Dims() > y.Dims() {
					retVal = x
				} else {
					retVal = y
				}
			}
		default:
			retVal = x
		}
	default:
		switch y := inputs[1].(type) {
		case tensor.Shape:
			retVal = y
		default:
			retVal = scalarShape
		}
	}
	return
}

// DiffWRT gives info on whether or not the operation is actually differentiable
// For example, this is differentiable:
//		c = a ** b
// The result of the differentiation wrt to a and b would be:
// 		dc/da = b * a ** (b-1)
// 		dc/db = <insert exp rule expansion here.. don't quite remember it> //TODO
//
// However, operators like < and > are NOT differentiable
//
// This method returns a slice of bools, indicating whether differentiation with regards to its operands
// can be done. Since binOp has 2 operands, we'll return a slice
func (op elemBinOp) DiffWRT(inputs int) []bool {
	if inputs != 2 {
		panic(fmt.Sprintf(binOpFail, inputs))
	}

	b := op.ʘBinaryOperator.binOpType()

	if b >= maxʘBinaryOpType {
		panic("Unsupported unary operator is not differentiable")
	}

	if b.isArith() {
		return []bool{true, true}
	}
	return []bool{false, false}
}

func (op elemBinOp) SymDiff(inputs Nodes, output, gradNode *Node) (retVal Nodes, err error) {
	if err = checkArity(op, len(inputs)); err != nil {
		return
	}

	b := op.ʘBinaryOperator.binOpType()

	if retVal, err = ʘBinOpDiffExprs[b](inputs[0], inputs[1], output, gradNode); err == nil {
		for _, n := range retVal {
			n.setGroup(gradClust)
		}
	}

	// needed to handle scalar gradients such as b in the logit regression example
	for i, grad := range retVal {
		if inputs[i].IsScalar() && !grad.IsScalar() {
			if retVal[i], err = Sum(grad); err != nil {
				err = errors.Wrap(err, operationError)
				return
			}
		}
	}

	return
}

func (op elemBinOp) Do(values ...Value) (Value, error) {
	return op.ʘBinaryOperator.Do(op.retSame, values...)
}

func (op elemBinOp) DoDiff(ctx ExecutionContext, inputs Nodes, output *Node) (err error) {
	if err = checkArity(op, len(inputs)); err != nil {
		return
	}

	b := op.ʘBinaryOperator.binOpType()
	if err = ʘBinOpDiffFns[b](ctx, inputs[0], inputs[1], output); err != nil {
		if _, ok := err.(AutoDiffError); !ok {
			return errors.Wrapf(err, autodiffFail, b)
		}
		err = nil
	}

	//handle scalar gradients
	for _, in := range inputs {
		indv := in.boundTo.(*dualValue)
		if _, ok := indv.d.(Scalar); in.IsScalar() && !ok {
			indvdT := indv.d.(tensor.Tensor)
			defer returnTensor(indvdT)

			var d Value
			var t tensor.Tensor
			if t, err = tensor.Sum(indvdT); err != nil {
				return errors.Wrap(err, operationError)
			}
			defer returnTensor(t)

			d, _ = anyToScalar(t.ScalarValue())
			indv.SetDeriv(d)
		}
	}
	return
}

func (op elemBinOp) ReturnsPtr() bool {
	// if _, ok := op.arg0.(TensorType); ok {
	// 	return true
	// } else if _, ok := op.arg1.(TensorType); ok {
	// 	return true
	// }

	// return false
	return true
}

func (op elemBinOp) OverwritesInput() int {
	if _, ok := op.arg0.(TensorType); ok {
		return 0
	}

	if _, ok := op.arg1.(TensorType); ok {
		return 1
	}
	return -1
}

func (op elemBinOp) WriteHash(h hash.Hash) {
	if err := binary.Write(h, binary.LittleEndian, op.binOpType()); err != nil {
		panic(err)
	}

	fmt.Fprintf(h, "%v,%v", op.arg0, op.arg1)
}

func (op elemBinOp) Hashcode() uint32 {
	h := fnv.New32a()
	op.WriteHash(h)
	return h.Sum32()
}

// Fulfils UsePreallocDoer interface
func (op elemBinOp) UsePreallocDo(prealloc Value, inputs ...Value) (retVal Value, err error) {
	if !op.ReturnsPtr() {
		return op.Do(inputs...)
	}

	if pd, ok := op.ʘBinaryOperator.(usePreallocDoerBinOp); ok {
		return pd.UsePreallocDo(prealloc, op.retSame, inputs...)
	}

	if retVal, err = op.Do(inputs...); err != nil {
		return
	}
	return Copy(prealloc, retVal)
}

// Fulfils UnsafeDoer interface
func (op elemBinOp) UnsafeDo(inputs ...Value) (retVal Value, err error) {
	if !op.ReturnsPtr() {
		return op.Do(inputs...)
	}

	if ud, ok := op.ʘBinaryOperator.(unsafeDoerBinOp); ok {
		return ud.UnsafeDo(op.retSame, inputs...)
	}
	return op.Do(inputs...)
}

// Fulfils the IncrDoer interface
func (op elemBinOp) IncrDo(incr Value, inputs ...Value) (err error) {
	if id, ok := op.ʘBinaryOperator.(incrDoerBinOp); ok {
		return id.IncrDo(incr, op.retSame, inputs...)
	}

	// if !op.ReturnsPtr() {
	var retVal Value
	if retVal, err = op.Do(inputs...); err != nil {
		return errors.Wrapf(err, doFail, op)
	}

	add := newEBOByType(addOpType, TypeOf(incr), TypeOf(retVal))
	if retVal, err = add.UnsafeDo(incr, retVal); err != nil {
		return errors.Wrapf(err, unsafeDoFail, add)
	}
	err = noIncrErr{retVal}
	return
	// }
}

func (op elemBinOp) String() string { return fmt.Sprintf("%v %t", op.ʘBinaryOperator, op.retSame) }

// Fulfils the BinaryOp interface
func (op elemBinOp) IsBinary() bool { return true }

/* ELEMENTWISE UNARY OP */

type elemUnaryOp struct {
	ʘUnaryOperator

	argTensor     bool
	numericResult bool // indicate if boolean results should be converted to 1 and 0 in the respective Dtype
}

func newElemUnaryOp(op ʘUnaryOperatorType, a *Node) elemUnaryOp {
	dt, err := dtypeOf(a.t)
	if err != nil {
		panic(err)
	}

	_, isTensor := a.t.(TensorType)

	var operator ʘUnaryOperator
	switch dt {
	case Float32:
		operator = sf32UnaryOperators[op]
	case Float64:
		operator = sf64UnaryOperators[op]
	}

	return elemUnaryOp{
		ʘUnaryOperator: operator,
		argTensor:      isTensor,
	}
}

func (op elemUnaryOp) Arity() int { return 1 }

// all pointwise unary operations have this type:
//		op :: (Arithable a) ⇒ a → a
func (op elemUnaryOp) Type() hm.Type {
	a := hm.TypeVariable('a')
	return hm.NewFnType(a, a)
}

func (op elemUnaryOp) InferShape(inputs ...DimSizer) (retVal tensor.Shape, err error) {
	if inputs[0] == nil {
		return nil, errors.Errorf(nyiFail, "inferShape", "nil shape")
	}

	return inputs[0].(tensor.Shape), nil
}

// diffWRT gives info on whether or not the operation is actually differentiable wrt to its inputs
//
// some operations, such as ceil(), sign(), floor cannot be differentiated wrt to its inputs (or I don't actually know how to do them)
func (op elemUnaryOp) DiffWRT(inputs int) []bool {
	if inputs != 1 {
		panic(fmt.Sprintf("unary operator only supports one input, got %d instead", inputs))
	}

	u := op.ʘUnaryOperator.unaryOpType()

	if u >= maxʘUnaryOperator {
		panic("Unsupported unary operator is not differentiable")
	}
	return []bool{ʘUnaryOpDifferentiable[u]}
}

func (op elemUnaryOp) SymDiff(inputs Nodes, output, gradNode *Node) (retVal Nodes, err error) {
	if err = checkArity(op, len(inputs)); err != nil {
		return
	}

	u := op.ʘUnaryOperator.unaryOpType()

	var n *Node
	if n, err = ʘUnaryOpDiffExprs[u](inputs[0], output, gradNode); err == nil {
		n.setGroup(gradClust)
		retVal = Nodes{n}
	}
	return
}

func (op elemUnaryOp) DoDiff(ctx ExecutionContext, inputs Nodes, output *Node) (err error) {
	if err = checkArity(op, len(inputs)); err != nil {
		return
	}

	u := op.ʘUnaryOperator.unaryOpType()
	return ʘUnaryOpDiffFns[u](inputs[0], output)
}

func (op elemUnaryOp) Do(inputs ...Value) (retVal Value, err error) {
	if err = checkArity(op, len(inputs)); err != nil {
		return
	}
	return op.do(inputs[0])
}

func (op elemUnaryOp) ReturnsPtr() bool {
	// if op.argTensor {
	// 	return true
	// }
	// return false
	return true
}

func (op elemUnaryOp) OverwritesInput() int {
	if op.argTensor {
		return 0
	}
	return -1
}

func (op elemUnaryOp) WriteHash(h hash.Hash) {
	if err := binary.Write(h, binary.LittleEndian, op.unaryOpType()); err != nil {
		panic(err)
	}

	if op.argTensor {
		h.Write([]byte{1})
	} else {
		h.Write([]byte{0})
	}
}

func (op elemUnaryOp) Hashcode() uint32 {
	h := fnv.New32a()
	op.WriteHash(h)
	return h.Sum32()
}

// fulfils UnsafeDoer interface
func (op elemUnaryOp) UnsafeDo(inputs ...Value) (Value, error) {
	if err := checkArity(op, len(inputs)); err != nil {
		return nil, err
	}
	return op.do(inputs[0], tensor.UseUnsafe())
}

// fulfils UnaryOp interface

func (op elemUnaryOp) isUnary() bool { return true }

// misc private methods

func (op elemUnaryOp) do(a Value, opts ...tensor.FuncOpt) (retVal Value, err error) {
	switch v := a.(type) {
	case tensor.Tensor:
		var t tensor.Tensor
		var fn interface{}
		switch opFn := op.ʘUnaryOperator.(type) {
		case *sf64UnaryOperator:
			fn = (func(float64) float64)(*opFn)
		case *sf32UnaryOperator:
			fn = (func(float32) float32)(*opFn)
		}

		if t, err = v.Apply(fn, opts...); err != nil {
			return nil, errors.Wrap(err, applyFail)
		}
		retVal = t
	case Scalar:
		vt := v.Dtype()
		switch vt {
		case tensor.Float32:
			vs := v.(*F32)
			f := float32(*vs)
			opFn := op.ʘUnaryOperator.(*sf32UnaryOperator)
			retVal, _ = anyToScalar((*opFn)(f))
		case tensor.Float64:
			vs := v.(*F64)
			f := float64(*vs)
			opFn := op.ʘUnaryOperator.(*sf64UnaryOperator)
			retVal, _ = anyToScalar((*opFn)(f))
		default:
			return nil, errors.Errorf(nyiFail, "elemUnaryOp.do", vt)
		}
	}
	return
}

/* LINEAR ALGEBRA RELATED OPERATIONS */

type linAlgBinOp struct {
	āBinaryOperator
	transA, transB bool
}

func (op linAlgBinOp) Arity() int { return 2 }

func (op linAlgBinOp) InferShape(inputs ...DimSizer) (retVal tensor.Shape, err error) {
	shapeLogf("Inferring shape of %v", op)
	enterLogScope()
	defer leaveLogScope()

	if inputs[0] == nil || inputs[1] == nil {
		return nil, nyi("InferShape for linalgBinOp", "runtime impl")
	}

	x, y := inputs[0].(tensor.Shape), inputs[1].(tensor.Shape)
	if x == nil || y == nil {
		return nil, errors.Errorf("Cannot infer shape from %v %v", x, y)
	}

	shapeLogf("x.shape: %v; y.shape: %v", x, y)
	// TODO: add checks for tensors greater than 2 d

	switch op.āBinaryOperator {
	case matMulOperator:
		if op.transA {
			x = transpose2D(x)
			defer tensor.ReturnInts(x)
		}
		if op.transB {
			y = transpose2D(y)
			defer tensor.ReturnInts(y)
		}

		retVal = tensor.Shape{x[0], y[1]}
	case matVecMulOperator:
		if op.transA {
			x = transpose2D(x)
			defer tensor.ReturnInts(x)
		}
		if x[0] != y[0] && x[1] != y[0] {
			return nil, errors.Errorf("Incompatible shapes: %v and %v", x, y)
		}

		switch {
		case x[0] == y[0]:
			retVal = tensor.Shape{x[1]}
		case x[1] == y[0]:
			retVal = tensor.Shape{x[0]}
		}

	case vecDotOperator:
		retVal = scalarShape
	case outerProdOperator:
		// outerprods only handles vec x vec for now
		retVal = tensor.Shape{x.TotalSize(), y.TotalSize()}
	case batchedMatMulOperator:
		// check that x and y are 3
		if x.Dims() != 3 {
			return nil, errors.Errorf("BatchedMatMul only works with 3D tensors as x")
		}
		if y.Dims() != 3 {
			return nil, errors.Errorf("BatchedMatMul only works with 3D tensors as y")
		}
		if x[0] != y[0] {
			return nil, errors.Errorf("BatchedMatMul has encounted a batch mismatch: %v %v", x, y)
		}
		batchSize := x[0]
		if op.transA {
			x = transpose2D(x[1:])
			defer tensor.ReturnInts(x)
		}
		if op.transB {
			y = transpose2D(y[1:])
			defer tensor.ReturnInts(y)
		}
		retVal = tensor.Shape{batchSize, x[0], y[1]}
	}
	return
}

func (op linAlgBinOp) SymDiff(inputs Nodes, output, gradNode *Node) (retVal Nodes, err error) {
	if err = checkArity(op, len(inputs)); err != nil {
		return
	}

	o := op.āBinaryOperator

	if retVal, err = āBinOpDiffExprs[o](op.transA, op.transB, inputs[0], inputs[1], output, gradNode); err != nil {
		return nil, errors.Wrap(err, "Failed to differentiate expressions")
	}

	for _, n := range retVal {
		n.setGroup(gradClust)
	}
	return
}

func (op linAlgBinOp) DoDiff(ctx ExecutionContext, inputs Nodes, output *Node) (err error) {
	if err = checkArity(op, len(inputs)); err != nil {
		return
	}

	o := op.āBinaryOperator
	return āBinOpDiffs[o](ctx, op.transA, op.transB, inputs[0], inputs[1], output)
}

func (op linAlgBinOp) Do(inputs ...Value) (retVal Value, err error) { return op.do(inputs) }
func (op linAlgBinOp) ReturnsPtr() bool                             { return true }
func (op linAlgBinOp) OverwritesInput() int                         { return -1 }
func (op linAlgBinOp) CallsExtern() bool {
	if op.āBinaryOperator != vecDotOperator {
		return true
	}
	return false
}

func (op linAlgBinOp) WriteHash(h hash.Hash) {
	if err := binary.Write(h, binary.LittleEndian, op.āBinaryOperator); err != nil {
		panic(err)
	}

	if op.transA {
		h.Write([]byte{1})
	} else {
		h.Write([]byte{0})
	}

	if op.transB {
		h.Write([]byte{1})
	} else {
		h.Write([]byte{0})
	}
}

func (op linAlgBinOp) Hashcode() uint32 {
	h := fnv.New32a()
	op.WriteHash(h)
	return h.Sum32()
}

func (op linAlgBinOp) String() string {
	var buf bytes.Buffer

	switch op.āBinaryOperator {
	case matMulOperator, matVecMulOperator:
		buf.WriteString("A")
	case vecDotOperator, outerProdOperator:
		buf.WriteString("a")
	}

	if op.transA {
		buf.WriteString("ᵀ")
	}

	switch op.āBinaryOperator {
	case matMulOperator:
		fmt.Fprintf(&buf, " %v B", op.āBinaryOperator)
	case matVecMulOperator, vecDotOperator, outerProdOperator:
		fmt.Fprintf(&buf, " %v b", op.āBinaryOperator)
	}

	if op.transB {
		buf.WriteString("ᵀ")
	}

	return buf.String()
}

// fulfils IncrDoer
func (op linAlgBinOp) IncrDo(incr Value, inputs ...Value) (err error) {
	t, ok := incr.(tensor.Tensor)

	switch {
	case ok && op.āBinaryOperator != batchedMatMulOperator:
		_, err = op.do(inputs, tensor.WithIncr(t))
		return
	case ok && op.āBinaryOperator == batchedMatMulOperator:
		_, err = op.preallocBatchMatMul(true, incr, inputs...)
		return
	}

	var retVal Value
	if retVal, err = op.do(inputs); err != nil {
		return errors.Wrapf(err, doFail, op)
	}

	add := newEBOByType(addOpType, TypeOf(incr), TypeOf(retVal))
	if retVal, err = add.UnsafeDo(incr, retVal); err != nil {
		return errors.Wrapf(err, unsafeDoFail, add)
	}

	err = noIncrErr{retVal}
	return
}

// fulfils UsePreallocDoer
func (op linAlgBinOp) UsePreallocDo(prealloc Value, inputs ...Value) (retVal Value, err error) {
	t, ok := prealloc.(tensor.Tensor)
	if !ok {
		return nil, errors.Errorf("Expected Tensor as preallocated value. Got %v of %T instead", prealloc, prealloc)
	}
	if op.āBinaryOperator == batchedMatMulOperator {
		return op.preallocBatchMatMul(false, prealloc, inputs...)
	}
	return op.do(inputs, tensor.WithReuse(t))
}

// fulfils BinaryOp
func (op linAlgBinOp) IsBinary() bool { return true }

/* PRIVATE METHODS */

func (op linAlgBinOp) do(inputs []Value, opts ...tensor.FuncOpt) (retVal Value, err error) {
	if err = checkArity(op, len(inputs)); err != nil {
		return
	}

	a, b := inputs[0].(tensor.Tensor), inputs[1].(tensor.Tensor)

	if op.transA && op.āBinaryOperator != batchedMatMulOperator {
		if err = a.T(); err != nil {
			return nil, errors.Wrap(err, tFail)
		}
		// untranspose
		defer a.T()
	}

	if op.transB && op.āBinaryOperator != batchedMatMulOperator {
		if err = b.T(); err != nil {
			return nil, errors.Wrap(err, tFail)
		}
		// untranspose
		defer b.T()
	}

	switch op.āBinaryOperator {
	case matMulOperator:
		retVal, err = tensor.MatMul(a, b, opts...)
	case matVecMulOperator:
		retVal, err = tensor.MatVecMul(a, b, opts...)
	case vecDotOperator:
		var ret interface{}
		if ret, err = tensor.Inner(a, b); err != nil {
			return nil, errors.Wrapf(err, "Failed to carry out linalgBinOp operation %v", op)
		}
		retVal, _ = anyToScalar(ret)
	case outerProdOperator:
		retVal, err = tensor.Outer(a, b, opts...)
	case batchedMatMulOperator:
		// checks were done when the op was created
		retVal, err = batchedMatMul(a, b, nil, op.transA, op.transB, false)
	}
	return

}

func (op linAlgBinOp) preallocBatchMatMul(incr bool, prealloc Value, inputs ...Value) (retVal Value, err error) {
	if err = checkArity(op, len(inputs)); err != nil {
		return
	}
	a, b := inputs[0].(tensor.Tensor), inputs[1].(tensor.Tensor)
	c := prealloc.(tensor.Tensor)
	return batchedMatMul(a, b, c, op.transA, op.transB, incr)
}
