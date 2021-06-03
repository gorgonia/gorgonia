package gorgonia

import (
	"fmt"
	"hash"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

// ConvType converts the type of the x Node from one type to other
func ConvType(x *Node, from, to tensor.Dtype) (*Node, error) {
	op := &dtConvOp{
		inshape: x.Shape(),
		from:    from,
		to:      to,
	}

	return ApplyOp(op, x)
}

type dtConvOp struct {
	inshape  tensor.Shape
	from, to tensor.Dtype
}

/* Graph Building Related Methods */

// Arity returns the number of inputs the Op expects. -1 indicates that it's n-ary and will be determined at runtime
func (op *dtConvOp) Arity() int { return 1 }

// Informs the type of the Op (not the node). This will be used by the type system to infer the final type of the node
func (op *dtConvOp) Type() hm.Type {
	if op.inshape.IsScalar() {
		return hm.NewFnType(op.from, op.to)
	}
	t := makeTensorType(op.inshape.Dims(), op.from)
	u := makeTensorType(op.inshape.Dims(), op.to)
	return hm.NewFnType(t, u)
}

// returns the output shape as a function of the inputs
func (op *dtConvOp) InferShape(_ ...DimSizer) (tensor.Shape, error) {
	return op.inshape.Clone(), nil
}

// Do executes the op
func (op *dtConvOp) Do(vals ...Value) (Value, error) {
	retVal := tensor.New(tensor.Of(op.to), tensor.WithShape(op.inshape.Clone()...))
	return op.UsePreallocDo(retVal, vals...)
}

/* Analysis Related Methods */ // indicates if the Op will return a pointer (allowing possible inplace edits) or by value
// if it's false, the return value of the Op will be a copy of its input
func (op *dtConvOp) ReturnsPtr() bool { return false }

// Does this op potentially call external (cgo or cuda) functions (thereby requiring extra overhead for Go's trampolining thing)
func (op *dtConvOp) CallsExtern() bool { return false }

// overwriteInput() is a method which states which input the output will be overwriting.
// This allows for some efficiency gains as the underlying arrays wouldn't have to be re-allocated.
// The method returns an int instead of a bool because potentially different operations may be allowed
// to overwrite certain inputs. For example, consider an operation to increment a value:
// the IncrementOp would be a unary operator, and assuming we would like to overwrite the input,
// the retVal of overwriteInput() will be 0 (inputs[0]).
// -1 is returned if overwriting of input is disallowed
func (op *dtConvOp) OverwritesInput() int { return -1 }

/* Other methods */
func (op *dtConvOp) WriteHash(h hash.Hash) { fmt.Fprintf(h, "(%v)", op.Type()) }

func (op *dtConvOp) Hashcode() uint32 { return simpleHash(op) }

func (op *dtConvOp) String() string { return fmt.Sprintf("%v", op.Type()) }

// DiffWRT indicates if the op is differentiable with regards to the given number of inputs
// returns []bool to indicate which input it is differentiable to
func (op *dtConvOp) DiffWRT(inputs int) []bool { return []bool{true} }

// SymDiff symbolically differentiates the op
func (op *dtConvOp) SymDiff(inputs Nodes, output *Node, grad *Node) (retVal Nodes, err error) {
	diffOp := &dtConvOp{
		inshape: grad.Shape().Clone(),
		from:    op.to,
		to:      op.from,
	}
	retVal = make(Nodes, op.Arity())
	retVal[0], err = ApplyOp(diffOp, grad)
	return retVal, err
}

// UsePreallocDo executes the Op with a preallocated value in the result.s
func (op *dtConvOp) UsePreallocDo(prealloc Value, inputs ...Value) (Value, error) {
	a := inputs[0]
	retVal := prealloc
	switch {
	case op.from == tensor.Float64 && op.to == tensor.Int:
		switch aData := a.Data().(type) {
		case []float64:
			retData := retVal.Data().([]int)
			for i := range aData {
				retData[i] = int(aData[i])
			}
		case float64:
			retVal = tensor.New(
				tensor.Of(tensor.Int),
				tensor.WithShape(1),
				tensor.WithBacking([]int{int(aData)}),
			)
		}
	case op.from == tensor.Float32 && op.to == tensor.Int:
		switch aData := a.Data().(type) {
		case []float32:
			retData := retVal.Data().([]int)
			for i := range aData {
				retData[i] = int(aData[i])
			}
		case float32:
			retVal = tensor.New(
				tensor.Of(tensor.Int),
				tensor.WithShape(1),
				tensor.WithBacking([]int{int(aData)}),
			)
		}
	case op.from == tensor.Int && op.to == tensor.Float64:
		switch aData := a.Data().(type) {
		case []int:
			retData := retVal.Data().([]float64)
			for i := range aData {
				retData[i] = float64(aData[i])
			}
		case int:
			retVal = tensor.New(
				tensor.Of(tensor.Float64),
				tensor.WithShape(1),
				tensor.WithBacking([]float64{float64(aData)}),
			)
		}
	case op.from == tensor.Int && op.to == tensor.Float32:
		switch aData := a.Data().(type) {
		case []int:
			retData := retVal.Data().([]float32)
			for i := range aData {
				retData[i] = float32(aData[i])
			}
		case int:
			retVal = tensor.New(
				tensor.Of(tensor.Float32),
				tensor.WithShape(1),
				tensor.WithBacking([]float32{float32(aData)}),
			)
		}
	default:
		return nil, errors.Errorf("Cannot do conversion %v", op.Type())
		// TODO: other types
	}

	return retVal, nil
}
