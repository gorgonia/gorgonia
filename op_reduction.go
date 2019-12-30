package gorgonia

/*
This file holds code for ndarray related reduction Ops.
What this means is we take a ndarray, and reduce the dimensions down - typically to 1.
For example, summing all the values in a matrix, or finding the max value.
There is an additional field in each of these Ops - the 'along' field. This is because it's not always we want to reduce a ndarray down to a single scalar number
*/

import (
	"encoding/binary"
	"fmt"
	"hash"
	"strings"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

func reductionType(d int, along []int) hm.Type {
	a := hm.TypeVariable('a')
	t := makeTensorType(d, a)

	axes := make(map[int]bool)
	for _, axis := range along {
		if axis < d {
			axes[axis] = true
		}
	}

	if d == 1 || len(axes) == 0 || len(axes) == d {
		// then it reduces down
		return hm.NewFnType(t, a)
	}

	var retType hm.Type
	if len(axes) == d-1 { // Only 1 non-reduced dim, so we can reduce to a vector as before.
		retType = makeTensorType(1, a)
	} else {
		retType = t
	}
	return hm.NewFnType(t, retType)
}

func reductionInferShape(along []int, in tensor.Shape) (tensor.Shape, error) {
	if len(along) == 0 {
		return tensor.ScalarShape(), nil
	}
	shape := in.Clone()
	for _, d := range along {
		if d >= shape.Dims() {
			return nil, fmt.Errorf("shape error, along %d is not a valid axis for shape %v", d, in)
		}
		shape[d] = 1
	}
	// special cases: if all dimensions are 1 -> ScalarShape, if exactly one dimension is != 1 -> vector
	vecD := 0
	numNot1 := 0
	for _, d := range shape {
		if d != 1 {
			vecD = d
			numNot1++
			if numNot1 > 1 {
				return shape, nil
			}
		}
	}
	if numNot1 == 0 {
		return tensor.ScalarShape(), nil
	}
	return tensor.Shape{vecD}, nil
}

func reductionDo(op Op, s string, f func(*tensor.Dense, ...int) (*tensor.Dense, error), along []int, inputs ...Value) (retVal Value, err error) {
	if err = checkArity(op, len(inputs)); err != nil {
		return
	}
	at := inputs[0].(tensor.Tensor)
	switch t := at.(type) {
	case *tensor.Dense:
		var ret *tensor.Dense
		if ret, err = f(t, along...); err == nil {
			if ret.IsScalar() {
				retVal, _ = anyToScalar(ret.ScalarValue())
			} else {
				// the tensor reduction ops remove collapsed dimensions, but here we preserve them except in special cases.
				// so we reshape the return to ensure the dimensions match.
				var sh tensor.Shape
				if sh, err = reductionInferShape(along, t.Shape()); err == nil {
					if err = ret.Reshape(sh...); err == nil {
						retVal = ret
					}
				}
			}
		} else {
			return nil, errors.Wrap(err, fmt.Sprintf("failed to apply *tensor.Dense.%s()", strings.Title(s)))
		}
	default:
		return nil, errors.Errorf(nyiFail, fmt.Sprintf("%sOp.Do()", s), at)
	}
	return

}

type maxOp struct {
	along axes
	d     int
}

func newMaxOp(along axes, dim int) *maxOp {
	return &maxOp{
		along: along,
		d:     dim,
	}
}

func (op maxOp) Arity() int { return 1 }

func (op maxOp) Type() hm.Type {
	return reductionType(op.d, op.along)
}

func (op maxOp) InferShape(dimsizers ...DimSizer) (tensor.Shape, error) {
	if len(dimsizers) != 1 {
		return nil, errors.Errorf("maxOp only takes one input shape to infer ")
	}
	return reductionInferShape(op.along, dimsizers[0].(tensor.Shape))
}
func (op maxOp) DiffWRT(i int) []bool { return []bool{true} }

func (op maxOp) SymDiff(inputs Nodes, output, gradNode *Node) (retVal Nodes, err error) {
	if err = checkArity(op, len(inputs)); err != nil {
		return
	}

	t := inputs[0]
	opDim := len(t.Shape())

	var leftAxes []byte
	for i := 0; i < opDim; i++ {
		for _, ax := range op.along {
			if i == ax {
				leftAxes = append(leftAxes, byte(i))
				break
			}
		}
	}

	var a, b, a2, b2, eq *Node
	bcpat := NewBroadcastPattern(leftAxes, nil)
	if a, b, err = Broadcast(output, t, bcpat); err != nil {
		return nil, errors.Wrap(err, operationError)
	}
	if eq, err = Eq(a, b, false); err != nil {
		return nil, errors.Wrap(err, operationError)
	}

	if a2, b2, err = Broadcast(gradNode, eq, bcpat); err != nil {
		return nil, errors.Wrap(err, operationError)
	}
	retVal = make(Nodes, 1)
	if retVal[0], err = Mul(a2, b2); err != nil {
		return nil, errors.Wrap(err, operationError)
	}
	return
}

func (op maxOp) Do(inputs ...Value) (retVal Value, err error) {
	return reductionDo(op, "max", (*tensor.Dense).Max, op.along, inputs...)
}

func (op maxOp) ReturnsPtr() bool     { return true }
func (op maxOp) OverwritesInput() int { return 0 }
func (op maxOp) CallsExtern() bool    { return false }

func (op maxOp) WriteHash(h hash.Hash) {
	h.Write([]byte("max"))
	if err := binary.Write(h, binary.LittleEndian, byte(op.d)); err != nil {
		panic(err)
	}
	fmt.Fprintf(h, "%v->%v", op.d, op.along)
}

func (op maxOp) Hashcode() uint32 { return simpleHash(op) }

func (op maxOp) String() string { return fmt.Sprintf("MaxAlong%v", op.along) }
func (op maxOp) isUnary() bool  { return true }

/* ARGMAX OP */
// type argmaxOp struct {
// 	along int // axis
// }

// func (op argmaxOp) Type() hm.Type {
// 	a := hm.TypeVariable('a')

// }

/* SUM OP */

type sumOp struct {
	along      axes
	d          int
	inputShape tensor.Shape
}

func newSumOp(along axes, s tensor.Shape, d int) sumOp {
	return sumOp{
		along:      along,
		d:          d,
		inputShape: s,
	}
}

func (op sumOp) Arity() int { return 1 }

// sumOp is a function with this type:
//		sumOp :: (Summable a) ⇒ Tensor d a → Tensor d-1 a
func (op sumOp) Type() hm.Type {
	return reductionType(op.d, op.along)
}

// InferShape infers the shape of a sumOp. It's purpose is to fulfil the Op interface. Only one input is expected, and the type is expected to be a tensor.Shape
func (op sumOp) InferShape(inputs ...DimSizer) (shape tensor.Shape, err error) {
	return reductionInferShape(op.along, inputs[0].(tensor.Shape))
}

func (op sumOp) DiffWRT(i int) []bool { return []bool{true} }

func (op sumOp) SymDiff(inputs Nodes, output, gradNode *Node) (retVal Nodes, err error) {
	if err = checkArity(op, len(inputs)); err != nil {
		return
	}

	children := make(Nodes, len(op.along)+1)
	children[0] = gradNode
	for i, a := range op.along {
		var n *Node
		if n, err = SizeOf(a, inputs[0]); err != nil {
			return nil, errors.Wrap(err, operationError)
		}
		WithGroupName(gradClust)(n)
		children[i+1] = n
	}

	retVal = make(Nodes, 1)
	repeat := newRepeatOp(op.along, children)

	symdiffLogf("repeat: %v", repeat.Type())
	symdiffLogf("children %#Y", children)
	symdiffLogf("children: %v", children)

	if retVal[0], err = ApplyOp(repeat, children...); err != nil {
		return nil, errors.Wrap(err, applyOpFail)
	}
	retVal[0].setGroup(gradClust)
	return
}

func (op sumOp) DoDiff(ctx ExecutionContext, inputs Nodes, output *Node) (err error) {
	if err = checkArity(op, len(inputs)); err != nil {
		return
	}

	x := inputs[0]
	xdv, ydv := getDV(x, output)
	xShape := xdv.Value.Shape()

	var T tensor.Tensor
	switch ydvd := ydv.d.(type) {
	case Scalar:
		dt := ydvd.Dtype()
		T = tensor.New(tensor.Of(dt), tensor.WithShape(xdv.d.Shape().Clone()...))
		T.Memset(ydvd.Data())
	case tensor.Tensor:
		// handle broadcasting
		if ydvd.Shape().Dims() == xdv.d.Shape().Dims()-len(op.along) {
			newShape := xdv.d.Shape().Clone()
			for _, a := range op.along {
				newShape[a] = 1
			}
			ydvd.Reshape(newShape...)
		}

		T = ydvd
	default:
		err = errors.Errorf(nyiTypeFail, "sumOp.DoDiff()", ydv.d)
		return
	}

	var val Value
	if !T.Shape().Eq(xdv.d.Shape()) {
		// TO DO: Optimize: figure out a way to bunch it all up so you can repeat in one call
		for _, a := range op.along {
			if xShape[a] == 1 {
				continue // don't need to repeat
			}

			if T, err = tensor.Repeat(T, a, xShape[a]); err != nil {
				return errors.Wrapf(err, repFail, a, xShape[a])
			}
		}
		val = T
	} else {
		val = T
	}

	// then just add the two
	add := newEBOByType(addOpType, TypeOf(xdv.d), TypeOf(val))
	addOp := NewExternalOp(add, ctx, nil)
	addOp.UseUnsafe = true
	addOp.Device = x.Device()

	dev := x.Device()
	if output.Device() != dev && dev != CPU {
		var valOnDev Value
		if valOnDev, err = ctx.Transfer(dev, output.Device(), val, false); err != nil {
			return
		}
		defer ctx.PutValue(dev, valOnDev)
		val = valOnDev

		// Copy(valOnDev, val)
	}
	var xd, d Value
	var extra bool
	if xd, extra, err = x.GradOnDevice(dev, ctx.External); err != nil {
		return errors.Wrapf(err, gradOnDeviceFail, x, dev)
	}
	if extra {
		defer ctx.PutValue(dev, xd)
	}
	if d, err = addOp.Do(xd, val); err != nil {
		return errors.Wrapf(err, unsafeDoFail, add)
	}

	return xdv.SetDeriv(d)

	// var d Value
	// if d, err = add.UnsafeDo(xdv.d, val); err != nil {
	// 	return errors.Wrapf(err, unsafeDoFail, add)
	// }
}

func (op sumOp) Do(inputs ...Value) (retVal Value, err error) {
	return reductionDo(op, "sum", (*tensor.Dense).Sum, op.along, inputs...)
}

func (op sumOp) ReturnsPtr() bool      { return true }
func (op sumOp) OverwritesInput() int  { return 0 }
func (op sumOp) CallsExtern() bool     { return false }
func (op sumOp) WriteHash(h hash.Hash) { fmt.Fprintf(h, "sum%v->%v", op.along, op.inputShape) }
func (op sumOp) Hashcode() uint32      { return simpleHash(op) }
func (op sumOp) String() string        { return fmt.Sprintf("Σ%v", op.along) }
func (op sumOp) isUnary() bool         { return true }
