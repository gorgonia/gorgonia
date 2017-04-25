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
	"hash/fnv"

	"github.com/chewxy/gorgonia/tensor"
	"github.com/chewxy/hm"
	"github.com/pkg/errors"
)

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
	a := hm.TypeVariable('a')
	t := newTensorType(op.d, a)

	var retType hm.Type
	if op.d == 1 || len(op.along) == 0 || len(op.along) == op.d {
		// then it redueces down
		return hm.NewFnType(t, a)
	}
	retType = newTensorType(op.d-1, a)
	return hm.NewFnType(t, retType)
}

func (op maxOp) InferShape(...DimSizer) (tensor.Shape, error) { return scalarShape, nil } // TODO, THIS IS INCORRECT
func (op maxOp) DiffWRT(i int) []bool                         { return []bool{true} }

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

	var eq *Node
	bcpat := NewBroadcastPattern(leftAxes, nil)
	if eq, err = Broadcast(eqOpType, output, t, bcpat); err != nil {
		return nil, errors.Wrap(err, operationError)
	}

	retVal[0], err = Broadcast(mulOpType, gradNode, eq, bcpat)
	if err != nil {
		return nil, errors.Wrap(err, operationError)
	}
	return
}

func (op maxOp) Do(inputs ...Value) (retVal Value, err error) {
	if err = checkArity(op, len(inputs)); err != nil {
		return
	}

	return nil, errors.Errorf(nyiFail, "maxOp.Do", "maxOp")
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

func (op maxOp) Hashcode() uint32 {
	h := fnv.New32a()
	op.WriteHash(h)
	return h.Sum32()
}

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
	a := hm.TypeVariable('a')
	t := newTensorType(op.d, a)

	if op.inputShape.IsVector() {
		return hm.NewFnType(t, a)
	}

	// if it's a monotonic axes, it's basically summing everything.
	if monotonic, incr1 := tensor.IsMonotonicInts(op.along); monotonic && incr1 && len(op.along) == len(op.inputShape) {
		return hm.NewFnType(t, a)
	}

	return hm.NewFnType(t, newTensorType(op.d-1, a))
}

func (op sumOp) InferShape(inputs ...DimSizer) (shape tensor.Shape, err error) {
	in := inputs[0].(tensor.Shape)
	shapeLogf("input shape: %v", in)
	switch {
	case in.IsScalar():
		shape = scalarShape
	case in.IsVector() && !in.IsRowVec() && !in.IsColVec():
		if len(op.along) > 1 || (len(op.along) == 1 && op.along[0] != 0) {
			return nil, errors.Errorf("Shape mismatch: along is %v. Shape is %v", op.along, in)
		}
		shape = scalarShape
	default:
		shape = in.Clone()
		if len(op.along) > len(shape) {
			return nil, errors.Errorf("Shape mismatch: %v and %v", shape, op.along)
		}

		// special case (sum all)
		if monotonic, incr1 := tensor.IsMonotonicInts(op.along); monotonic && incr1 && len(op.along) == len(shape) && op.along[0] == 0 {
			shape = scalarShape
			return
		}

		for _, a := range op.along {
			if a >= len(shape) {
				return nil, errors.Errorf("Axis %d is greater or equal to the length of the shape %v", a, shape)
			}
			shape[a] = 1
		}

		switch {

		case shape.IsColVec():
			shape = shape[:1]
		case shape.IsRowVec():
			shape = shape[1:]
		}

	}
	return
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
	retVal[0], err = applyOp(repeat, children...)
	if err != nil {
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
	xdv := x.boundTo.(*dualValue)
	ydv := output.boundTo.(*dualValue)
	xShape := xdv.Value.Shape()

	logf("sumOp DoDiff: 0x%x | x Device %v | outputDevice %v", xdv.d.Uintptr(), x.Device(), output.Device())

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
	if dev == CPU {
		addOp.UseCPU = true
	}

	logf("addOp.USECPU %v | %v", addOp.UseCPU, addOp.Device)

	if output.Device() != dev && dev != CPU {
		// transfer to device
		var mem Memory
		if mem, err = ctx.GetFromValue(dev, val); err != nil {
			return
		}
		if val, err = makeValueFromMem(TypeOf(val), val.Shape(), mem); err != nil {
			return
		}

		defer ctx.PutValue(dev, val)
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

	logf("0x%x d 0x%x", xdv.d.Uintptr, d.Uintptr())
	return xdv.SetDeriv(d)

	// var d Value
	// if d, err = add.UnsafeDo(xdv.d, val); err != nil {
	// 	return errors.Wrapf(err, unsafeDoFail, add)
	// }
}

func (op sumOp) Do(inputs ...Value) (retVal Value, err error) {
	if err = checkArity(op, len(inputs)); err != nil {
		return
	}

	a := inputs[0]
	at := a.(tensor.Tensor)
	switch t := at.(type) {
	case *tensor.Dense:
		var ret *tensor.Dense
		if ret, err = t.Sum(op.along...); err == nil {
			if ret.IsScalar() {
				retVal, _ = anyToScalar(ret.ScalarValue())
			} else {
				retVal = ret
			}
		} else {
			return nil, errors.Wrap(err, "failed to apply *tensor.Dense.Sum()")
		}
	default:
		return nil, errors.Errorf(nyiFail, "sumOp.Do()", at)
	}
	return
}

func (op sumOp) ReturnsPtr() bool     { return true }
func (op sumOp) OverwritesInput() int { return 0 }
func (op sumOp) CallsExtern() bool    { return false }

func (op sumOp) WriteHash(h hash.Hash) {
	h.Write([]byte("sum"))
	fmt.Fprintf(h, "%v->%v", op.along, op.inputShape)
}

func (op sumOp) Hashcode() uint32 {
	h := fnv.New32a()
	op.WriteHash(h)
	return h.Sum32()
}

func (op sumOp) String() string { return fmt.Sprintf("Σ%v", op.along) }
func (op sumOp) isUnary() bool  { return true }
