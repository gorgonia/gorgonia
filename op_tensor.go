package gorgonia

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"hash"
	"hash/fnv"
	"log"

	"github.com/chewxy/gorgonia/tensor"
	tb "github.com/chewxy/gorgonia/tensor/b"
	tf32 "github.com/chewxy/gorgonia/tensor/f32"
	tf64 "github.com/chewxy/gorgonia/tensor/f64"
	ti "github.com/chewxy/gorgonia/tensor/i"
	"github.com/chewxy/gorgonia/tensor/types"
	"github.com/chewxy/hm"
	"github.com/pkg/errors"
)

/* This file contains tensor related Ops */

// atOp takes a Tensor and returns the value at the coordinates.
type atOp struct {
	coordinates coordinates
	d           int
}

func (op atOp) Arity() int { return 1 }

// atOp has this type
//		op :: Tensor a → a
func (op atOp) Type() hm.Type {
	a := hm.TypeVariable('a')
	tt := newTensorType(op.d, a)

	return hm.NewFnType(tt, a)
}

func (op atOp) ReturnsPtr() bool                                       { return false }
func (op atOp) OverwritesInput() int                                   { return -1 }
func (op atOp) CallsExtern() bool                                      { return false }
func (op atOp) InferShape(...DimSizer) (retVal types.Shape, err error) { return scalarShape, nil }
func (op atOp) DiffWRT(i int) []bool                                   { return make([]bool, i) }
func (op atOp) SymDiff(Nodes, *Node, *Node) (Nodes, error)             { return nil, nondiffErr(op) }
func (op atOp) String() string                                         { return fmt.Sprintf("At(%v)", op.coordinates) }

func (op atOp) Do(inputs ...Value) (retVal Value, err error) {
	if err = checkArity(op, len(inputs)); err != nil {
		return
	}

	t := inputs[0].(types.Tensor)
	switch tt := t.(type) {
	case *tf64.Tensor:
		r := tt.At(op.coordinates...)
		retVal, _, _, err = anyToValue(r)
	case *tf32.Tensor:
		r := tt.At(op.coordinates...)
		retVal, _, _, err = anyToValue(r)
	default:
		err = errors.Errorf(nyiTypeFail, "atOp.Do()", t)
	}
	return
}

func (op atOp) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, "atOp")
	if err := binary.Write(h, binary.LittleEndian, op.d); err != nil {
		panic(err)
	}
	fmt.Fprintf(h, "at%v", op.coordinates)
}

func (op atOp) Hashcode() uint32 {
	h := fnv.New32a()
	op.WriteHash(h)
	return h.Sum32()
}

func (op atOp) isStmt() bool { return true }

type sizeOp struct {
	axis, d int
	val     int // if we know ahead of time what the size is...
}

func (op sizeOp) Arity() int { return 1 }

// sizeOp is a function with this type:
//		sizeOp :: Tensor d a → a
func (op sizeOp) Type() hm.Type {
	a := hm.TypeVariable('a')

	// handle scalar cases
	if op.d == 0 {
		return hm.NewFnType(a, a)
	}

	tt := newTensorType(op.d, a)
	return hm.NewFnType(tt, a)
}

func (op sizeOp) ReturnsPtr() bool                            { return false }
func (op sizeOp) OverwritesInput() int                        { return -1 }
func (op sizeOp) CallsExtern() bool                           { return false }
func (op sizeOp) InferShape(...DimSizer) (types.Shape, error) { return scalarShape, nil } // TODO: return error
func (op sizeOp) DiffWRT(i int) []bool                        { return []bool{false} }
func (op sizeOp) String() string {
	if op.val != 0 {
		return fmt.Sprintf("SizeOf=%d", op.val)
	}
	return fmt.Sprintf("SizeOf(%d)", op.axis)
}

func (op sizeOp) SymDiff(inputs Nodes, output, gradNode *Node) (Nodes, error) {
	return nil, nondiffErr(op)
}

func (op sizeOp) Do(inputs ...Value) (retVal Value, err error) {
	if err = checkArity(op, len(inputs)); err != nil {
		return
	}

	switch t := inputs[0].(type) {
	case Scalar:
		retVal = one(DtypeOf(t))

		// bools are special
		if _, ok := t.(B); ok {
			retVal = I(1)
		}
	case types.Tensor:
		sh := t.Shape()
		if op.axis >= len(sh) {
			return nil, errors.Errorf("Shape is %v. Want size of %d", sh, op.axis)
		}
		size := sh[op.axis]

		// cast as ... types
		switch DtypeOf(t) {
		case Float64:
			retVal = F64(size)
		case Float32:
			retVal = F32(size)
		case Int:
			retVal = I(size)
		default:
			return nil, errors.Errorf(nyiFail, "sizeOf.Do()", t.Dtype())
		}
	}

	return
}

func (op sizeOp) WriteHash(h hash.Hash) {
	h.Write([]byte("sizeOf"))
	if err := binary.Write(h, binary.LittleEndian, byte(op.d)); err != nil {
		panic(err)
	}
	h.Write([]byte("on"))
	if err := binary.Write(h, binary.LittleEndian, byte(op.axis)); err != nil {
		panic(err)
	}
}

func (op sizeOp) Hashcode() uint32 {
	h := fnv.New32a()
	op.WriteHash(h)
	return h.Sum32()
}

func (op sizeOp) DimSize(d int) (int, error) {
	if d != op.axis {
		return -1, errors.Errorf("Dimension mismatch. Size Op is for axis %d. Want Dim Size of %d", op.axis, d)
	}
	return op.val, nil
}

type repeatOp struct {
	along axes

	inputShape types.Shape
	d          int

	arg0Dim  int
	children int
}

func newRepeatOp(along axes, children Nodes) *repeatOp {
	retVal := &repeatOp{
		along:    along,
		children: len(children),
		arg0Dim:  children[0].Dims(),
	}

	if s, err := retVal.InferShape(children.dimSizers()...); err == nil {
		retVal.inputShape = s
		if s.IsColVec() {
			retVal.d = 1
		} else {
			retVal.d = s.Dims()
		}
	} else {
		panic(err)
	}

	return retVal
}

func (op repeatOp) Arity() int { return -1 }

// repeat is an overload of three functions:
//		repeat :: a → a → a → Tensor a
// 		repeat :: Tensor a → a → a → Tensor a
//		repeat :: a → 1 → 1 → a
//
// The last of which is a special case of the first. But I didn't want to create a dependent-type system
// for a trivial language, so I'll just hardcode this in
func (op repeatOp) Type() hm.Type {
	a := hm.TypeVariable('a')
	var i0t hm.Type
	var rt hm.Type

	if op.arg0Dim == 0 {
		i0t = a
	} else {
		i0t = newTensorType(op.arg0Dim, a)
	}

	// if we know the result already, then we know the return type as well
	if op.d == 0 && op.inputShape != nil {
		rt = a
	} else {
		rt = newTensorType(op.d, a)
	}

	var ft hm.Types
	ft = append(ft, i0t)

	for i := 1; i < op.children; i++ {
		ft = append(ft, a)
	}
	ft = append(ft, rt)
	return hm.NewFnType(ft...)
}

func (op repeatOp) ReturnsPtr() bool     { return true }
func (op repeatOp) OverwritesInput() int { return -1 }
func (op repeatOp) CallsExtern() bool    { return false }

func (op repeatOp) InferShape(inputs ...DimSizer) (retVal types.Shape, err error) {
	input := inputs[0].(types.Shape)
	repeats := inputs[1:]

	knownRepeats := make([]int, len(repeats))
	for i, rep := range repeats {
		if r, ok := rep.(sizeOp); ok {
			knownRepeats[i] = r.val
		}
	}

	if monotonic, incr := types.IsMonotonicInts(op.along); monotonic && incr && input.IsScalar() {
		if input.IsScalar() {
			retVal = types.Shape(types.BorrowInts(len(knownRepeats)))
			copy(retVal, knownRepeats)
			return
		}
	} else {
		retVal = input
	}

	for i, axis := range op.along {
		rep := knownRepeats[i]
		if rep == 1 || rep == 0 { // 0 means unknown
			continue
		}
		if retVal, _, _, err = retVal.Repeat(axis, rep); err != nil {
			return
		}
	}
	return
}

func (op repeatOp) DiffWRT(i int) []bool {
	symdiffLogf("DiffWRT: %d", i)
	retVal := make([]bool, i)
	retVal[0] = true
	return retVal
}

func (op repeatOp) SymDiff(inputs Nodes, output, gradNode *Node) (retVal Nodes, err error) {
	var n *Node
	if n, err = Sum(gradNode, op.along...); err == nil {
		n.setGroup(gradClust)
	}
	retVal = make(Nodes, len(inputs))
	retVal[0] = n
	return
}

func (op repeatOp) DoDiff(inputs Nodes, output *Node) (err error) {
	if err = checkArity(op, len(inputs)); err != nil {
		return
	}

	xdv := inputs[0].boundTo.(*dualValue)
	ydv := output.boundTo.(*dualValue)

	var reps []int
	var repeats []Value
	for _, r := range inputs[1:] {
		repeats = append(repeats, r.Value())
	}

	if reps, err = valuesToInts(repeats); err != nil {
		return
	}

	xshape := xdv.Shape()
	var d Value
	d = ydv.d

	// we make it a colVec
	if xshape.IsVector() && !xshape.IsColVec() && !xshape.IsRowVec() {
		xshape = types.Shape{xshape[0], 1}
	}

	if xshape.IsScalar() {
		sum := newSumOp(op.along, output.shape, output.Dims())
		if d, err = sum.Do(d); err != nil {
			err = errors.Wrapf(err, doFail, sum)
			return
		}
	} else {
		for _, axis := range op.along {
			if xshape[axis] == 1 {
				sum := newSumOp(op.along, output.shape, output.Dims())
				if d, err = sum.Do(d); err != nil {
					err = errors.Wrapf(err, doFail, sum)
					return
				}
			} else {
				newShape := xshape.Clone()
				newShape = newShape[0 : axis+1]
				newShape = append(newShape, reps...)
				if axis+1 < xshape.Dims() {
					newShape = append(newShape, xshape[axis+1:]...)
				}

				along := []int{axis + 1}

				// a scalar can never get to this path
				t := d.(types.Tensor)
				if err = t.Reshape(newShape...); err != nil {
					err = errors.Wrapf(err, reshapeFail, newShape, t.DataSize())
					return
				}

				sum := newSumOp(along, newShape, len(newShape))
				if d, err = sum.Do(d); err != nil {
					err = errors.Wrapf(err, doFail, sum)
					return
				}
			}
		}
	}

	add := newEBOByType(addOpType, TypeOf(xdv.d), TypeOf(d))
	if d, err = add.UnsafeDo(xdv.d, d); err != nil {
		return
	}

	if !add.ReturnsPtr() || inputs[0].IsScalar() {
		err = xdv.SetDeriv(d)
	}

	return
}

func (op repeatOp) String() string { return fmt.Sprintf("Repeat%v", op.along) }

func (op repeatOp) Do(inputs ...Value) (retVal Value, err error) {
	if err = checkArity(op, len(inputs)); err != nil {
		return
	}

	// process inputs[1:]
	var reps []int
	repeats := inputs[1:]
	if len(repeats) != len(op.along) {
		err = errors.Errorf("Repeat Mismatch. Expected %d inputs. Got %d inputs instead", len(op.along), len(repeats))
		return
	}

	if reps, err = valuesToInts(repeats); err != nil {
		err = errors.Wrap(err, "Values To Ints failed in repeatOp.Do")
		return
	}

	monotonic, incr := types.IsMonotonicInts(op.along)

	// process inputs[0]
	var t types.Tensor
	switch iv := inputs[0].(type) {
	case F64:
		s := float64(iv)
		if monotonic && incr {
			ret := tf64.NewTensor(tf64.WithShape(reps...))
			err = ret.SetAll(s)
			retVal = ret

			return
		}
		t = tf64.NewTensor(tf64.AsScalar(s))
	case F32:
		s := float32(iv)
		if monotonic && incr {
			ret := tf32.NewTensor(tf32.WithShape(reps...))
			err = ret.SetAll(s)
			retVal = ret

			return
		}
		t = tf32.NewTensor(tf32.AsScalar(s))
	case I:
		s := int(iv)
		if monotonic && incr {
			ret := ti.NewTensor(ti.WithShape(reps...))
			err = ret.SetAll(s)
			retVal = ret

			return
		}
		t = ti.NewTensor(ti.AsScalar(s))
	case B:
		s := bool(iv)
		if monotonic && incr {
			ret := tb.NewTensor(tb.WithShape(reps...))
			err = ret.SetAll(s)
			retVal = ret

			return
		}
		t = tb.NewTensor(tb.AsScalar(s))

	// case I32:
	// 	s := int32(iv)
	// case I64:
	// 	s := int64(iv)
	// case U8:
	// 	s := byte(iv)
	case types.Tensor:
		t = iv
	default:
		err = errors.Errorf(nyiTypeFail, "repeatOp.Do()", inputs[0])
		return
	}

	// actually do repeat

	for i, axis := range op.along {
		rep := reps[i]
		if rep == 1 {
			// then no need to waste CPU
			continue
		}
		if t, err = tensor.Repeat(t, axis, rep); err != nil {
			err = errors.Wrapf(err, repFail, axis, rep)
			return
		}
	}
	retVal = t
	return
}

func (op repeatOp) WriteHash(h hash.Hash) {
	h.Write([]byte("repeat"))
	if err := binary.Write(h, binary.LittleEndian, byte(op.d)); err != nil {
		panic(err)
	}

	fmt.Fprintf(h, "%v", op.along)
	if err := binary.Write(h, binary.LittleEndian, byte(op.children)); err != nil {
		panic(err)
	}

	if op.arg0Dim == 0 {
		h.Write([]byte{1})
	} else {
		h.Write([]byte{0})
	}
}

func (op repeatOp) Hashcode() uint32 {
	h := fnv.New32a()
	op.WriteHash(h)
	return h.Sum32()
}

// sliceOp represents a slicing operation. If end <= start, it means ":"
type sliceOp struct {
	types.Slice

	along int // along which axis to slice?

	a int // along which axis of the original tensor
	d int // how many dimensions were the original tensor
}

func newSliceOp(s types.Slice, along, d int) *sliceOp {
	return &sliceOp{
		Slice: s,
		along: along,
		d:     d,
	}
}

func (op *sliceOp) Arity() int { return 1 }

// slicing a tensor value T[:] has type
// 		slice :: Tensor a → Tensor a
// 		slice :: Tensor a → a
//
// The latter is in the case where the resulting dimensions is 0, returning a scalar
func (op *sliceOp) Type() hm.Type {
	a := hm.TypeVariable('a')
	tt := newTensorType(op.d, a)

	var selection int

	if op.Slice == nil {
		selection = -1
	} else {
		selection = op.End() - op.Start()
	}

	if selection == 1 {
		if op.d == 1 {
			return hm.NewFnType(tt, a)
		}

		tt2 := newTensorType(op.d-1, a)
		return hm.NewFnType(tt, tt2)
	}

	return hm.NewFnType(tt, tt)
}

func (op *sliceOp) InferShape(inputs ...DimSizer) (s types.Shape, err error) {
	input := inputs[0].(types.Shape)
	slices := make([]types.Slice, op.along+1)
	slices[op.along] = op.Slice

	return input.S(slices...)

	// return input.S(op.Slice)
}

func (op *sliceOp) DiffWRT(i int) []bool {
	if i > 1 {
		// error
		err := errors.Errorf("sliceOp should only have one or more inputs. Got %v instead", i)
		panic(err)
	}

	return []bool{true}
}

func (op *sliceOp) SymDiff(inputs Nodes, outputNode, gradNode *Node) (retVal Nodes, err error) {
	if err = checkArity(op, len(inputs)); err != nil {
		return
	}

	t := inputs[0]
	incrOp := sliceIncrOp{op}

	retVal = make(Nodes, 1)
	retVal[0], err = applyOp(incrOp, t, gradNode)
	return
}

func (op *sliceOp) DoDiff(inputs Nodes, output *Node) (err error) {
	if err = checkArity(op, len(inputs)); err != nil {
		return
	}

	xdv := inputs[0].boundTo.(*dualValue)
	ydv := output.boundTo.(*dualValue)
	incrOp := sliceIncrOp{op}

	var d Value
	if d, err = incrOp.Do(xdv.Value, ydv.d); err != nil {
		return errors.Wrapf(err, doFail, incrOp)
	}

	// there is no need to handle scalars, because you can never slice a scalar
	add := newElemBinOp(addOpType, inputs[0], output)
	if _, err = add.UnsafeDo(xdv.d, d); err != nil {
		return errors.Wrapf(err, unsafeDoFail, add)
	}

	return
}

func (op *sliceOp) Do(inputs ...Value) (retVal Value, err error) {
	if err = checkArity(op, len(inputs)); err != nil {
		return
	}

	t := inputs[0]
	// prep the slices
	var slices []types.Slice
	slices = make([]types.Slice, len(t.Shape()))

	if !op.all() {
		slices[op.along] = op
	}
	switch T := t.(type) {
	case types.Tensor:
		switch tt := T.(type) {
		case *tf64.Tensor:
			// actually do shit
			var v64 *tf64.Tensor // it's a view though
			if v64, err = tt.Slice(slices...); err != nil {
				return nil, errors.Wrapf(err, sliceFail, slices)
			}

			// prep retVal
			if v64.IsScalar() {
				retVal, _ = anyToScalar(v64.ScalarValue())
			} else {
				retVal = v64
			}
		case *tf32.Tensor:
			// actually do shit
			var v32 *tf32.Tensor // it's a view though
			if v32, err = tt.Slice(slices...); err != nil {
				return nil, errors.Wrapf(err, sliceFail, slices)
			}

			// prep retVal
			if v32.IsScalar() {
				retVal, _ = anyToScalar(v32.ScalarValue())
			} else {
				retVal = v32
			}
		case *ti.Tensor:
			// actually do shit
			var vi *ti.Tensor // it's a view though
			if vi, err = tt.Slice(slices...); err != nil {
				return nil, errors.Wrapf(err, sliceFail, slices)
			}

			// prep retVal
			if vi.IsScalar() {
				retVal, _ = anyToScalar(vi.ScalarValue())
			} else {
				retVal = vi
			}
		// case *tb.Tensor:
		default:
			return nil, errors.Errorf(nyiFail, "sliceOp.Do()", T)
		}
	case Scalar:
		return nil, errors.New("Cannot slice a scalar value")
	default:
		return nil, errors.Errorf(nyiFail, "sliceOp.Do()", t)
	}
	return
}

func (op *sliceOp) ReturnsPtr() bool     { return true }
func (op *sliceOp) CallsExtern() bool    { return false }
func (op *sliceOp) OverwritesInput() int { return -1 }
func (op *sliceOp) WriteHash(h hash.Hash) {
	h.Write([]byte("slice"))
	if err := binary.Write(h, binary.LittleEndian, byte(op.d)); err != nil {
		panic(err)
	}
	fmt.Fprintf(h, "%v", op.along)
	if op.Slice == nil {
		fmt.Fprintf(h, ":")
		return
	}

	if err := binary.Write(h, binary.LittleEndian, byte(op.Start())); err != nil {
		panic(err)
	}
	if err := binary.Write(h, binary.LittleEndian, byte(op.End())); err != nil {
		panic(err)
	}
	if err := binary.Write(h, binary.LittleEndian, byte(op.Step())); err != nil {
		panic(err)
	}

}
func (op sliceOp) Hashcode() uint32 {
	h := fnv.New32a()
	op.WriteHash(h)
	return h.Sum32()
}

func (op sliceOp) String() string {
	var buf bytes.Buffer
	buf.WriteString("T[")
	for i := 0; i < op.along; i++ {
		buf.WriteString(":, ")
	}

	if op.all() {
		buf.WriteString(":")
	} else {
		fmt.Fprintf(&buf, "%d:%d:%d", op.Start(), op.End(), op.Step())
	}

	buf.WriteString("...]")
	return buf.String()
}

func (op sliceOp) all() bool { return op.Slice == nil || op.End() <= op.Start() }

// T[:] +=incr
// THIS IS AN UNSAFE OPERATION
type sliceIncrOp struct {
	*sliceOp
}

// slicing a tensor value T[:] has type
// 		slice :: Tensor a → b → Tensor a
//
// b can be a or Vector a
func (op sliceIncrOp) Type() hm.Type {
	a := hm.TypeVariable('a')
	b := hm.TypeVariable('c')
	tt := newTensorType(op.d, a)

	retVal := hm.NewFnType(tt, b, tt)
	return retVal
}

func (op sliceIncrOp) Arity() int { return 2 }

func (op sliceIncrOp) InferShape(inputs ...DimSizer) (retVal types.Shape, err error) {
	retVal = inputs[0].(types.Shape)
	return
}

func (op sliceIncrOp) DiffWRT(i int) []bool {
	if err := checkArity(op, i); err != nil {
		panic(err)
	}

	return []bool{true, false}
}

func (op sliceIncrOp) SymDiff(inputs Nodes, outputNode, gradNode *Node) (retVal Nodes, err error) {
	var slicedRes *Node
	if slicedRes, err = applyOp(op.sliceOp, gradNode); err != nil {
		return nil, errors.Wrap(err, operationError)
	}
	retVal = Nodes{gradNode, slicedRes}
	return
}

func (op sliceIncrOp) DoDiff(inputs Nodes, output *Node) (err error) {
	xdv := inputs[0].boundTo.(*dualValue)
	ydv := inputs[1].boundTo.(*dualValue)
	zdv := output.boundTo.(*dualValue)

	// dzdx
	add := newElemBinOp(addOpType, inputs[0], output)

	if _, err = add.UnsafeDo(xdv.d, zdv.d); err != nil {
		return errors.Wrapf(err, unsafeDoFail, add)
	}

	// dzdy
	var d Value
	if d, err = op.sliceOp.Do(zdv.d); err != nil {
		return errors.Wrapf(err, doFail, op)
	}

	add = newElemBinOp(addOpType, inputs[1], output)
	if _, err = add.UnsafeDo(ydv.d, d); err != nil {
		return errors.Wrapf(err, doFail, add)
	}
	return
}

func (op sliceIncrOp) Do(inputs ...Value) (retVal Value, err error) {
	machineLogf("Doing %v", op)
	enterLoggingContext()
	defer leaveLoggingContext()

	if err = checkArity(op, len(inputs)); err != nil {
		return
	}

	t := inputs[0]
	incr := inputs[1]

	// prep the slices
	slices := make([]types.Slice, op.d)
	if !op.all() {
		slices[op.along] = op
	}

	switch T := t.(type) {
	case types.Tensor:
		switch tt := T.(type) {
		case *tf64.Tensor:
			// actually do shit
			cloned := tf64.NewTensor(tf64.WithShape(tt.Shape()...))
			var v64 *tf64.Tensor
			if v64, err = cloned.Slice(slices...); err != nil {
				return nil, errors.Wrapf(err, sliceFail, slices)
			}

			var val interface{}
			switch i := incr.(type) {
			case F64:
				val = float64(i)
			case *tf64.Tensor:
				val = i
			default:
				err = errors.Errorf("Incr is of %T. Cannot increment on input which is a *tf64.Tensor", incr)
				return
			}

			v64.VAdd(val)
			retVal = cloned

		case *tf32.Tensor:
			// actually do shit
			cloned := tf32.NewTensor(tf32.WithShape(tt.Shape()...))
			var v32 *tf32.Tensor
			if v32, err = cloned.Slice(slices...); err != nil {
				return nil, errors.Wrapf(err, sliceFail, slices)
			}

			var val interface{}
			switch i := incr.(type) {
			case F32:
				val = float32(i)
			case *tf32.Tensor:
				val = i
			default:
				err = errors.Errorf("Incr is of %T. Cannot increment on input which is a *tf32.Tensor", incr)
				return
			}

			v32.VAdd(val)
			retVal = cloned
		case *ti.Tensor:
			// actually do shit
			cloned := ti.NewTensor(ti.WithShape(tt.Shape()...))
			var vi *ti.Tensor
			if vi, err = cloned.Slice(slices...); err != nil {
				return nil, errors.Wrapf(err, sliceFail, slices)
			}

			var val interface{}
			switch i := incr.(type) {
			case I:
				val = int(i)
			case *ti.Tensor:
				val = i
			default:
				err = errors.Errorf("Incr is of %T. Cannot increment on input which is a *ti.Tensor", incr)
				return
			}

			vi.VAdd(val)
			retVal = cloned
		// case *tb.Tensor:
		default:
			return nil, errors.Errorf(nyiFail, "sliceIncrOp()", T)
		}
	case Scalar:
		return nil, errors.New("Cannot slice a scalar value")
	default:
		return nil, errors.Errorf(nyiFail, "sliceIncrOp()", t)
	}
	return
}

func (op sliceIncrOp) OverwritesInput() int { return 0 }

func (op sliceIncrOp) WriteHash(h hash.Hash) {
	h.Write([]byte("sliceIncr"))
	if err := binary.Write(h, binary.LittleEndian, byte(op.d)); err != nil {
		panic(err)
	}
	if err := binary.Write(h, binary.LittleEndian, byte(op.along)); err != nil {
		panic(err)
	}

	if op.Slice == nil {
		fmt.Fprintf(h, ":")
		return
	}

	if err := binary.Write(h, binary.LittleEndian, byte(op.Start())); err != nil {
		panic(err)
	}
	if err := binary.Write(h, binary.LittleEndian, byte(op.End())); err != nil {
		panic(err)
	}
	if err := binary.Write(h, binary.LittleEndian, byte(op.Step())); err != nil {
		panic(err)
	}
}

func (op sliceIncrOp) Hashcode() uint32 {
	h := fnv.New32a()
	op.WriteHash(h)
	return h.Sum32()
}

func (op sliceIncrOp) String() string {
	var buf bytes.Buffer
	buf.WriteString("T[")
	for i := 0; i < op.along; i++ {
		buf.WriteString(":, ")
	}

	if op.all() {
		buf.WriteString(":")
	} else {
		fmt.Fprintf(&buf, "%d:%d:%d", op.Start(), op.End(), op.Step())
	}

	buf.WriteString("...]+=...")
	return buf.String()
}

// func (op sliceIncrOp) UsePreallocDo(val Value, inputs ...Value) (Value, error) {

// }

type transposeOp struct {
	pattern []int
	d       int
}

func (op transposeOp) Arity() int { return 1 }

// transposing a tensor has type
// 		transpose :: Tensor a → Tensor a
func (op transposeOp) Type() hm.Type {
	a := hm.TypeVariable('a')
	tt := newTensorType(op.d, a)

	return hm.NewFnType(tt, tt)
}

func (op transposeOp) InferShape(inputs ...DimSizer) (retVal types.Shape, err error) {
	input := inputs[0].(types.Shape)
	if input.IsScalar() {
		return nil, errors.Errorf(undefinedOnShape, op, input)
	}

	retVal = make(types.Shape, len(input))
	copy(retVal, input)
	err = types.UnsafePermute(op.pattern, retVal)
	return
}

func (op transposeOp) DiffWRT(i int) []bool {
	if err := checkArity(op, i); err != nil {
		panic(err)
	}

	return []bool{true}
}

func (op transposeOp) SymDiff(inputs Nodes, outputNode, gradNode *Node) (retVal Nodes, err error) {
	newPattern := make([]int, len(op.pattern))
	for i, p := range op.pattern {
		newPattern[p] = i
	}
	op2 := transposeOp{pattern: newPattern, d: op.d}

	retVal = make(Nodes, 1)
	retVal[0], err = applyOp(op2, gradNode)
	return
}

func (op transposeOp) DoDiff(inputs Nodes, output *Node) (err error) {
	xdv := inputs[0].boundTo.(*dualValue)
	zdv := output.boundTo.(*dualValue)

	newPattern := make([]int, len(op.pattern))
	for i, p := range op.pattern {
		newPattern[p] = i
	}

	var zdvdT types.Tensor
	var ok bool
	if zdvdT, ok = zdv.d.(types.Tensor); !ok {
		return errors.Errorf("Expected the gradient of the output node to be a Tensor. Got %v instead", zdv.d)
	}

	if err = zdvdT.T(newPattern...); err != nil {
		return errors.Wrap(err, "Failed to T()")
	}

	d := zdvdT.Materialize()
	zdvdT.UT()

	add := newEBOByType(addOpType, inputs[0].t, TypeOf(zdvdT))
	if _, err = add.UnsafeDo(xdv.d, d); err != nil {
		err = errors.Wrapf(err, doFail, add)
	}
	return
}

func (op transposeOp) Do(inputs ...Value) (retVal Value, err error) {
	machineLogf("Doing %v", op)
	enterLoggingContext()
	defer leaveLoggingContext()

	if err = checkArity(op, len(inputs)); err != nil {
		return
	}

	t := inputs[0].(types.Tensor)

	throwaway := types.BorrowInts(len(op.pattern))
	copy(throwaway, op.pattern)
	return tensor.T(t, throwaway...)

	// DEPRECATED
	// the reason for this is because the .T() method of a Tensor
	// will use the axes in the .transposedWith field
	// Later when .UT() is called, the .transposedWith field is recycled into the pool
	// throwaway := types.BorrowInts(len(op.pattern))
	// copy(throwaway, op.pattern)

	// t.T(throwaway...)
	// ret := t.Materialize()
	// t.UT()
}

func (op transposeOp) ReturnsPtr() bool     { return true }
func (op transposeOp) CallsExtern() bool    { return false }
func (op transposeOp) OverwritesInput() int { return 0 }

func (op transposeOp) WriteHash(h hash.Hash) {
	h.Write([]byte("transposeOp"))
	fmt.Fprintf(h, "%v", op.pattern)
	if err := binary.Write(h, binary.LittleEndian, byte(op.d)); err != nil {
		panic(err)
	}
}

func (op transposeOp) Hashcode() uint32 {
	h := fnv.New32a()
	op.WriteHash(h)
	return h.Sum32()
}

func (op transposeOp) String() string {
	var buf bytes.Buffer
	buf.WriteString("Aᵀ{")
	for i, ax := range op.pattern {
		fmt.Fprintf(&buf, "%d", ax)
		if i < len(op.pattern)-1 {
			buf.WriteString(", ")
		}
	}

	buf.WriteString("}")
	return buf.String()
}

type concatOp struct {
	axis     int
	d        int
	children int
}

func (op concatOp) Arity() int { return -1 }

// concat only works for Tensor types
//		concat :: Tensor a → Tensor a → ... → Tensor a
func (op concatOp) Type() hm.Type {
	tt := newTensorType(op.d, hm.TypeVariable('a'))
	fnt := make([]hm.Type, op.children+1)
	for i := range fnt {
		fnt[i] = tt
	}

	return hm.NewFnType(fnt...)
}

func (op concatOp) InferShape(ds ...DimSizer) (types.Shape, error) {
	if len(ds) == 0 {
		return nil, errors.Errorf("No shapes passed in!")
	}
	shapes, err := DimSizersToShapes(ds)
	if err != nil {
		return nil, err
	}

	return shapes[0].Concat(op.axis, shapes[1:]...)
}

func (op concatOp) Do(vals ...Value) (Value, error) {
	if len(vals) == 1 {
		return vals[0], nil
	}

	ts, err := valuesToTensors(vals)
	if err != nil {
		return nil, err
	}

	return tensor.Concat(op.axis, ts[0], ts[1:]...)
}

func (op concatOp) ReturnsPtr() bool     { return true }
func (op concatOp) CallsExtern() bool    { return false }
func (op concatOp) OverwritesInput() int { return -1 }

func (op concatOp) WriteHash(h hash.Hash) {
	h.Write([]byte("concatOp"))
	fmt.Fprintf(h, "axis: %d, dims: %d", op.axis, op.d)
}

func (op concatOp) Hashcode() uint32 {
	h := fnv.New32a()
	op.WriteHash(h)
	return h.Sum32()
}

func (op concatOp) String() string {
	return fmt.Sprintf("Concat(axis=%d)", op.axis)
}
func (op concatOp) DiffWRT(inputs int) []bool {
	retVal := make([]bool, inputs)
	for i := range retVal {
		retVal[i] = true
	}
	return retVal
}

func (op concatOp) SymDiff(inputs Nodes, output *Node, grad *Node) (retVal Nodes, err error) {
	var start int

	retVal = make(Nodes, len(inputs))
	for i, in := range inputs {
		if op.axis >= len(in.shape) {
			return nil, errors.Errorf("Wanted dimension %d is larger than the shape %v", op.axis, in.shape)
		}
		end := in.shape[op.axis] + start

		s := newSliceOp(S(start, end), op.axis, op.d)
		if retVal[i], err = applyOp(s, grad); err != nil {
			return
		}
		start = end
	}
	return
}

func (op concatOp) DoDiff(inputs Nodes, output *Node) error {
	odv := output.boundTo.(*dualValue)
	odvd := odv.d.(types.Tensor)

	var start int
	for _, in := range inputs {
		if op.axis >= len(in.shape) {
			return errors.Errorf("Wanted dimension %d is larger than the shape %v", op.axis, in.shape)
		}
		end := in.shape[op.axis] + start

		idv := in.boundTo.(*dualValue)
		idvd := idv.d.(types.Tensor)

		sliced, err := tensor.Slice(odvd, S(start, end))
		if err != nil {
			return err
		}

		// TODO: fix VAdd hack
		// add to odvd
		switch st := sliced.(type) {
		case *tf64.Tensor:
			d := idvd.(*tf64.Tensor)
			d.VAdd(st)
		case *tf32.Tensor:
			d := idvd.(*tf32.Tensor)
			d.VAdd(st)
		case *ti.Tensor:
			d := idvd.(*ti.Tensor)
			d.VAdd(st)
		default:
			return errors.Errorf(nyiTypeFail, "DoDiff (hack) ", st)
		}

		start = end
	}
	return nil
}
