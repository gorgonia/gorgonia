package tensor

import (
	"log"

	"github.com/pkg/errors"
)

// Apply applies a function to all the values in the ndarray
func (t *Dense) Apply(fn interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	var e Engine = t.e
	if e == nil {
		e = StdEng{}
	}
	if m, ok := e.(Mapper); ok {
		return m.Map(fn, t, opts...)
	}
	return nil, errors.Errorf("Execution engine for %v not a mapper", t)
}

// T performs a thunked transpose. It doesn't actually do anything, except store extra information about the post-transposed shapes and strides
// Usually this is more than enough, as BLAS will handle the rest of the transpose
func (t *Dense) T(axes ...int) (err error) {
	var transform *AP
	if transform, axes, err = t.AP.T(axes...); err != nil {
		return handleNoOp(err)
	}

	// is there any old transposes that need to be done first?
	// this is important, because any old transposes for dim >=3 are merely permutations of the strides
	if t.old != nil {
		if t.IsVector() {
			// the transform that was calculated was a waste of time - return it to the pool then untranspose
			ReturnAP(transform)
			t.UT()
			return
		}

		// check if the current axes are just a reverse of the previous transpose's
		isReversed := true
		for i, s := range t.oshape() {
			if transform.Shape()[i] != s {
				isReversed = false
				break
			}
		}

		// if it is reversed, well, we just restore the backed up one
		if isReversed {
			ReturnAP(transform)
			t.UT()
			return
		}

		// cool beans. No funny reversals. We'd have to actually do transpose then
		t.Transpose()
	}

	// swap out the old and the new
	t.old = t.AP
	t.transposeWith = axes
	t.AP = transform
	return nil
}

// UT is a quick way to untranspose a currently transposed *Dense
// The reason for having this is quite simply illustrated by this problem:
//		T = NewTensor(WithShape(2,3,4))
//		T.T(1,2,0)
//
// To untranspose that, we'd need to apply a transpose of (2,0,1).
// This means having to keep track and calculate the transposes.
// Instead, here's a helpful convenience function to instantly untranspose any previous transposes.
//
// Nothing will happen if there was no previous transpose
func (t *Dense) UT() {
	if t.old != nil {
		ReturnAP(t.AP)
		ReturnInts(t.transposeWith)
		t.AP = t.old
		t.old = nil
		t.transposeWith = nil
	}
}

// SafeT is exactly like T(), except it returns a new *Dense. The data is also copied over, unmoved.
func (t *Dense) SafeT(axes ...int) (retVal *Dense, err error) {
	var transform *AP
	if transform, axes, err = t.AP.T(axes...); err != nil {
		if err = handleNoOp(err); err != nil {
			return
		}
	}

	retVal = recycledDense(t.t, Shape{t.len()})
	copyDense(retVal, t)

	retVal.AP = transform
	retVal.old = t.AP.Clone()
	retVal.transposeWith = axes

	return
}

// Transpose() actually transposes the data.
// This is a generalized version of the inplace matrix transposition algorithm from Wikipedia:
// https://en.wikipedia.org/wiki/In-place_matrix_transposition
func (t *Dense) Transpose() error {
	// if there is no oldinfo, that means the current info is the latest, and not the transpose
	if t.old == nil {
		return nil
	}

	if t.IsScalar() {
		return nil // cannot transpose scalars - no data movement
	}

	defer func() {
		ReturnAP(t.old)
		t.old = nil
		t.transposeWith = nil
	}()

	expShape := t.Shape()

	// important! because the strides would have changed once the underlying data changed
	var expStrides []int
	if t.AP.o.isColMajor() {
		expStrides = expShape.calcStridesColMajor()
	} else {
		expStrides = expShape.calcStrides()
	}
	defer ReturnInts(expStrides)
	defer func() {
		t.setShape(expShape...)
		t.sanity()
	}()

	if t.IsVector() {
		// no data movement
		return nil
	}

	// actually move data
	var e Engine = t.e
	if e == nil {
		e = StdEng{}
	}
	transposer, ok := e.(Transposer)
	if !ok {
		return errors.Errorf("Engine does not support Transpose()")
	}
	return transposer.Transpose(t, expStrides)
}

// At returns the value at the given coordinate
func (t *Dense) At(coords ...int) (interface{}, error) {
	if !t.IsNativelyAccessible() {
		return nil, errors.Errorf(inaccessibleData, t)
	}
	if len(coords) != t.Dims() {
		return nil, errors.Errorf(dimMismatch, t.Dims(), len(coords))
	}

	at, err := t.at(coords...)
	if err != nil {
		return nil, errors.Wrap(err, "At()")
	}

	return t.Get(at), nil
}

// MaskAt returns the value of the mask at a given coordinate
// returns false (valid) if not tensor is not masked
func (t *Dense) MaskAt(coords ...int) (bool, error) {
	if !t.IsMasked() {
		return false, nil
	}
	if !t.IsNativelyAccessible() {
		return false, errors.Errorf(inaccessibleData, t)
	}
	if len(coords) != t.Dims() {
		return true, errors.Errorf(dimMismatch, t.Dims(), len(coords))
	}

	at, err := t.maskAt(coords...)
	if err != nil {
		return true, errors.Wrap(err, "MaskAt()")
	}

	return t.mask[at], nil
}

// SetAt sets the value at the given coordinate
func (t *Dense) SetAt(v interface{}, coords ...int) error {
	if !t.IsNativelyAccessible() {
		return errors.Errorf(inaccessibleData, t)
	}

	if len(coords) != t.Dims() {
		return errors.Errorf(dimMismatch, t.Dims(), len(coords))
	}

	at, err := t.at(coords...)
	if err != nil {
		return errors.Wrap(err, "SetAt()")
	}
	t.Set(at, v)
	return nil
}

// SetMaskAtDataIndex set the value of the mask at a given index
func (t *Dense) SetMaskAtIndex(v bool, i int) error {
	if !t.IsMasked() {
		return nil
	}
	t.mask[i] = v
	return nil
}

// SetMaskAt sets the mask value at the given coordinate
func (t *Dense) SetMaskAt(v bool, coords ...int) error {
	if !t.IsMasked() {
		return nil
	}
	if !t.IsNativelyAccessible() {
		return errors.Errorf(inaccessibleData, t)
	}
	if len(coords) != t.Dims() {
		return errors.Errorf(dimMismatch, t.Dims(), len(coords))
	}

	at, err := t.maskAt(coords...)
	if err != nil {
		return errors.Wrap(err, "SetAt()")
	}
	t.mask[at] = v
	return nil
}

// Repeat is like Numpy's repeat. It repeats the elements of an array.
// The repeats param defines how many times each element in the axis is repeated.
// Just like NumPy, the repeats param is broadcasted to fit the size of the given axis.
func (t *Dense) Repeat(axis int, repeats ...int) (retVal Tensor, err error) {
	e := t.Engine()
	if e == nil {
		e = StdEng{}
	}
	if rp, ok := e.(Repeater); ok {
		return rp.Repeat(t, axis, repeats...)
	}
	return nil, errors.New("Engine does not support Repeat")
}

// CopyTo copies the underlying data to the destination *Dense. The original data is untouched.
// Note: CopyTo doesn't care about the metadata of the destination *Dense. Take for example:
//		T = NewTensor(WithShape(6))
//		T2 = NewTensor(WithShape(2,3))
//		err = T.CopyTo(T2) // err == nil
//
// The only time that this will fail is if the underlying sizes are different
func (t *Dense) CopyTo(other *Dense) error {
	if other == t {
		return nil // nothing to copy to. Maybe return NoOpErr?
	}

	if other.Size() != t.Size() {
		return errors.Errorf(sizeMismatch, t.Size(), other.Size())
	}

	// easy peasy lemon squeezy
	if t.viewOf == nil && other.viewOf == nil {
		copyDense(other, t)
		return nil
	}

	return errors.Errorf(methodNYI, "CopyTo", "views")
}

// Slice performs slicing on the *Dense Tensor. It returns a view which shares the same underlying memory as the original *Dense.
//
// Given:
//		T = NewTensor(WithShape(2,2), WithBacking(RangeFloat64(0,4)))
//		V, _ := T.Slice(nil, singleSlice(1)) // T[:, 1]
//
// Any modification to the values in V, will be reflected in T as well.
//
// The method treats <nil> as equivalent to a colon slice. T.Slice(nil) is equivalent to T[:] in Numpy syntax
func (t *Dense) Slice(slices ...Slice) (retVal View, err error) {
	var newAP *AP
	var ndStart, ndEnd int

	if newAP, ndStart, ndEnd, err = t.AP.S(t.len(), slices...); err != nil {
		return
	}

	view := new(Dense)
	view.t = t.t
	view.e = t.e
	view.flag = t.flag
	view.viewOf = t
	view.AP = newAP
	view.array = t.array
	view.slice(ndStart, ndEnd)

	if t.IsMasked() {
		view.mask = t.mask[ndStart:ndEnd]
	}
	return view, err
}

// RollAxis rolls the axis backwards until it lies in the given position.
//
// This method was adapted from Numpy's Rollaxis. The licence for Numpy is a BSD-like licence and can be found here: https://github.com/numpy/numpy/blob/master/LICENSE.txt
//
// As a result of being adapted from Numpy, the quirks are also adapted. A good guide reducing the confusion around rollaxis can be found here: http://stackoverflow.com/questions/29891583/reason-why-numpy-rollaxis-is-so-confusing (see answer by hpaulj)
func (t *Dense) RollAxis(axis, start int, safe bool) (retVal *Dense, err error) {
	dims := t.Dims()

	if !(axis >= 0 && axis < dims) {
		err = errors.Errorf(invalidAxis, axis, dims)
		return
	}

	if !(start >= 0 && start <= dims) {
		err = errors.Wrap(errors.Errorf(invalidAxis, axis, dims), "Start axis is wrong")
		return
	}

	if axis < start {
		start--
	}

	if axis == start {
		retVal = t
		return
	}

	axes := BorrowInts(dims)
	defer ReturnInts(axes)

	for i := 0; i < dims; i++ {
		axes[i] = i
	}
	copy(axes[axis:], axes[axis+1:])
	copy(axes[start+1:], axes[start:])
	axes[start] = axis

	if safe {
		return t.SafeT(axes...)
	}
	err = t.T(axes...)
	retVal = t
	return
}

// Concat concatenates the other tensors along the given axis. It is like Numpy's concatenate() function.
func (t *Dense) Concat(axis int, Ts ...*Dense) (retVal *Dense, err error) {
	ss := make([]Shape, len(Ts))

	var isMasked = false
	for i, T := range Ts {
		ss[i] = T.Shape()
		isMasked = isMasked || T.IsMasked()
	}

	var newShape Shape
	if newShape, err = t.Shape().Concat(axis, ss...); err != nil {
		return
	}
	retVal = recycledDense(t.t, newShape)
	if isMasked {
		retVal.makeMask()
	}

	all := make([]*Dense, len(Ts)+1)
	all[0] = t
	copy(all[1:], Ts)

	// special case
	var start, end int

	for _, T := range all {
		end += T.Shape()[axis]
		slices := make([]Slice, axis+1)
		slices[axis] = makeRS(start, end)

		var v *Dense
		if v, err = sliceDense(retVal, slices...); err != nil {
			return
		}

		if v.IsVector() && T.IsMatrix() && axis == 0 {
			v.reshape(v.shape[0], 1)
		}

		if err = assignArray(v, T); err != nil {
			return
		}
		start = end
	}

	return
}

// Hstack stacks other tensors columnwise (horizontal stacking)
func (t *Dense) Hstack(others ...*Dense) (*Dense, error) {
	// check that everything is at least 1D
	if t.Dims() == 0 {
		return nil, errors.Errorf(atleastDims, 1)
	}

	for _, d := range others {
		if d.Dims() < 1 {
			return nil, errors.Errorf(atleastDims, 1)
		}
	}

	if t.Dims() == 1 {
		return t.Concat(0, others...)
	}
	return t.Concat(1, others...)
}

// Vstack stacks other tensors rowwise (vertical stacking). Vertical stacking requires all involved Tensors to have at least 2 dimensions
func (t *Dense) Vstack(others ...*Dense) (*Dense, error) {
	// check that everything is at least 2D
	if t.Dims() < 2 {
		return nil, errors.Errorf(atleastDims, 2)
	}

	for _, d := range others {
		if d.Dims() < 2 {
			return nil, errors.Errorf(atleastDims, 2)
		}
	}
	return t.Concat(0, others...)
}

// Stack stacks the other tensors along the axis specified. It is like Numpy's stack function.
func (t *Dense) Stack(axis int, others ...*Dense) (retVal *Dense, err error) {
	opdims := t.Dims()
	if axis >= opdims+1 {
		err = errors.Errorf(dimMismatch, opdims+1, axis)
		return
	}

	newShape := Shape(BorrowInts(opdims + 1))
	newShape[axis] = len(others) + 1
	shape := t.Shape()
	var cur int
	for i, s := range shape {
		if i == axis {
			cur++
		}
		newShape[cur] = s
		cur++
	}

	var newStrides []int
	if t.AP.o.isColMajor() {
		newStrides = newShape.calcStridesColMajor()
	} else {
		newStrides = newShape.calcStrides()

	}
	ap := NewAP(newShape, newStrides)
	ap.o = t.AP.o
	ap.Δ = t.AP.Δ

	allNoMat := !t.IsMaterializable()
	for _, ot := range others {
		if allNoMat && ot.IsMaterializable() {
			allNoMat = false
		}
	}

	retVal = recycledDense(t.t, ap.Shape())
	ReturnAP(retVal.AP)
	retVal.AP = ap

	// the "viewStack" method is the more generalized method
	// and will work for all Tensors, regardless of whether it's a view
	// But the simpleStack is faster, and is an optimization

	if allNoMat {
		retVal = t.simpleStack(retVal, axis, others...)
	} else {
		retVal = t.viewStack(retVal, axis, others...)
	}
	return
}

/* Private Methods */

// returns the new index given the old index
func (t *Dense) transposeIndex(i int, transposePat, strides []int) int {
	oldCoord, err := Itol(i, t.oshape(), t.ostrides())
	if err != nil {
		panic(err)
	}

	/*
		coordss, _ := Permute(transposePat, oldCoord)
		coords := coordss[0]
		expShape := t.Shape()
		index, _ := Ltoi(expShape, strides, coords...)
	*/

	// The above is the "conceptual" algorithm.
	// Too many checks above slows things down, so the below is the "optimized" edition
	var index int
	for i, axis := range transposePat {
		index += oldCoord[axis] * strides[i]
	}
	return index
}

// at returns the index at which the coordinate is referring to.
// This function encapsulates the addressing of elements in a contiguous block.
// For a 2D ndarray, ndarray.at(i,j) is
//		at = ndarray.strides[0]*i + ndarray.strides[1]*j
// This is of course, extensible to any number of dimensions.
func (t *Dense) at(coords ...int) (at int, err error) {
	return Ltoi(t.Shape(), t.Strides(), coords...)
}

// maskat returns the mask index at which the coordinate is referring to.
func (t *Dense) maskAt(coords ...int) (at int, err error) {
	//TODO: Add check for non-masked tensor
	return t.at(coords...)
}

// simpleStack is the data movement function for non-view tensors. What it does is simply copy the data according to the new strides
func (t *Dense) simpleStack(retVal *Dense, axis int, others ...*Dense) *Dense {
	switch axis {
	case 0:
		copyDense(retVal, t)
		next := t.len()
		for _, ot := range others {
			copyDenseSliced(retVal, next, retVal.len(), ot, 0, ot.len())
			next += ot.len()
		}
	default:
		axisStride := retVal.AP.Strides()[axis]
		batches := retVal.len() / axisStride

		destStart := 0
		start := 0
		end := start + axisStride

		for i := 0; i < batches; i++ {
			copyDenseSliced(retVal, destStart, retVal.len(), t, start, end)
			for _, ot := range others {
				destStart += axisStride
				copyDenseSliced(retVal, destStart, retVal.len(), ot, start, end)
				i++
			}
			destStart += axisStride
			start += axisStride
			end += axisStride
		}
	}
	return retVal
}

// viewStack is the data movement function for Stack(), applied on views
func (t *Dense) viewStack(retVal *Dense, axis int, others ...*Dense) *Dense {
	axisStride := retVal.AP.Strides()[axis]
	batches := retVal.len() / axisStride

	it := NewFlatIterator(t.AP)
	ch := it.Chan()
	chs := make([]chan int, len(others))
	chs = chs[:0]
	for _, ot := range others {
		oter := NewFlatIterator(ot.AP)
		chs = append(chs, oter.Chan())
	}

	log.Printf("Chan %d, %d", len(ch), cap(ch))
	t.doViewStack(retVal, axisStride, batches, ch, others, chs)
	return retVal
}
