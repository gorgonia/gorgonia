package tensorb

import (
	"fmt"

	"github.com/chewxy/gorgonia/tensor/types"
	"github.com/pkg/errors"
)

/*
This file contains Tensor methods that deal with operations of a matrix/tensor.

*/

// Apply applies a function to all the values in the ndarray
func (t *Tensor) Apply(fn func(bool) bool, opts ...types.FuncOpt) (retVal *Tensor, err error) {
	safe, incr, reuse := parseSafeReuse(opts...)

	// check reuse and stuff
	var res []bool
	switch {
	case reuse != nil:
		res = reuse.data
		if len(res) != t.Size() {
			err = shapeMismatchError(t.Shape(), reuse.Shape())
			return
		}
	case !safe:
		res = t.data
	default:
		if t.IsMaterializable() {
			res = make([]bool, t.Shape().TotalSize())

		} else {
			res = make([]bool, len(t.data))
		}
	}
	// do
	switch {
	case t.viewOf == nil && !incr:
		for i, v := range t.data {
			res[i] = fn(v)
		}

	case t.viewOf != nil && !incr:
		it := types.NewFlatIterator(t.AP)
		var next, i int
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			if _, noop := err.(NoOpError); err != nil && !noop {
				return
			}
			res[i] = fn(t.data[next])
			i++
		}
		err = nil

	default:
		err = notyetimplemented("Apply not implemented for this state: isView: %t and incr: %t", t.viewOf == nil, incr)
		return
	}
	// set retVal
	switch {
	case reuse != nil:
		if err = reuseCheckShape(reuse, t.Shape()); err != nil {
			return
		}
		retVal = reuse
	case !safe:
		retVal = t
	default:
		retVal = NewTensor(WithBacking(res), WithShape(t.Shape()...))
	}
	return
}

// T performs a thunked transpose. It doesn't actually do anything, except store extra information about the post-transposed shapes and strides
// Usually this is more than enough, as BLAS will handle the rest of the transpose
func (t *Tensor) T(axes ...int) (err error) {
	var transform *types.AP
	if transform, axes, err = t.AP.T(axes...); err != nil {
		if _, ok := err.(NoOpError); !ok {
			return
		}
		err = nil
		return
	}

	// is there any old transposes that need to be done first?
	// this is important, because any old transposes for dim >=3 are merely permutations of the strides
	if t.old != nil {
		if t.IsVector() {
			// the transform that was calculated was a waste of time - return it to the pool then untranspose
			types.ReturnAP(transform)
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
			types.ReturnAP(transform)
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

// UT is a quick way to untranspose a currently transposed *Tensor
// The reason for having this is quite simply illustrated by this problem:
//		T = NewTensor(WithShape(2,3,4))
//		T.T(1,2,0)
//
// To untranspose that, we'd need to apply a transpose of (2,0,1).
// This means having to keep track and calculate the transposes.
// Instead, here's a helpful convenience function to instantly untranspose any previous transposes.
//
// Nothing will happen if there was no previous transpose
func (t *Tensor) UT() {
	if t.old != nil {
		types.ReturnAP(t.AP)
		types.ReturnInts(t.transposeWith)
		t.AP = t.old
		t.old = nil
		t.transposeWith = nil
	}
}

// SafeT is exactly like T(), except it returns a new *Tensor. The data is also copied over, unmoved.
func (t *Tensor) SafeT(axes ...int) (retVal *Tensor, err error) {
	var transform *types.AP
	if transform, axes, err = t.AP.T(axes...); err != nil {
		if _, ok := err.(NoOpError); !ok {
			return
		}
		err = nil
		return
	}

	retVal = newBorrowedTensor(len(t.data))
	copy(retVal.data, t.data)
	retVal.AP = transform
	retVal.old = t.AP.Clone()
	retVal.transposeWith = axes

	return
}

// Transpose() actually transposes the data.
// This is a generalized version of the inplace matrix transposition algorithm from Wikipedia:
// https://en.wikipedia.org/wiki/In-place_matrix_transposition
func (t *Tensor) Transpose() {
	// if there is no oldinfo, that means the current info is the latest, and not the transpose
	if t.old == nil {
		return
	}

	if t.IsScalar() {
		return // cannot transpose scalars
	}

	defer func() {
		types.ReturnAP(t.old)
		t.old = nil
		t.transposeWith = nil
	}()

	expShape := t.Shape()
	expStrides := expShape.CalcStrides() // important! because the strides would have changed once the underlying data changed
	defer types.ReturnInts(expStrides)

	size := len(t.data)
	axes := t.transposeWith

	if t.IsVector() {
		t.setShape(expShape...)
		// no change of strides.
		return
	}

	// here we'll create a bit-map -- 64 bits should be more than enough
	// (I don't expect to be dealing with matrices that are larger than 64 elements that requires transposes to be done)
	//
	// The purpose of the bit-map is to track which elements have been moved to their correct places
	//
	// To set ith bit: track |= (1 << i)
	// To check if ith bit is set: track & (1 << i)
	// To check every bit up to size is unset: (1 << size)
	//

	track := types.NewBitMap(size)
	track.Set(0)
	track.Set(size - 1) // first and last don't change

	// // we start our iteration at 1, because transposing 0 does noting.
	var saved, tmp bool
	var i int

	for i = 1; ; {
		dest := t.transposeIndex(i, axes, expStrides)

		if track.IsSet(i) && track.IsSet(dest) {
			t.data[i] = saved
			saved = false //@DEFAULTZERO

			for i < size && track.IsSet(i) {
				i++
			}

			if i >= size {
				break
			}
			continue
		}

		track.Set(i)
		tmp = t.data[i]
		t.data[i] = saved
		saved = tmp

		i = dest
	}

	t.setShape(expShape...)
	t.sanity()
}

func (t *Tensor) At(coords ...int) bool {
	if len(coords) != t.Dims() {
		panic(fmt.Sprintf("Shape Mismatch. Coordinates has %d dimensions, ndarry has %d dimensions", len(coords), t.Dims()))
	}

	at, err := t.at(coords...)
	if err != nil {
		panic(err)
	}

	return t.data[at]
}

// Repeat is like Numpy's repeat. It repeats the elements of an array.
// The repeats param defines how many times each element in the axis is repeated.
// Just like NumPy, the repeats param is broadcasted to fit the size of the given axis.
func (t *Tensor) Repeat(axis int, repeats ...int) (retVal *Tensor, err error) {
	var newShape types.Shape
	// var toBroadcast bool
	var size, newSize int

	switch {
	// special case where axis == -1, meaning for all axes
	case axis == types.AllAxes:
		size = t.Shape().TotalSize()
		newShape = types.Shape{size}
		// newShape = types.Shape(types.BorrowInts(1))
		// newShape[0] = size
		axis = 0
	case t.IsScalar():
		size = 1
		// special case for row vecs
		if axis == 1 {
			newShape = types.Shape{1, 0}
		} else {
			// other wise it gets repeated into a vanilla vector
			newShape = types.Shape{0}
		}
	// vanilla vectors will get treated as if it's a colvec if it's axis 1
	case t.IsVector() && !t.IsRowVec() && !t.IsColVec() && axis == 1:
		size = 1
		newShape = t.Shape().Clone()
		newShape = append(newShape, 1)
	default:
		size = t.Shape()[axis]
		newShape = t.Shape().Clone()
	}

	// special case to allow generic repeats
	if len(repeats) == 1 {
		rep := repeats[0]
		repeats = make([]int, size)
		for i := range repeats {
			repeats[i] = rep
		}
	}
	reps := len(repeats)
	if reps != size {
		err = types.NewError(types.ShapeMismatch, "Cannot broadcast together. Resulting shape will be at least (%d, 1). Repeats is (%d, 1)", size, reps)
		return
	}

	newSize = types.SumInts(repeats)
	newShape[axis] = newSize
	retVal = NewTensor(WithShape(newShape...))

	var outers int
	if t.IsScalar() {
		outers = 1
	} else {
		outers = types.ProdInts(t.Shape()[0:axis])
		if outers == 0 {
			outers = 1
		}
	}

	var stride, newStride int
	if newShape.IsVector() {
		stride = 1 // special case
	} else if t.IsVector() {
		stride = 1 // special case because CalcStrides() will return []int{1} as the strides for a vector
	} else {
		stride = t.ostrides()[axis]
	}

	if newShape.IsVector() {
		newStride = 1
	} else {
		newStride = retVal.ostrides()[axis]
	}

	var destStart, srcStart int
	for i := 0; i < outers; i++ {
		for j := 0; j < size; j++ {
			var tmp int
			tmp = repeats[j]

			for k := 0; k < tmp; k++ {
				if srcStart >= len(t.data) || destStart+stride > len(retVal.data) {
					break
				}
				copy(retVal.data[destStart:], t.data[srcStart:]) // TODO: maybe don't just copy wholesale?
				destStart += newStride
			}
			srcStart += stride
		}
	}

	return
}

// CopyTo copies the underlying data to the destination *Tensor. The original data is untouched.
// Note: CopyTo doesn't care about the metadata of the destination *Tensor. Take for example:
//		T = NewTensor(WithShape(6))
//		T2 = NewTensor(WithShape(2,3))
//		err = T.CopyTo(T2) // err == nil
//
// The only time that this will fail is if the underlying sizes are different
func (t *Tensor) CopyTo(other *Tensor) error {
	if other == t {
		return nil // nothing to copy to. Maybe return NoOpErr?
	}

	if other.Size() != t.Size() {
		return types.NewError(types.SizeMismatch, "Cannot copy to destination tensor. Differing sizes %d and %d", t.Size(), other.Size())
	}

	// easy peasy lemon squeezy
	if t.viewOf == nil && other.viewOf == nil {
		copy(other.data, t.data)
		return nil
	}

	return notyetimplemented("CopyTo is not yet implemented for views")
}

// Slice performs slicing on the ndarrays. It returns a view which shares the same underlying memory as the original ndarray.
// In the original design, views are read-only. However, as things have changed, views are now mutable.
//
// Example. Given:
//		T = NewTensor(WithShape(2,2), WithBacking(RangeBool(0,4)))
//		V, _ := T.Slice(nil, singleSlice(1)) // T[:, 1]
//
// Any modification to the values in V, will be reflected in T as well.
//
// The method treats <nil> as equivalent to a colon slice. T.Slice(nil) is equivalent to T[:] in Numpy syntax
func (t *Tensor) Slice(slices ...types.Slice) (view *Tensor, err error) {
	var newAP *types.AP
	var ndStart, ndEnd int
	if newAP, ndStart, ndEnd, err = t.AP.S(len(t.data), slices...); err != nil {
		return
	}

	view = new(Tensor)
	view.viewOf = t
	view.AP = newAP
	view.data = t.data[ndStart:ndEnd]
	return
}

// RollAxis rolls the axis backwards until it lies in the given position.
//
// This method was adapted from Numpy's Rollaxis. The licence for Numpy is a BSD-like licence and can be found here: https://github.com/numpy/numpy/blob/master/LICENSE.txt
//
// As a result of being adapted from Numpy, the quirks are also adapted. A good guide reducing the confusion around rollaxis can be found here: http://stackoverflow.com/questions/29891583/reason-why-numpy-rollaxis-is-so-confusing (see answer by hpaulj)
func (t *Tensor) RollAxis(axis, start int, safe bool) (retVal *Tensor, err error) {
	dims := t.Opdims()

	if !(axis >= 0 && axis < dims) {
		err = types.NewError(types.OpError, "rollaxis cannot be completed. Axis(%d) must be >= 0 and < %d", axis, dims)
		return
	}

	if !(start >= 0 && start <= dims) {
		err = types.NewError(types.OpError, "rollaxis cannot be completed. Start(%d) must be >= 0 and <= %d", start, dims)
		return
	}

	if axis < start {
		start--
	}

	if axis == start {
		retVal = t
		return
	}

	axes := types.BorrowInts(dims)
	defer types.ReturnInts(axes)

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
func (t *Tensor) Concat(axis int, Ts ...*Tensor) (retVal *Tensor, err error) {
	// check that all tensors to be concatenated have the same number of dimensions
	dims := t.Dims()
	for _, T := range Ts {
		if T.Dims() != dims {
			err = dimMismatchError(dims, T.Dims())
			return
		}
	}

	if axis < 0 {
		err = types.AxisErr("Axis %d is less than 0", axis)
		return
	}

	var newStrides []int
	newShape := types.Shape(types.BorrowInts(dims))
	copy(newShape, t.Shape())
	for _, T := range Ts {
		for d := 0; d < dims; d++ {
			if d == axis {
				newShape[d] += T.Shape()[d]
			} else {
				// validate that the rest of the dimensions match up
				if newShape[d] != T.Shape()[d] {
					err = dimMismatchError(newShape[d], T.Shape()[d])
					return
				}
			}
		}
	}

	aps := make([]*types.AP, len(Ts)+1)
	aps[0] = t.AP
	for i, T := range Ts {
		aps[i+1] = T.AP
	}

	newStrides = newShape.CalcStrides()
	data := make([]bool, newShape.TotalSize())

	retVal = new(Tensor)
	retVal.AP = types.NewAP(newShape, newStrides)
	retVal.data = data

	all := make([]*Tensor, len(Ts)+1)
	all[0] = t
	copy(all[1:], Ts)

	var start, end int
	for _, T := range all {
		end += T.Shape()[axis]
		slices := make([]types.Slice, axis+1)
		slices[axis] = makeRS(start, end)

		var v *Tensor
		if v, err = retVal.Slice(slices...); err != nil {
			return
		}
		if err = assignArray(v, T); err != nil {
			return
		}
		start = end
	}

	return
}

// Stack stacks the other tensors along the axis specified. It is like Numpy's stack function.
func (t *Tensor) Stack(axis int, others ...*Tensor) (retVal *Tensor, err error) {
	opdims := t.Opdims()
	if axis >= opdims+1 {
		err = dimMismatchError(opdims+1, axis)
		return
	}

	newShape := types.Shape(types.BorrowInts(opdims + 1))
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

	newStrides := newShape.CalcStrides()
	ap := types.NewAP(newShape, newStrides)

	allNoMat := !t.IsMaterializable()
	for _, ot := range others {
		if allNoMat && ot.IsMaterializable() {
			allNoMat = false
		}
	}

	var data []bool

	// the "viewStack" method is the more generalized method
	// and will work for all Tensors, regardless of whether it's a view
	// But the simpleStack is faster, and is an optimization
	if allNoMat {
		data = t.simpleStack(axis, ap, others...)
	} else {
		data = t.viewStack(axis, ap, others...)
	}

	retVal = new(Tensor)
	retVal.AP = ap
	retVal.data = data
	return
}

/* Private Methods */

// returns the new index given the old index
func (t *Tensor) transposeIndex(i int, transposePat, strides []int) int {
	oldCoord, err := types.Itol(i, t.oshape(), t.ostrides())
	if err != nil {
		panic(err)
	}

	/*
		coordss, _ := types.Permute(transposePat, oldCoord)
		coords := coordss[0]
		expShape := t.Shape()
		index, _ := types.Ltoi(expShape, strides, coords...)
	*/

	// The above is the "conceptual" algorithm.
	// Too many checks above slows things down, so the below is the "optimized" edition
	var index int
	for i, axis := range transposePat {
		index += oldCoord[axis] * strides[i]
	}
	return index
}

// at returns the index at which the coordinate is refering to.
// This function encapsulates the addressing of elements in a contiguous block.
// For a 2D ndarray, ndarray.at(i,j) is
//		at = ndarray.strides[0]*i + ndarray.strides[1]*j
// This is of course, extensible to any number of dimensions.
func (t *Tensor) at(coords ...int) (at int, err error) {
	return types.Ltoi(t.Shape(), t.Strides(), coords...)
}

// iToCoord is the inverse function of at().
func (t *Tensor) itol(i int) (coords []int, err error) {
	var oShape types.Shape
	var oStrides []int

	if t.old != nil {
		oShape = t.old.Shape()
		oStrides = t.old.Strides()
	} else {
		oShape = t.Shape()
		oStrides = t.Strides()
	}

	// use the original shape, permute the coordinates later
	if coords, err = types.Itol(i, oShape, oStrides); err != nil {
		err = errors.Wrapf(err, "Failed to do Itol with i: %d, oShape: %v; oStrides: %v", i, oShape, oStrides)
		return
	}

	if t.transposeWith != nil {
		var res [][]int
		if res, err = types.Permute(t.transposeWith, coords); err == nil {
			coords = res[0]
		}
	}
	return
}

// simpleStack is the data movement function for non-view tensors. What it does is simply copy the data according to the new strides
func (t *Tensor) simpleStack(axis int, ap *types.AP, others ...*Tensor) (data []bool) {
	data = make([]bool, ap.Size())
	switch axis {
	case 0:
		copy(data, t.data)
		next := len(t.data)
		for _, ot := range others {
			copy(data[next:], ot.data)
			next += len(ot.data)
		}
	default:
		axisStride := ap.Strides()[axis]
		batches := len(data) / axisStride

		destStart := 0
		start := 0
		end := start + axisStride

		for i := 0; i < batches; i++ {
			copy(data[destStart:], t.data[start:end])
			for _, ot := range others {
				destStart += axisStride
				copy(data[destStart:], ot.data[start:end])
				i++
			}
			destStart += axisStride
			start += axisStride
			end += axisStride
		}
	}
	return
}

// viewStack is the data movement function for Stack(), applied on views
func (t *Tensor) viewStack(axis int, ap *types.AP, others ...*Tensor) (data []bool) {
	data = make([]bool, ap.Size())

	axisStride := ap.Strides()[axis]
	batches := len(data) / axisStride

	it := types.NewFlatIterator(t.AP)
	ch := it.Chan()
	chs := make([]chan int, len(others))
	chs = chs[:0]
	for _, ot := range others {
		oter := types.NewFlatIterator(ot.AP)
		chs = append(chs, oter.Chan())
	}

	data = data[:0]
	for i := 0; i < batches; i++ {
		for j := 0; j < axisStride; j++ {
			id, ok := <-ch
			if !ok {
				break
			}
			data = append(data, t.data[id])
		}
		for j, ot := range others {
			for k := 0; k < axisStride; k++ {
				id, ok := <-chs[j]
				if !ok {
					break
				}
				data = append(data, ot.data[id])
			}
		}
	}
	return
}
