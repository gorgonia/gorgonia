package tensor

import "github.com/pkg/errors"

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
