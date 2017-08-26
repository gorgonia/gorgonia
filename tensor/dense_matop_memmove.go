package tensor

import "github.com/pkg/errors"

// This file contains code pertaining to tensor operations that actually move memory

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
		copy(t.AP.strides, expStrides) // dimensions do not change, so it's actually safe to do this
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

// Concat concatenates the other tensors along the given axis. It is like Numpy's concatenate() function.
func (t *Dense) Concat(axis int, Ts ...*Dense) (retVal *Dense, err error) {
	e := t.Engine()
	if e == nil {
		e = StdEng{}
	}
	if c, ok := e.(Concater); ok {
		var ret Tensor
		others := densesToTensors(Ts)
		if ret, err = c.Concat(t, axis, others...); err != nil {
			return nil, errors.Wrapf(err, opFail, "Concat")
		}
		return ret.(*Dense), nil
	}
	return nil, errors.New("Engine does not support Concat")
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
	e := t.Engine()
	if e == nil {
		e = StdEng{}
	}

	if ds, ok := e.(DenseStacker); ok {
		var ret DenseTensor
		ots := densesToDenseTensors(others)
		if ret, err = ds.StackDense(t, axis, ots...); err != nil {
			return
		}
		return ret.(*Dense), err
	}
	return nil, errors.Errorf("Engine does not support DenseStacker")
}
