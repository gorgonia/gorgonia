package tensor

import "github.com/pkg/errors"

// Apply applies a function to all the values in the ndarray
// func (t *Dense) Apply(fn interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
// fo := parseFuncOpts(opts...)
// reuseT, incr := fo.incrReuse()
// safe := fo.safe()

// var reuse *Dense
// if reuse, err = getDense(reuseT); err != nil {
// 	return
// }

// // check reuse and stuff
// var res Array
// switch {
// case reuse != nil:
// 	res = reuse.data
// 	if res.Len() != t.Size() {
// 		err = errors.Errorf(shapeMismatch, t.Shape(), reuse.Shape())
// 		return
// 	}
// case !safe:
// 	res = t.data
// default:
// 	if t.IsMaterializable() {
// 		res = makeArray(t.t, t.Shape().TotalSize())
// 	} else {
// 		res = cloneArray(t.data)
// 	}
// }
// // do
// switch {
// case t.viewOf == nil && !incr:
// 	res.Map(fn)
// case t.viewOf == nil && incr:
// 	rn, ok := res.(Number)
// 	if !ok {
// 		err = errors.Errorf("Can only incr on Number arrays")
// 		return
// 	}

// 	tn, ok := t.data.(Number)
// 	if !ok {
// 		err = errors.Errorf("Can only incr on Number Arrays")
// 		return
// 	}

// 	cloned := cloneArray(tn).(Number)
// 	if err = cloned.Map(fn); err != nil {
// 		return
// 	}

// 	if err = rn.Add(cloned); err != nil {
// 		return
// 	}
// case t.viewOf != nil:
// 	var im IterMapper
// 	var ok bool
// 	if im, ok = res.(IterMapper); !ok {
// 		panic("Not handled yet")
// 	}

// 	it := NewFlatIterator(t.AP)
// 	if err = im.IterMap(t.data, nil, it, fn, incr); err != nil {
// 		return
// 	}

// default:
// 	err = errors.Errorf("Apply not implemented for this state: isView: %t and incr: %t", t.viewOf == nil, incr)
// 	return
// }
// // set retVal
// switch {
// case reuse != nil:
// 	if err = reuseCheckShape(reuse, t.Shape()); err != nil {
// 		return
// 	}
// 	retVal = reuse
// case !safe:
// 	retVal = t
// default:
// 	retVal = New(Of(t.t), WithBacking(res), WithShape(t.Shape()...))
// }
// return
// }

// T performs a thunked transpose. It doesn't actually do anything, except store extra information about the post-transposed shapes and strides
// Usually this is more than enough, as BLAS will handle the rest of the transpose
func (t *Dense) T(axes ...int) (err error) {
	var transform *AP
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
		if _, ok := err.(NoOpError); !ok {
			return
		}
		err = nil
		return
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
func (t *Dense) Transpose() {
	// if there is no oldinfo, that means the current info is the latest, and not the transpose
	if t.old == nil {
		return
	}

	if t.IsScalar() {
		return // cannot transpose scalars
	}

	defer func() {
		ReturnAP(t.old)
		t.old = nil
		t.transposeWith = nil
	}()

	expShape := t.Shape()
	expStrides := expShape.CalcStrides() // important! because the strides would have changed once the underlying data changed
	defer ReturnInts(expStrides)
	defer func() {
		t.setShape(expShape...)
		t.sanity()
	}()

	if t.IsVector() {
		// no change of strides.
		return
	}

	t.transpose(expStrides)
}

// At returns the value at the given coordinate
func (t *Dense) At(coords ...int) (interface{}, error) {
	if len(coords) != t.Dims() {
		return nil, errors.Errorf(dimMismatch, t.Dims(), len(coords))
	}

	at, err := t.at(coords...)
	if err != nil {
		return nil, errors.Wrap(err, "At()")
	}

	return t.get(at), nil
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

// at returns the index at which the coordinate is refering to.
// This function encapsulates the addressing of elements in a contiguous block.
// For a 2D ndarray, ndarray.at(i,j) is
//		at = ndarray.strides[0]*i + ndarray.strides[1]*j
// This is of course, extensible to any number of dimensions.
func (t *Dense) at(coords ...int) (at int, err error) {
	return Ltoi(t.Shape(), t.Strides(), coords...)
}
