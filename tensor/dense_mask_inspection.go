package tensor

import (
	"runtime"
)

// MaskedAll returns True if all mask elements evaluate to True.
// If object is not masked, returns false
func (t *Dense) MaskedAll() bool {
	if !t.IsMasked() {
		return false
	}
	m := t.mask
	for i := 0; i < t.MaskSize(); i++ {
		if !m[i] {
			return false
		}
	}
	return true
}

// MaskedAny returns True if any mask elements evaluate to True.
// If object is not masked, returns false
func (t *Dense) MaskedAny() bool {
	if !t.IsMasked() {
		return false
	}
	m := t.mask
	for i := 0; i < t.MaskSize(); i++ {
		if m[i] {
			return true
		}
	}
	return false
}

// MaskedCount counts the masked elements of the array (optionally along the given axis)
// returns -1 if axis out of bounds
func (t *Dense) MaskedCount(axis ...int) interface{} {
	if len(axis) == 0 || t.IsVector() {
		return t.doMaskCt()
	}
	ax := axis[0]
	if ax >= t.Dims() {
		return -1
	}
	// create object to be used for slicing
	slices := make([]Slice, t.Dims())

	// calculate shape of tensor to be returned
	slices[ax] = makeRS(0, 0)
	tt, _ := t.Slice(slices...)
	ts := tt.(*Dense)
	retVal := NewDense(Int, ts.shape) //retVal is array to be returned

	it := NewFlatIterator(retVal.Info())

	// iterate through retVal
	slices[ax] = makeRS(0, t.shape[ax])
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		coord := it.Coord()
		k := 0
		for d := range slices {
			if d != ax {
				slices[d] = makeRS(coord[k], coord[k]+1)
				k++
			} else {
				slices[d] = nil
			}
		}
		tt, _ = t.Slice(slices...)
		ts = tt.(*Dense)
		retVal.SetAt(ts.doMaskCt(), coord...)
	}
	return retVal
}

func (t *Dense) doMaskCt() int {
	// non masked case
	if !t.IsMasked() {
		return 0
	}
	// contiguous mask case
	if t.MaskSize() == t.Size() {
		count := 0
		m := t.mask
		for i := 0; i < t.MaskSize(); i++ {
			if m[i] {
				count++
			}
		}
		return count
	}
	//non-contiguous mask case
	it := MultIteratorFromDense(t)
	runtime.SetFinalizer(it, destroyMultIterator)
	j := 0
	for _, err := it.NextInvalid(); err == nil; _, err = it.NextInvalid() {
		j++
	}
	return j
}

// NonMaskedCount counts the non-masked elements of the array (optionally along the given axis)
// returns -1 if axis out of bounds
// MaskedCount counts the masked elements of the array (optionally along the given axis)
// returns -1 if axis out of bounds
func (t *Dense) NonMaskedCount(axis ...int) interface{} {
	if len(axis) == 0 || t.IsVector() {
		return t.doNonMaskCt()
	}
	ax := axis[0]
	if ax >= t.Dims() {
		return -1
	}
	// create object to be used for slicing
	slices := make([]Slice, t.Dims())

	// calculate shape of tensor to be returned
	slices[ax] = makeRS(0, 0)
	tt, _ := t.Slice(slices...)
	ts := tt.(*Dense)
	retVal := NewDense(Int, ts.shape) //retVal is array to be returned

	it := NewFlatIterator(retVal.Info())

	// iterate through retVal
	slices[ax] = makeRS(0, t.shape[ax])
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		coord := it.Coord()
		k := 0
		for d := range slices {
			if d != ax {
				slices[d] = makeRS(coord[k], coord[k]+1)
				k++
			} else {
				slices[d] = nil
			}
		}
		tt, _ = t.Slice(slices...)
		ts = tt.(*Dense)
		retVal.SetAt(ts.doNonMaskCt(), coord...)
	}
	return retVal
}

func (t *Dense) doNonMaskCt() int {
	if !t.IsMasked() {
		return t.Size()
	}
	return t.Size() - t.doMaskCt()
}
