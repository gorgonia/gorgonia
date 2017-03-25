package tensor

import (
	"runtime"
)

type maskedReduceFn func(Tensor) interface{}

// MaskedReduce applies a reduction function of type maskedReduceFn to mask, and returns
// either an int, or another array
func MaskedReduce(t *Dense, retType Dtype, fn maskedReduceFn, axis ...int) interface{} {
	if len(axis) == 0 || t.IsVector() {
		return fn(t)
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
	retVal := NewDense(retType, ts.shape) //retVal is array to be returned

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
		retVal.SetAt(fn(ts), coord...)

	}
	return retVal
}

// MaskedAny returns True if any mask elements evaluate to True.
// If object is not masked, returns false
// !!! Not the same as numpy's, which looks at data elements and not at mask
// Instead, equivalent to numpy ma.getmask(t).any(axis)
func (t *Dense) MaskedAny(axis ...int) interface{} {
	return MaskedReduce(t, Bool, doMaskAny, axis...)
}

// MaskedAll returns True if all mask elements evaluate to True.
// If object is not masked, returns false
// !!! Not the same as numpy's, which looks at data elements and not at mask
// Instead, equivalent to numpy ma.getmask(t).all(axis)
func (t *Dense) MaskedAll(axis ...int) interface{} {
	return MaskedReduce(t, Bool, doMaskAll, axis...)
}

// MaskedCount counts the masked elements of the array (optionally along the given axis)
// returns -1 if axis out of bounds
func (t *Dense) MaskedCount(axis ...int) interface{} {
	return MaskedReduce(t, Int, doMaskCt, axis...)
}

// NonMaskedCount counts the non-masked elements of the array (optionally along the given axis)
// returns -1 if axis out of bounds
// MaskedCount counts the masked elements of the array (optionally along the given axis)
// returns -1 if axis out of bounds
func (t *Dense) NonMaskedCount(axis ...int) interface{} {
	return MaskedReduce(t, Int, doNonMaskCt, axis...)
}

func doMaskAll(T Tensor) interface{} {
	switch t := T.(type) {
	case *Dense:
		if !t.IsMasked() {
			return false
		}
		m := t.mask
		// contiguous mask case
		if t.MaskSize() == t.Size() {
			for i := 0; i < t.MaskSize(); i++ {
				if !m[i] {
					return false
				}
			}
			return true
		}
		//non-contiguous mask case
		it := MultIteratorFromDense(t)
		runtime.SetFinalizer(it, destroyMultIterator)

		for _, err := it.Next(); err == nil; _, err = it.Next() {
			i := it.LastMaskIndex(0)
			if !m[i] {
				return false
			}
		}
		return true
	default:
		panic("Incompatible type")
	}
}

func doMaskAny(T Tensor) interface{} {

	switch t := T.(type) {
	case *Dense:
		if !t.IsMasked() {
			return false
		}
		m := t.mask
		// contiguous mask case
		if t.MaskSize() == t.Size() {
			for i := 0; i < t.MaskSize(); i++ {
				if m[i] {
					return true
				}
			}
			return false
		}
		//non-contiguous mask case
		it := MultIteratorFromDense(t)
		runtime.SetFinalizer(it, destroyMultIterator)

		for _, err := it.NextInvalid(); err == nil; _, err = it.NextInvalid() {
			i := it.LastMaskIndex(0)
			if m[i] {
				return true
			}
		}
		return false
	default:
		panic("Incompatible type")
	}
}

func doMaskCt(T Tensor) interface{} {
	switch t := T.(type) {
	case *Dense:
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
	default:
		panic("Incompatible type")
	}
}

func doNonMaskCt(T Tensor) interface{} {
	switch t := T.(type) {
	case *Dense:
		if !t.IsMasked() {
			return t.Size()
		}
		return t.Size() - doMaskCt(t).(int)
	default:
		panic("Incompatible type")
	}
}
