package tensor

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

	it := NewIterator(retVal.Info())

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
		if len(t.mask) == t.Size() {
			for _, v := range m {
				if !v {
					return false
				}
			}
		} else {
			it := IteratorFromDense(t)
			i, _, _ := it.NextValid()
			if i != -1 {
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
		if len(t.mask) == t.Size() {
			for _, v := range m {
				if v {
					return true
				}
			}
		} else {
			it := IteratorFromDense(t)
			i, _, _ := it.NextInvalid()
			if i != -1 {
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

		count := 0
		m := t.mask
		if len(t.mask) == t.Size() {
			for _, v := range m {
				if v {
					count++
				}
			}
		} else {
			it := IteratorFromDense(t)
			for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
				count++
			}
		}
		return count
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

/* -----------
************ Finding masked data
----------*/

// FlatNotMaskedContiguous is used to find contiguous unmasked data in a masked array.
// Applies to a flattened version of the array.
// Returns:A sorted sequence of slices (start index, end index).
func (t *Dense) FlatNotMaskedContiguous() []Slice {
	sliceList := make([]Slice, 0, 4)

	it := IteratorFromDense(t)

	for start, _, err := it.NextValid(); err == nil; start, _, err = it.NextValid() {
		end, _, _ := it.NextInvalid()
		if end == -1 {
			end = t.Size()
		}
		sliceList = append(sliceList, makeRS(start, end))
	}

	return sliceList
}

// FlatMaskedContiguous is used to find contiguous masked data in a masked array.
// Applies to a flattened version of the array.
// Returns:A sorted sequence of slices (start index, end index).
func (t *Dense) FlatMaskedContiguous() []Slice {
	sliceList := make([]Slice, 0, 4)

	it := IteratorFromDense(t)

	for start, _, err := it.NextInvalid(); err == nil; start, _, err = it.NextInvalid() {
		end, _, _ := it.NextValid()
		if end == -1 {
			end = t.Size()
		}
		sliceList = append(sliceList, makeRS(start, end))
	}
	return sliceList
}

// FlatNotMaskedEdges is used to find the indices of the first and last unmasked values
// Applies to a flattened version of the array.
// Returns: A pair of ints. -1 if all values are masked.
func (t *Dense) FlatNotMaskedEdges() (int, int) {
	if !t.IsMasked() {
		return 0, t.Size() - 1
	}

	var start, end int
	it := IteratorFromDense(t)

	it.SetForward()
	start, _, err := it.NextValid()
	if err != nil {
		return -1, -1
	}

	it.SetReverse()
	end, _, _ = it.NextValid()

	return start, end
}

// FlatMaskedEdges is used to find the indices of the first and last masked values
// Applies to a flattened version of the array.
// Returns: A pair of ints. -1 if all values are unmasked.
func (t *Dense) FlatMaskedEdges() (int, int) {
	if !t.IsMasked() {
		return 0, t.Size() - 1
	}
	var start, end int
	it := IteratorFromDense(t)

	it.SetForward()
	start, _, err := it.NextInvalid()
	if err != nil {
		return -1, -1
	}

	it.SetReverse()
	end, _, _ = it.NextInvalid()

	return start, end
}

// ClumpMasked returns a list of slices corresponding to the masked clumps of a 1-D array
// Added to match numpy function names
func (t *Dense) ClumpMasked() []Slice {
	return t.FlatMaskedContiguous()
}

// ClumpUnmasked returns a list of slices corresponding to the unmasked clumps of a 1-D array
// Added to match numpy function names
func (t *Dense) ClumpUnmasked() []Slice {
	return t.FlatNotMaskedContiguous()
}
