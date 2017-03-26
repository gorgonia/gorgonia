package tensor

// MultIterator is an iterator that iterates over multiple tensors, including masked tensors.
//  It utilizes the *AP of a Tensor to determine what the next index is.
// This data structure is similar to Numpy's flatiter, with some standard Go based restrictions of course
// (such as, not allowing negative indices)
type MultIterator struct {
	fit0      *FlatIterator
	trackIdx  []int
	fitArr    []*FlatIterator
	lastIndex []*int
	masks     [][]bool
	termInt   int // Should be -1
}

func genIterator(m map[int]*FlatIterator, shape []int, strides []int) (*FlatIterator, int) {
	key := hashIntArrayPair(shape, strides)
	f := m[key]
	if f == nil {
		f = NewFlatIterator(&AP{shape: shape, strides: strides})
		m[key] = f
	}
	return f, key
}

// NewMultIterator creates a new MultIterator from a list of APs
func NewMultIterator(aps ...*AP) *MultIterator {
	nit := len(aps)
	if nit < 1 {
		return nil
	}

	it := new(MultIterator)

	for i := 1; i < len(aps); i++ {
		if aps[i] != nil {
			if !(aps[i].IsScalar()) {
				for j := 0; j < len(aps); j++ {
					if aps[j] != nil {
						if !(aps[j].IsScalar()) {
							if !EqInts(aps[i].shape, aps[j].shape) {
								//panic("Can not iterate through tensors of different shape")
								return nil
							}
						}
					}
				}
				break
			}
		}
	}

	it.fitArr = make([]*FlatIterator, 2*nit)
	it.lastIndex = make([]*int, 2*nit)
	it.trackIdx = BorrowInts(2 * nit)
	it.termInt = -1

	if nit == 1 && !aps[0].IsMasked() {
		it.fit0 = NewFlatIterator(&AP{shape: aps[0].shape, strides: aps[0].strides})
		it.fitArr[0] = it.fit0
		it.lastIndex[0] = &(it.fit0.lastIndex)
		it.trackIdx[0] = 0
		return it
	}

	m := make(map[int]*FlatIterator)

	for i, ap := range aps {
		it.lastIndex[2*i] = &it.termInt
		it.lastIndex[2*i+1] = &it.termInt

		if ap != nil {
			if ap.IsMasked() {
				f, key := genIterator(m, ap.shape, ap.maskStrides)
				it.lastIndex[2*i+1] = &(f.lastIndex)
				it.trackIdx[2*i+1] = key
			}
			f, key := genIterator(m, ap.shape, ap.strides)
			it.lastIndex[2*i] = &(f.lastIndex)
			it.trackIdx[2*i] = key
		}
	}

	i := 0
	for key, f := range m {
		it.fitArr[i] = f
		for j := range it.trackIdx {
			if it.trackIdx[j] == key {
				it.trackIdx[j] = i
			}
		}
		i++
	}
	it.fitArr = it.fitArr[:i]
	return it
}

// MultIteratorFromDense creates a new MultIterator from a list of dense tensors
func MultIteratorFromDense(tts ...*Dense) *MultIterator {
	aps := BorrowAPList(len(tts))
	var masked = false
	for i, tt := range tts {
		aps[i] = tt.Info()
		masked = masked || aps[i].IsMasked()
	}
	it := NewMultIterator(aps...)
	if masked {
		masks := BorrowMaskList(len(tts))
		for i, tt := range tts {
			masks[i] = tt.mask
		}
		it.masks = masks
	}
	ReturnAPList(aps)
	return it
}

// destroyMultIterator creates a new MultIterator from a list of dense tensors
func destroyMultIterator(it *MultIterator) {
	if cap(it.masks) > 0 {
		ReturnMaskList(it.masks)
		it.masks = nil
	}
	if cap(it.trackIdx) > 0 {
		ReturnInts(it.trackIdx)
		it.trackIdx = nil
	}
}

// SetReverse initializes iterator to run backwards
func (it *MultIterator) SetReverse() {
	for _, f := range it.fitArr {
		if f != nil {
			f.SetReverse()
		}
	}
}

//Done checks whether iterators are done
func (it *MultIterator) Done() bool {
	if it.fit0 != nil {
		return it.fit0.done
	}
	for _, f := range it.fitArr {
		if !f.done {
			return false
		}
	}
	return true
}

// Next returns the index of the next coordinate
func (it *MultIterator) Next() (int, error) {
	if it.fit0 != nil {
		return it.fit0.Next()
	}
	if it.Done() {
		return -1, noopError{}
	}
	for _, f := range it.fitArr {
		f.Next()
	}
	return *(it.lastIndex[0]), nil
}

// NextValid returns the index of the next valid coordinate
func (it *MultIterator) NextValid() (int, error) {
	if len(it.masks) < 1 {
		return -1, noopError{} // Need to find right error code
	}
	var invalid = true
	for invalid {
		if it.Done() {
			return -1, noopError{}
		}
		for _, f := range it.fitArr {
			f.Next()
		}
		for i, idp := range it.lastIndex {
			if i%2 == 1 {
				invalid = invalid && it.masks[i>>1][*idp]
			}
		}
	}
	return *(it.lastIndex[0]), nil
}

// NextInvalid returns the index of the next invalid coordinate
func (it *MultIterator) NextInvalid() (int, error) {
	if len(it.masks) < 1 {
		return -1, noopError{} // Need to find right error code
	}
	var invalid = true
	for invalid {
		if it.Done() {
			return -1, noopError{}
		}
		for _, f := range it.fitArr {
			f.Next()
		}
		for i, idp := range it.lastIndex {
			if i%2 == 1 {
				invalid = invalid && !it.masks[i>>1][*idp]
			}
		}
	}
	return *(it.lastIndex[0]), nil
}

// LastIndex returns the index of the current coordinate.
func (it *MultIterator) LastIndex(i int) int {
	return *(it.lastIndex[2*i])
}

// LastMaskIndex returns the index of the current coordinate.
func (it *MultIterator) LastMaskIndex(i int) int {
	return *(it.lastIndex[2*i+1])
}

// Coord returns the next coordinate.
// When Next() is called, the coordinates are updated AFTER the Next() returned.
// See example for more details.
func (it *MultIterator) Coord(i int) []int {
	return it.fitArr[it.trackIdx[2*i]].track
}

// MaskCoord returns the next coordinate of mask.
// When Next() is called, the coordinates are updated AFTER the Next() returned.
// See example for more details.
func (it *MultIterator) MaskCoord(i int) []int {
	return it.fitArr[it.trackIdx[2*i+1]].track
}

// Reset resets the iterator state.
func (it *MultIterator) Reset() {
	if it.fit0 != nil {
		it.fit0.Reset()
		return
	}
	for _, f := range it.fitArr {
		f.Reset()
	}
}

/*
// Chan returns a channel of ints. This is useful for iterating multiple Tensors at the same time.
func (it *FlatIterator) Chan() (retVal chan int) {
	retVal = make(chan int)

	go func() {
		for next, err := it.Next(); err == nil; next, err = it.Next() {
			retVal <- next
		}
		close(retVal)
	}()

	return
}

/* TEMPORARILY REMOVED
// SortedMultiStridePerm takes multiple input strides, and creates a sorted stride permutation.
// It's based very closely on Numpy's PyArray_CreateMultiSortedStridePerm, where a stable insertion sort is used
// to create the permutations.
func SortedMultiStridePerm(dims int, aps []*AP) (retVal []int) {
	retVal = BorrowInts(dims)
	for i := 0; i < dims; i++ {
		retVal[i] = i
	}

	for i := 1; i < dims; i++ {
		ipos := i
		axisi := retVal[i]

		for j := i - 1; j >= 0; j-- {
			var ambig, swap bool
			ambig = true
			axisj := retVal[j]

			for _, ap := range aps {
				if ap.shape[axisi] != 1 && ap.shape[axisj] != 1 {
					if ap.strides[axisi] <= ap.strides[axisj] {
						swap = true
					} else if ambig {
						swap = true
					}
					ambig = false
				}
			}

			if !ambig && swap {
				ipos = j
			} else {
				break
			}

		}
		if ipos != i {
			for j := i; j > ipos; j-- {
				retVal[j] = retVal[j-1]
			}
			retVal[ipos] = axisi
		}
	}
	return
}
*/
