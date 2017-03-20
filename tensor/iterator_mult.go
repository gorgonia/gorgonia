package tensor

// MultIterator is an iterator that iterates over multiple tensors, including masked tensors.
//  It utilizes the *AP of a Tensor to determine what the next index is.
// This data structure is similar to Numpy's flatiter, with some standard Go based restrictions of course
// (such as, not allowing negative indices)

type MultIterator struct {
	clock         int
	aps           []*AP
	fitArr        []*FlatIterator
	lastDataIndex []*int
	lastMaskIndex []*int
	track         [][]int
	done          bool
	dummyInt      int // Should be -1
}

func GenIterator(m map[int]*FlatIterator, shape []int, strides []int) *FlatIterator {
	key := hashIntArrayPair(shape, strides)
	f := m[key]
	if f == nil {
		f = NewFlatIterator(&AP{shape: shape, strides: strides})
		m[key] = f
	}
	return f
}

// NewMultIterator creates a new MultIterator
func NewMultIterator(aps ...*AP) *MultIterator {
	nit := len(aps)
	if nit < 1 {
		return nil
	}

	/*for i := 1; i < len(aps); i++ {
		if aps[i] != nil {
			if !(aps[i].IsScalar()) {
				for j := 0; j < len(aps); i++ {
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
	}*/

	it := new(MultIterator)

	it.fitArr = make([]*FlatIterator, 2*nit)
	it.lastDataIndex = make([]*int, nit)
	it.lastMaskIndex = make([]*int, nit)
	it.track = make([][]int, 2*nit)
	it.dummyInt = -1

	m := make(map[int]*FlatIterator)

	for i, ap := range aps {
		it.lastDataIndex[i] = &it.dummyInt
		it.lastMaskIndex[i] = &it.dummyInt

		if ap != nil {
			if ap.IsMasked() {
				f := GenIterator(m, ap.shape, ap.maskStrides)
				it.lastMaskIndex[i] = &(f.lastIndex)
				it.track[2*i+1] = f.track
			}
			f := GenIterator(m, ap.shape, ap.strides)
			it.lastDataIndex[i] = &(f.lastIndex)
			it.track[2*i] = f.track
		}
	}

	i := 0
	for _, f := range m {
		it.fitArr[i] = f
		i++
	}
	it.fitArr = it.fitArr[:i]
	return it
}

//Done checks whether iterators are done
func (it *MultIterator) Done() bool {
	for _, f := range it.fitArr {
		if !f.done {
			it.done = false
			return false
		}
	}
	it.done = true
	return true
}

// Next returns the index of the current coordinate.
func (it *MultIterator) Next() error {
	if it.Done() {
		return noopError{}
	}
	for _, f := range it.fitArr {
		f.Next()
	}
	return nil
}

// LastIndex returns the index of the current coordinate.
func (it *MultIterator) LastIndex(i int) int {
	return *(it.lastDataIndex[i])
}

// LastMaskIndex returns the index of the current coordinate.
func (it *MultIterator) LastMaskIndex(i int) int {
	return *(it.lastMaskIndex[i])
}

// Coord returns the next coordinate.
// When Next() is called, the coordinates are updated AFTER the Next() returned.
// See example for more details.
func (it *MultIterator) Coord(i int) []int {
	return it.track[2*i]
}

// MaskCoord returns the next coordinate of mask.
// When Next() is called, the coordinates are updated AFTER the Next() returned.
// See example for more details.
func (it *MultIterator) MaskCoord(i int) []int {
	return it.track[2*i+1]
}

// Reset resets the iterator state.
func (it *MultIterator) Reset() {
	it.done = false
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
