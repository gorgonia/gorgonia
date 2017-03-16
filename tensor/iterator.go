package tensor

// FlatIterator is an iterator that iterates over Tensors. It utilizes the *AP
// of a Tensor to determine what the next index is.
// This data structure is similar to Numpy's flatiter, with some standard Go based restrictions of course
// (such as, not allowing negative indices)
type FlatIterator struct {
	*AP

	//state
	lastIndex int
	strides0  int
	size      int
	track     []int
	done      bool
}

// NewFlatIterator creates a new FlatIterator
func NewFlatIterator(ap *AP) *FlatIterator {
	var strides0 int
	if ap.IsVector() {
		strides0 = ap.strides[0]
	}

	return &FlatIterator{
		AP:       ap,
		track:    make([]int, len(ap.shape)),
		size:     ap.shape.TotalSize(),
		strides0: strides0,
	}
}

// Next returns the index of the current coordinate.
func (it *FlatIterator) Next() (int, error) {
	if it.done {
		return -1, noopError{}
	}

	switch {
	case it.IsScalar():
		it.done = true
		return 0, nil
	case it.IsVector():
		return it.singleNext()
	default:
		return it.ndNext()
	}
}

func (it *FlatIterator) singleNext() (int, error) {
	retVal := it.lastIndex
	// it.lastIndex += it.strides[0]
	it.lastIndex += it.strides0

	var tracked int
	switch {
	case it.IsRowVec():
		it.track[1]++
		tracked = it.track[1]
	case it.IsColVec(), it.IsVector():
		it.track[0]++
		tracked = it.track[0]
	default:
		panic("This ain't supposed to happen")
	}

	if tracked >= it.size {
		it.done = true
	}

	return retVal, nil
}

func (it *FlatIterator) ndNext() (int, error) {
	retVal := it.lastIndex
	for i := len(it.shape) - 1; i >= 0; i-- {
		it.track[i]++
		if it.track[i] == it.shape[i] {
			if i == 0 {
				it.done = true
			}
			it.track[i] = 0
			it.lastIndex -= (it.shape[i] - 1) * it.strides[i]
			continue
		}
		it.lastIndex += it.strides[i]
		break
	}
	return retVal, nil
}

// Coord returns the next coordinate.
// When Next() is called, the coordinates are updated AFTER the Next() returned.
// See example for more details.
func (it *FlatIterator) Coord() []int {
	return it.track
}

// Slice is a convenience function that augments
func (it *FlatIterator) Slice(sli Slice) (retVal []int, err error) {
	var next int
	var nexts []int
	for next, err = it.Next(); err == nil; next, err = it.Next() {
		nexts = append(nexts, next)
	}
	if _, ok := err.(NoOpError); err != nil && !ok {
		return
	}

	if sli == nil {
		retVal = nexts
		return
	}

	start := sli.Start()
	end := sli.End()
	step := sli.Step()

	// sanity checks
	if err = CheckSlice(sli, len(nexts)); err != nil {
		return
	}

	if step < 0 {
		// reverse the nexts
		for i := len(nexts)/2 - 1; i >= 0; i-- {
			j := len(nexts) - 1 - i
			nexts[i], nexts[j] = nexts[j], nexts[i]
		}
		step = -step
	}

	// cleanup before loop
	if end > len(nexts) {
		end = len(nexts)
	}
	// nexts = nexts[:end]

	for i := start; i < end; i += step {
		retVal = append(retVal, nexts[i])
	}

	err = nil
	return
}

// Reset resets the iterator state.
func (it *FlatIterator) Reset() {
	it.done = false
	it.lastIndex = 0

	if it.done {
		return
	}

	for i := range it.track {
		it.track[i] = 0
	}
}

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
