package tensor

import (
	"runtime"
)

// MultIterator is an iterator that iterates over multiple tensors, including masked tensors.
//  It utilizes the *AP of a Tensor to determine what the next index is.
// This data structure is similar to Numpy's flatiter, with some standard Go based restrictions of course
// (such as, not allowing negative indices)
type MultIterator struct {
	*AP                // Uses AP of the largest tensor in list
	fit0 *FlatIterator //largest fit in fitArr (by AP total size)
	mask []bool

	numMasked    int
	lastIndexArr []int
	shape        Shape
	whichBlock   []int
	fitArr       []*FlatIterator
	strides      []int

	size    int
	done    bool
	reverse bool
}

func genIterator(m map[int]int, strides []int, idx int) (int, bool) {
	key := hashIntArray(strides)
	f, ok := m[key]
	if !ok {
		m[key] = idx
		return idx, ok
	}
	return f, ok
}

// NewMultIterator creates a new MultIterator from a list of APs
func NewMultIterator(aps ...*AP) *MultIterator {
	nit := len(aps)
	if nit < 1 {
		return nil
	}
	for _, ap := range aps {
		if ap == nil {
			panic("ap is nil") //TODO: Probably remove this panic
		}
	}

	var maxDims int
	var maxShape = aps[0].shape

	for i := range aps {
		if aps[i].Dims() >= maxDims {
			maxDims = aps[i].Dims()
			if aps[i].Size() > maxShape.TotalSize() {
				maxShape = aps[i].shape
			}
		}

	}

	it := new(MultIterator)

	it.whichBlock = BorrowInts(nit)
	it.lastIndexArr = BorrowInts(nit)
	it.strides = BorrowInts(nit * maxDims)

	shape := BorrowInts(len(maxShape))
	copy(shape, maxShape)
	it.shape = shape

	for _, ap := range aps {
		_, err := BroadcastStrides(shape, ap.shape, it.strides[:maxDims], ap.strides)
		if err != nil {
			panic("can not broadcast strides")
		}
	}

	for i := range it.strides {
		it.strides[i] = 0
	}

	it.fitArr = make([]*FlatIterator, nit)

	//TODO: Convert this make to Borrow perhaps?
	m := make(map[int]int)

	nBlocks := 0
	offset := 0
	for i, ap := range aps {
		f, ok := genIterator(m, ap.strides, nBlocks)
		if !ok {
			offset = nBlocks * maxDims
			apStrides, _ := BroadcastStrides(shape, ap.shape, it.strides[offset:offset+maxDims], ap.strides)
			copy(it.strides[offset:offset+maxDims], apStrides)
			ReturnInts(apStrides) // Borrowed in BroadcastStrides but returned here - dangerous pattern?
			nBlocks++
		}
		it.whichBlock[i] = f
		it.fitArr[nBlocks-1] = NewFlatIterator(NewAP(it.shape[:maxDims], it.strides[offset:offset+maxDims]))
	}

	it.fitArr = it.fitArr[:nBlocks]
	it.strides = it.strides[:nBlocks*maxDims]

	it.fit0 = it.fitArr[0]
	for _, f := range it.fitArr {
		if it.fit0.size < f.size {
			it.fit0 = f
			it.AP = f.AP
		}
	}
	return it
}

// MultIteratorFromDense creates a new MultIterator from a list of dense tensors
func MultIteratorFromDense(tts ...*Dense) *MultIterator {
	aps := BorrowAPList(len(tts))
	hasMask := BorrowBools(len(tts))
	defer ReturnBools(hasMask)

	var masked = false
	numMasked := 0
	for i, tt := range tts {
		aps[i] = tt.Info()
		hasMask[i] = tt.IsMasked()
		masked = masked || hasMask[i]
		if hasMask[i] {
			numMasked++
		}
	}

	it := NewMultIterator(aps...)
	runtime.SetFinalizer(it, destroyIterator)

	if masked {
		// create new mask slice if more than tensor is masked
		if numMasked > 1 {
			it.mask = BorrowBools(it.shape.TotalSize())
			memsetBools(it.mask, false)
			for i, err := it.Start(); err == nil; i, err = it.Next() {
				for j, k := range it.lastIndexArr {
					if hasMask[j] {
						it.mask[i] = it.mask[i] || tts[j].mask[k]
					}
				}
			}
		}
	}
	it.numMasked = numMasked
	ReturnAPList(aps)
	return it
}

// destroyMultIterator returns any borrowed objects back to pool
func destroyMultIterator(it *MultIterator) {

	if cap(it.whichBlock) > 0 {
		ReturnInts(it.whichBlock)
		it.whichBlock = nil
	}
	if cap(it.lastIndexArr) > 0 {
		ReturnInts(it.lastIndexArr)
		it.lastIndexArr = nil
	}
	if cap(it.strides) > 0 {
		ReturnInts(it.strides)
		it.strides = nil
	}
	if it.numMasked > 1 {
		if cap(it.mask) > 0 {
			ReturnBools(it.mask)
			it.mask = nil
		}
	}
}

// SetReverse initializes iterator to run backward
func (it *MultIterator) SetReverse() {
	for _, f := range it.fitArr {
		f.SetReverse()
	}
}

// SetForward initializes iterator to run forward
func (it *MultIterator) SetForward() {
	for _, f := range it.fitArr {
		f.SetForward()
	}
}

//Start begins iteration
func (it *MultIterator) Start() (int, error) {
	it.Reset()
	return it.Next()
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

// Next returns the index of the next coordinate
func (it *MultIterator) Next() (int, error) {
	if it.done {
		return -1, noopError{}
	}
	it.done = false
	for _, f := range it.fitArr {
		f.Next()
		it.done = it.done || f.done
	}
	for i, j := range it.whichBlock {
		it.lastIndexArr[i] = it.fitArr[j].lastIndex
	}
	return it.fit0.lastIndex, nil
}

// NextValid returns the index of the next valid coordinate
func (it *MultIterator) NextValid() (int, int, error) {
	var invalid = true
	var count int
	var mult = 1
	if it.reverse {
		mult = -1
	}
	for invalid {
		if it.done {
			for i, j := range it.whichBlock {
				it.lastIndexArr[i] = it.fitArr[j].lastIndex
			}
			return -1, 0, noopError{}
		}
		for _, f := range it.fitArr {
			f.Next()
			it.done = it.done || f.done
		}
		count++
		invalid = !it.mask[it.fit0.lastIndex]
	}
	return it.fit0.lastIndex, mult * count, nil
}

// NextInvalid returns the index of the next invalid coordinate
func (it *MultIterator) NextInvalid() (int, int, error) {
	var valid = true

	var count = 0
	var mult = 1
	if it.reverse {
		mult = -1
	}
	for valid {
		if it.done {
			for i, j := range it.whichBlock {
				it.lastIndexArr[i] = it.fitArr[j].lastIndex
			}
			return -1, 0, noopError{}
		}
		for _, f := range it.fitArr {
			f.Next()
			it.done = it.done || f.done
		}
		count++
		valid = !it.mask[it.fit0.lastIndex]
	}
	return it.fit0.lastIndex, mult * count, nil
}

// Coord returns the next coordinate.
// When Next() is called, the coordinates are updated AFTER the Next() returned.
// See example for more details.
func (it *MultIterator) Coord() []int {
	return it.fit0.track
}

// Reset resets the iterator state.
func (it *MultIterator) Reset() {
	for _, f := range it.fitArr {
		f.Reset()
	}
	for i, j := range it.whichBlock {
		it.lastIndexArr[i] = it.fitArr[j].lastIndex
	}
	it.done = false
}

// LastIndex returns index of requested iterator
func (it *MultIterator) LastIndex(j int) int {
	return it.lastIndexArr[j]
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
