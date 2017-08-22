package tensor

import "runtime"

func requiresIterator(a Tensor) bool {
	switch tt := a.(type) {
	case DenseTensor:
		if mt, ok := tt.(MaskedTensor); ok && mt.IsMasked() {
			return true
		}
		if v, ok := tt.(View); ok && v.IsMaterializable() {
			if tt.DataOrder().isContiguous() {
				return false
			}
			return true
		}
		if tt.DataOrder().isNotContiguous() {
			return true
		}
		return false
	case SparseTensor:
		return true
	}
	panic("Unreachable")
}

func requiresOrderedIterator(e Engine, t Tensor) bool {
	if requiresIterator(t) {
		return true
	}
	switch tt := t.(type) {
	case DenseTensor:
		return !e.DataOrder().hasSameOrder(tt.DataOrder())
	case SparseTensor:
		return true
	}
	panic("Unreachable")
}

// Iterator is the generic iterator interface.
// It's used to iterate across multi-dimensional slices, no matter the underlying data arrangement
type Iterator interface {
	// Start returns the first index
	Start() (int, error)

	// Next returns the next index. Next is defined as the next value in the coordinates
	// For example: let x be a (5,5) matrix that is row-major. Current index is for the coordinate (3,3).
	// Next() returns the index of (3,4).
	//
	// If there is no underlying data store for (3,4) - say for example, the matrix is a sparse matrix, it return an error.
	// If however, there is an underlying data store for (3,4), but it's not valid (for example, masked tensors), it will not return an error.
	//
	// Second example: let x be a (5,5) matrix that is col-major. Current index is for coordinate (3,3).
	// Next() returns the index of (4,3).
	Next() (int, error)

	// NextValidity is like Next, but returns the validity of the value at the index as well.
	NextValidity() (int, bool, error)

	// NextValid returns the next valid index, as well as a skip count.
	NextValid() (int, int, error)

	// NextInvalid returns the next invalid index, as well as a skip count.
	NextInvalid() (int, int, error)

	// Reset resets the iterator
	Reset()

	// SetReverse tells the iterator to iterate in reverse
	SetReverse()

	// SetForward tells the iterator to iterate forwards
	SetForward()

	// Coord returns the coordinates
	Coord() []int

	// Done returns true when the iterator is done iterating.
	Done() bool

	// Shape returns the shape of the multidimensional tensor it's iterating on.
	Shape() Shape
}

// NewIterator creates a new Iterator from an ap. The type of iterator depends on number of
// aps passed, and whether they are masked or not
func NewIterator(aps ...*AP) Iterator {
	switch len(aps) {
	case 0:
		return nil
	case 1:
		return NewFlatIterator(aps[0])
	default:
		return NewMultIterator(aps...)
	}
}

// IteratorFromDense creates a new Iterator from a list of dense tensors
func IteratorFromDense(tts ...DenseTensor) Iterator {
	switch len(tts) {
	case 0:
		return nil
	case 1:
		if mt, ok := tts[0].(MaskedTensor); ok && mt.IsMasked() {
			return FlatMaskedIteratorFromDense(mt)
		}
		return FlatIteratorFromDense(tts[0])
	default:
		return MultIteratorFromDense(tts...)
	}
}

func IteratorFromTensor(a Tensor) Iterator {
	switch at := a.(type) {
	case DenseTensor:
		return IteratorFromDense(at)
	case *CS:
		return NewFlatSparseIterator(at)
	}
	panic("Unreachable")
}

func destroyIterator(it Iterator) {
	switch itt := it.(type) {
	case *MultIterator:
		destroyMultIterator(itt)
	}
}

func iteratorLoadAP(it Iterator, ap *AP) {
	switch itt := it.(type) {
	case *FlatIterator:
		itt.AP = ap
	case *FlatMaskedIterator:
		itt.AP = ap
	case *MultIterator: // Do nothing, TODO: perhaps add something here

	}
}

/* FLAT ITERATOR */

// FlatIterator is an iterator that iterates over Tensors. It utilizes the *AP
// of a Tensor to determine what the next index is.
// This data structure is similar to Numpy's flatiter, with some standard Go based restrictions of course
// (such as, not allowing negative indices)
type FlatIterator struct {
	*AP

	//state
	nextIndex int
	lastIndex int
	strides0  int
	size      int
	track     []int
	done      bool
	reverse   bool // if true, iterator starts at end of array and runs backwards
}

// NewFlatIterator creates a new FlatIterator.
func NewFlatIterator(ap *AP) *FlatIterator {
	var strides0 int
	if ap.IsVector() {
		strides0 = ap.strides[0]
	} else if ap.o.isColMajor() {
		strides0 = ap.strides[0]
	}

	return &FlatIterator{
		AP:       ap,
		track:    make([]int, len(ap.shape)),
		size:     ap.shape.TotalSize(),
		strides0: strides0,
	}
}

// FlatIteratorFromDense creates a new FlatIterator from a dense tensor
func FlatIteratorFromDense(tt DenseTensor) *FlatIterator {
	return NewFlatIterator(tt.Info())
}

// SetReverse initializes iterator to run backwards
func (it *FlatIterator) SetReverse() {
	it.reverse = true
	it.Reset()
	return
}

// SetForward initializes iterator to run forwards
func (it *FlatIterator) SetForward() {
	it.reverse = false
	it.Reset()
	return
}

//Start begins iteration
func (it *FlatIterator) Start() (int, error) {
	it.Reset()
	return it.Next()
}

//Done checks whether iterators are done
func (it *FlatIterator) Done() bool {
	return it.done
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
		if it.reverse {
			return it.singlePrevious()
		}
		return it.singleNext()
	default:
		if it.reverse {
			return it.ndPrevious()
		}
		return it.ndNext()
	}
}

// NextValidity returns the index of the current coordinate, and whether or not it's valid. Identical to Next()
func (it *FlatIterator) NextValidity() (int, bool, error) {
	i, err := it.Next()
	return i, true, err
}

// NextValid returns the index of the current coordinate. Identical to Next for FlatIterator
// Also returns the number of increments to get to next element ( 1,  or -1 in reverse case). This is to maintain
// consistency with the masked iterator, for which the step between valid elements can be more than 1
func (it *FlatIterator) NextValid() (int, int, error) {
	if it.done {
		return -1, 1, noopError{}
	}
	switch {
	case it.IsScalar():
		it.done = true
		return 0, 0, nil
	case it.IsVector():
		if it.reverse {
			a, err := it.singlePrevious()
			return a, -1, err
		}
		a, err := it.singleNext()
		return a, 1, err
	default:
		if it.reverse {
			a, err := it.ndPrevious()
			return a, -1, err
		}
		a, err := it.ndNext()
		return a, 1, err
	}
}

// NextInvalid returns the index of the current coordinate. Identical to Next for FlatIterator
// also returns the number of increments to get to next invalid element (1 or -1 in reverse case).
// Like NextValid, this method's purpose is to maintain consistency with the masked iterator,
// for which the step between invalid elements can be anywhere from 0 to the  tensor's length
func (it *FlatIterator) NextInvalid() (int, int, error) {
	if it.reverse {
		return -1, -it.lastIndex, noopError{}
	}
	return -1, it.Size() - it.lastIndex, noopError{}
}

func (it *FlatIterator) singleNext() (int, error) {
	it.lastIndex = it.nextIndex
	// it.lastIndex += it.strides[0]
	it.nextIndex += it.strides0

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

	return it.lastIndex, nil
}

func (it *FlatIterator) singlePrevious() (int, error) {
	it.lastIndex = it.nextIndex
	// it.lastIndex += it.strides[0]
	it.nextIndex -= it.strides0

	var tracked int
	switch {
	case it.IsRowVec():
		it.track[1]--
		tracked = it.track[1]
	case it.IsColVec(), it.IsVector():
		it.track[0]--
		tracked = it.track[0]
	default:
		panic("This ain't supposed to happen")
	}

	if tracked < 0 {
		it.done = true
	}

	return it.lastIndex, nil
}

func (it *FlatIterator) ndNext() (int, error) {
	it.lastIndex = it.nextIndex
	for i := len(it.shape) - 1; i >= 0; i-- {
		it.track[i]++
		if it.track[i] == it.shape[i] {
			if i == 0 {
				it.done = true
			}
			it.track[i] = 0
			it.nextIndex -= (it.shape[i] - 1) * it.strides[i]
			continue
		}
		it.nextIndex += it.strides[i]
		break
	}
	return it.lastIndex, nil
}

func (it *FlatIterator) colMajorNDNext() (int, error) {
	return 0, nil
}

func (it *FlatIterator) ndPrevious() (int, error) {
	it.lastIndex = it.nextIndex
	for i := len(it.shape) - 1; i >= 0; i-- {
		it.track[i]--
		if it.track[i] < 0 {
			if i == 0 {
				it.done = true
			}
			it.track[i] = it.shape[i] - 1
			it.nextIndex += (it.shape[i] - 1) * it.strides[i]
			continue
		}
		it.nextIndex -= it.strides[i]
		break
	}
	return it.lastIndex, nil
}

func (it *FlatIterator) colMajorNDPrevious() (int, error) {
	return 0, nil
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
	if it.reverse {
		for i := range it.track {
			it.track[i] = it.shape[i] - 1
		}

		switch {
		case it.IsScalar():
			it.nextIndex = 0
		case it.IsRowVec():
			it.nextIndex = (it.shape[1] - 1) * it.strides[0]
		case it.IsColVec(), it.IsVector():
			it.nextIndex = (it.shape[0] - 1) * it.strides[0]
		default:
			it.nextIndex = 0
			for i := range it.track {
				it.nextIndex += (it.shape[i] - 1) * it.strides[i]
			}
		}
	} else {
		it.nextIndex = 0
		for i := range it.track {
			it.track[i] = 0
		}
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

/* FLAT MASKED ITERATOR */

// FlatMaskedIterator is an iterator that iterates over simple masked Tensors.
// It is used when the mask stride is identical to data stride with the exception of trailing zeros,
// in which case the data index is always a perfect integer multiple of the mask index
type FlatMaskedIterator struct {
	*FlatIterator
	mask []bool
}

// NewFlatMaskedIterator creates a new flat masked iterator
func NewFlatMaskedIterator(ap *AP, mask []bool) *FlatMaskedIterator {
	it := new(FlatMaskedIterator)
	runtime.SetFinalizer(it, destroyIterator)
	it.FlatIterator = NewFlatIterator(ap)
	it.mask = mask
	return it
}

// FlatMaskedIteratorFromDense creates a new FlatMaskedIterator from dense tensor
func FlatMaskedIteratorFromDense(tt MaskedTensor) *FlatMaskedIterator {
	it := new(FlatMaskedIterator)
	runtime.SetFinalizer(it, destroyIterator)
	it.FlatIterator = FlatIteratorFromDense(tt)
	it.mask = tt.Mask()
	return it
}

func (it *FlatMaskedIterator) NextValidity() (int, bool, error) {
	if len(it.mask) == 0 {
		return it.FlatIterator.NextValidity()
	}

	var i int
	var err error
	if i, err = it.Next(); err == nil {
		return i, !it.mask[i], err
	}
	return -1, false, err
}

// NextValid returns the index of the next valid element,
// as well as the number of increments to get to next element
func (it *FlatMaskedIterator) NextValid() (int, int, error) {
	if len(it.mask) == 0 {
		return it.FlatIterator.NextValid()
	}
	var count int
	var mult = 1
	if it.reverse {
		mult = -1
	}

	for i, err := it.Next(); err == nil; i, err = it.Next() {
		count++
		if !(it.mask[i]) {
			return i, mult * count, err
		}
	}
	return -1, mult * count, noopError{}
}

// NextInvalid returns the index of the next invalid element
// as well as the number of increments to get to next invalid element
func (it *FlatMaskedIterator) NextInvalid() (int, int, error) {
	if it.mask == nil {
		return it.FlatIterator.NextInvalid()
	}
	var count int
	var mult = 1
	if it.reverse {
		mult = -1
	}
	for i, err := it.Next(); err == nil; i, err = it.Next() {
		count++
		if it.mask[i] {
			return i, mult * count, err
		}
	}
	return -1, mult * count, noopError{}
}

// FlatSparseIterator is an iterator that works very much in the same way as flatiterator, except for sparse tensors
type FlatSparseIterator struct {
	*CS

	//state
	nextIndex int
	lastIndex int
	track     []int
	done      bool
	reverse   bool
}

func NewFlatSparseIterator(t *CS) *FlatSparseIterator {
	it := new(FlatSparseIterator)
	it.CS = t
	it.track = BorrowInts(len(t.s))
	return it
}

func (it *FlatSparseIterator) Start() (int, error) {
	it.Reset()
	return it.Next()
}

func (it *FlatSparseIterator) Next() (int, error) {
	if it.done {
		return -1, noopError{}
	}

	// var ok bool
	it.lastIndex, _ = it.at(it.track...)

	// increment the coordinates
	for i := len(it.s) - 1; i >= 0; i-- {
		it.track[i]++
		if it.track[i] == it.s[i] {
			if i == 0 {
				it.done = true
			}
			it.track[i] = 0
			continue
		}
		break
	}

	return it.lastIndex, nil
}

func (it *FlatSparseIterator) NextValidity() (int, bool, error) {
	i, err := it.Next()
	if i == -1 {
		return i, false, err
	}
	return i, true, err
}

func (it *FlatSparseIterator) NextValid() (int, int, error) {
	var i int
	var err error
	for i, err = it.Next(); err == nil && i == -1; i, err = it.Next() {

	}
	return i, -1, err
}

func (it *FlatSparseIterator) NextInvalid() (int, int, error) {
	var i int
	var err error
	for i, err = it.Next(); err == nil && i != -1; i, err = it.Next() {

	}
	return i, -1, err
}

func (it *FlatSparseIterator) Reset() {
	if it.reverse {
		for i := range it.track {
			it.track[i] = it.s[i] - 1
		}

	} else {
		it.nextIndex = 0
		for i := range it.track {
			it.track[i] = 0
		}
	}
	it.done = false
}

func (it *FlatSparseIterator) SetReverse() {
	it.reverse = true
	it.Reset()
}

func (it *FlatSparseIterator) SetForward() {
	it.reverse = false
	it.Reset()
}

func (it *FlatSparseIterator) Coord() []int {
	return it.track
}

func (it *FlatSparseIterator) Done() bool {
	return it.done
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
