package tensor

import "github.com/pkg/errors"

func overlaps(a, b *Dense) bool {
	if a.cap() == 0 || b.cap() == 0 {
		return false
	}

	if a.ptr == b.ptr {
		return true
	}
	aptr := uintptr(a.ptr)
	bptr := uintptr(b.ptr)

	capA := aptr + uintptr(a.c)*a.t.Size()
	capB := bptr + uintptr(b.c)*b.t.Size()

	switch {
	case aptr < bptr:
		if bptr < capA {
			return true
		}
	case aptr > bptr:
		if aptr < capB {
			return true
		}
	}
	return false
}

func assignArray(dest, src *Dense) (err error) {
	// var copiedSrc bool

	if src.IsScalar() {
		panic("HELP")
	}

	dd := dest.Dims()
	sd := src.Dims()

	dstrides := dest.Strides()
	sstrides := src.Strides()

	var ds, ss int
	ds = dstrides[0]
	if src.IsVector() {
		ss = sstrides[0]
	} else {
		ss = sstrides[sd-1]
	}

	// when dd == 1, and the strides point in the same direction
	// we copy to a temporary if there is an overlap of data
	if ((dd == 1 && sd >= 1 && ds*ss < 0) || dd > 1) && overlaps(dest, src) {
		// create temp
		// copiedSrc = true
	}

	// broadcast src to dest for raw iteration
	tmpShape := Shape(BorrowInts(sd))
	tmpStrides := BorrowInts(len(src.Strides()))
	copy(tmpShape, src.Shape())
	copy(tmpStrides, src.Strides())
	defer ReturnInts(tmpShape)
	defer ReturnInts(tmpStrides)

	if sd > dd {
		tmpDim := sd
		for tmpDim > dd && tmpShape[0] == 1 {
			tmpDim--

			// this is better than tmpShape = tmpShape[1:]
			// because we are going to return these ints later
			copy(tmpShape, tmpShape[1:])
			copy(tmpStrides, tmpStrides[1:])
		}
	}

	var newStrides []int
	if newStrides, err = BroadcastStrides(dest.Shape(), tmpShape, dstrides, tmpStrides); err != nil {
		err = errors.Wrapf(err, "BroadcastStrides failed")
		return
	}
	dap := dest.AP
	sap := NewAP(tmpShape, newStrides)
	sap.o = src.AP.o

	diter := NewFlatIterator(dap)
	siter := NewFlatIterator(sap)
	_, err = copyDenseIter(dest, src, diter, siter)
	return
}
