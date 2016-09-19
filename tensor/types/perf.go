package types

import "sync"

const (
	maxAPDims = 8
)

var intsPool [8]sync.Pool

func init() {
	for i := range intsPool {
		size := i
		intsPool[i].New = func() interface{} { return make([]int, size) }
	}

	for i := range apPool {
		l := i
		apPool[i].New = func() interface{} {
			ap := new(AP)
			ap.dims = l
			ap.strides = make([]int, l)
			ap.shape = make(Shape, l)
			return ap
		}
	}
}

func BorrowInts(size int) []int {
	if size >= 8 {
		return make([]int, 8)
	}

	return intsPool[size].Get().([]int)
}

func ReturnInts(ints []int) {
	size := cap(ints)
	if size >= 8 {
		return
	}
	ints = ints[:cap(ints)]
	for i := range ints {
		ints[i] = 0
	}

	intsPool[size].Put(ints)
}

// apPool supports tensors up to 4-dimensions. Because, c'mon, you're not likely to use anything more than 5
var apPool [maxAPDims]sync.Pool

func BorrowAP(dims int) *AP {
	if dims >= maxAPDims {
		ap := new(AP)
		ap.shape = make(Shape, dims)
		ap.strides = make([]int, dims)
		return ap
	}

	ap := apPool[dims].Get().(*AP)

	// restore strides and shape to whatever that may have been truncated
	ap.strides = ap.strides[:cap(ap.strides)]
	return ap
}

func ReturnAP(ap *AP) {
	if ap.dims >= maxAPDims {
		return
	}
	apPool[ap.dims].Put(ap)
}
