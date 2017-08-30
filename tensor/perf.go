package tensor

import (
	"runtime"
	"sync"

	"github.com/chewxy/gorgonia/tensor/internal/storage"
)

var habbo sync.Mutex
var usePool = true

// tensorPool is a pool of *Tensor grouped by size. It's guarded by poolsClosed

const (
	maxAPDims = 8
	maxDims   = 8
	PoolSize  = 4096
)

// UsePool enables the use of a pool of *Tensors as provided in the package. This is the default option
func UsePool() {
	habbo.Lock()
	usePool = true
	habbo.Unlock()
}

// DontUsePool makes sure the functions don't use the tensor pool provided.
// This is useful as certain applications don't lend themselves well to use of the pool.
// Examples of such applications would be one where many tensors of wildly different sizes are created all the time.
func DontUsePool() {
	habbo.Lock()
	usePool = false
	habbo.Unlock()
}

// headerPool should ever only be used by scalarToHeader
var headerPool = make(chan *storage.Header, PoolSize)

func borrowHeader() *storage.Header {
	select {
	case hdr := <-headerPool:
		return hdr
	default:
		hdr := new(storage.Header)
		runtime.SetFinalizer(hdr, destroyHeader)
		return hdr
	}
}

func returnHeader(hdr *storage.Header) {
	destroyHeader(hdr)
	if len(headerPool) < cap(headerPool) {
		headerPool <- hdr
	}
}

func destroyHeader(hdr *storage.Header) {
	hdr.Ptr = nil
	hdr.L = 0
	hdr.C = 0
}

var densePool = make(chan *Dense, PoolSize)

func borrowDense() *Dense {
	select {
	case t := <-densePool:
		return t
	default:
		t := new(Dense)
		t.e = StdEng{}
		return t
	}
	// return densePool.Get().(*Dense)
}

// ReturnTensor returns a Tensor to their respective pools. USE WITH CAUTION
func ReturnTensor(t Tensor) {
	if !usePool {
		return
	}
	switch tt := t.(type) {
	case *Dense:
		if tt.old != nil {
			ReturnAP(tt.old)
			tt.old = nil
		}

		if tt.transposeWith != nil {
			ReturnInts(tt.transposeWith)
			tt.transposeWith = nil
		}

		// return AP
		ReturnAP(tt.AP)

		// array reset
		tt.t = Dtype{}
		tt.array.Ptr = nil
		tt.array.L = 0
		tt.array.C = 0
		tt.array.v = nil

		// engine and flag reset
		tt.e = StdEng{}
		tt.flag = 0

		// other reset
		tt.old = nil
		tt.viewOf = 0
		tt.transposeWith = nil

		// mask related stuff - TODO: deprecate
		tt.mask = nil
		tt.maskIsSoft = false

		// densePool.Put(tt)
		if len(densePool) < cap(densePool) {
			densePool <- tt
		}
	}
}

/* AP POOL */

var apPool = make(chan *AP, PoolSize)

func borrowAP() *AP {
	select {
	case ap := <-apPool:
		return ap
	default:
		return new(AP)
	}
	// return apPool.Get().(*AP)
}

// BorrowAP gets an AP from the pool. USE WITH CAUTION.
func BorrowAP(dims int) *AP {
	ap := borrowAP()
	ap.shape = BorrowInts(dims)
	ap.strides = BorrowInts(dims)
	ap.shape = ap.shape[:cap(ap.shape)]
	ap.strides = ap.strides[:cap(ap.strides)]
	return ap
}

// ReturnAP returns the AP to the pool. USE WITH CAUTION.
func ReturnAP(ap *AP) {
	ReturnInts([]int(ap.shape))
	ReturnInts(ap.strides)
	ap.fin = false

	ap.o = 0
	ap.Δ = 0

	if len(apPool) < cap(apPool) {
		apPool <- ap
	}
	// apPool.Put(ap)
}

/* ----------------------------------------------------------------
------------------ Create Pools
------------------------------------------------------------------*/

/* APLIST POOL */

var apListPool [maxAPDims]sync.Pool

// Init function
func init() {
	for i := range apListPool {
		size := i
		apListPool[i].New = func() interface{} { return make([]*AP, size) }
	}

	for i := 0; i < PoolSize; i++ {
		intsPool <- make([]int, 8, 8)
	}

	// for i := range intsPool {
	// 	size := i
	// 	intsPool[i].New = func() interface{} { return make([]int, size) }
	// }

	// for i := range boolsPool {
	// 	size := i
	// 	boolsPool[i].New = func() interface{} { return make([]bool, size) }
	// }
}

/* INTS POOL */

// var intsPool [PoolSize]sync.Pool
var intsPool = make(chan []int, PoolSize)

/* BOOLS POOL */
var boolsPool = make(chan []bool, PoolSize)

// var boolsPool [PoolSize]sync.Pool

var check bool

// BorrowInts borrows a slice of ints from the pool. USE WITH CAUTION.
func BorrowInts(size int) []int {
	if size > 8 {
		return make([]int, size, size)
	}

	select {
	case ints := <-intsPool:
		ints = ints[:size]
		return ints
	default:
		ints := make([]int, size, 8)
		return ints
	}
	// retVal := intsPool[size].Get()
	// if retVal == nil {
	// 	return make([]int, size)
	// }
	// return retVal.([]int)
}

// ReturnInts returns a slice from the pool. USE WITH CAUTION.
func ReturnInts(is []int) {
	if is == nil {
		return
	}
	size := cap(is)
	if size != 8 {
		return
	}
	is = is[:cap(is)]
	for i := range is {
		is[i] = 0
	}

	if len(intsPool) < cap(intsPool) {
		intsPool <- is
	}

	// intsPool[size].Put(is)
}

// BorrowBools borrows a slice of bools from the pool. USE WITH CAUTION.
func BorrowBools(size int) []bool {
	if size >= 8 {
		return make([]bool, size)
	}

	select {
	case bools := <-boolsPool:
		return bools
	default:
		bools := make([]bool, 8)
		bools = bools[:size]
		return bools
	}

	// retVal := boolsPool[size].Get()
	// if retVal == nil {
	// 	return make([]bool, size)
	// }
	// return retVal.([]bool)
}

// ReturnBools returns a slice from the pool. USE WITH CAUTION.
func ReturnBools(is []bool) {
	if is == nil {
		return
	}
	size := cap(is)
	if size >= 8 {
		return
	}
	is = is[:cap(is)]
	for i := range is {
		is[i] = false
	}

	if len(boolsPool) < cap(boolsPool) {
		boolsPool <- is
	}
	// boolsPool[size].Put(is)
}

// BorrowAPList gets an APList from the pool. USE WITH CAUTION.
func BorrowAPList(size int) []*AP {
	if size >= 8 {
		return make([]*AP, size)
	}

	retVal := apListPool[size].Get()
	if retVal == nil {
		return make([]*AP, size)
	}
	return retVal.([]*AP)
}

// ReturnAPList returns the APList to the pool. USE WITH CAUTION.
func ReturnAPList(aps []*AP) {
	if aps == nil {
		return
	}
	size := cap(aps)
	if size >= 8 {
		return
	}
	aps = aps[:cap(aps)]
	for i := range aps {
		aps[i] = nil
	}

	apListPool[size].Put(aps)
}

// var optPool = make(chan *OpOpt, PoolSize)
// var optPool = newRingbuffer(PoolSize)
var optPool = &sync.Pool{
	New: func() interface{} { return new(OpOpt) },
}

func borrowOpOpt() *OpOpt {
	// select {
	// case fo := <-optPool:
	// 	return fo
	// default:
	// 	return new(OpOpt)
	// }

	return optPool.Get().(*OpOpt)

	// if fo, err := optPool.Get(); err == nil {
	// 	return (*OpOpt)(fo)
	// }
	// return new(OpOpt)
}

func returnOpOpt(oo *OpOpt) {
	oo.reuse = nil
	oo.incr = nil
	oo.unsafe = false
	oo.same = false
	oo.t = Dtype{}
	// if len(optPool) < cap(optPool) {
	// 	optPool <- oo
	// }

	optPool.Put(oo)

	// optPool.Put(unsafe.Pointer(oo))
}
