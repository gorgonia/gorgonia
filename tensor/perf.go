package tensor

import "sync"

var habbo sync.Mutex
var usePool = true

// tensorPool is a pool of *Tensor grouped by size. It's guarded by poolsClosed
var densePool = &sync.Pool{
	New: func() interface{} { d := new(Dense); d.e = StdEng{}; return d },
}

const (
	maxAPDims = 8
	PoolSize  = 16
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

func borrowDense() *Dense {
	return densePool.Get().(*Dense)
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
		tt.array.Ptr = nil
		tt.array.L = 0
		tt.array.C = 0

		// engine and flag reset
		tt.e = StdEng{}
		tt.flag = 0

		// other reset
		tt.old = nil
		tt.viewOf = nil
		tt.transposeWith = nil

		// mask related stuff - TODO: deprecate
		tt.mask = nil
		tt.maskIsSoft = false

		densePool.Put(tt)
	}
}

/* AP POOL */

var apPool = &sync.Pool{
	New: func() interface{} { return new(AP) },
}

func borrowAP() *AP {
	return apPool.Get().(*AP)
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

	for i := range intsPool {
		size := i
		intsPool[i].New = func() interface{} { return make([]int, size) }
	}

	for i := range boolsPool {
		size := i
		boolsPool[i].New = func() interface{} { return make([]bool, size) }
	}
}

/* INTS POOL */

var intsPool [PoolSize]sync.Pool

/* BOOLS POOL */

var boolsPool [PoolSize]sync.Pool

// BorrowInts borrows a slice of ints from the pool. USE WITH CAUTION.
func BorrowInts(size int) []int {
	if size >= 8 {
		return make([]int, size)
	}

	retVal := intsPool[size].Get()
	if retVal == nil {
		return make([]int, size)
	}
	return retVal.([]int)
}

// ReturnInts returns a slice from the pool. USE WITH CAUTION.
func ReturnInts(is []int) {
	if is == nil {
		return
	}
	size := cap(is)
	if size >= 8 {
		return
	}
	is = is[:cap(is)]
	for i := range is {
		is[i] = 0
	}

	intsPool[size].Put(is)
}

// BorrowBools borrows a slice of bools from the pool. USE WITH CAUTION.
func BorrowBools(size int) []bool {
	if size >= 8 {
		return make([]bool, size)
	}

	retVal := boolsPool[size].Get()
	if retVal == nil {
		return make([]bool, size)
	}
	return retVal.([]bool)
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

	boolsPool[size].Put(is)
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

var optPool *sync.Pool = &sync.Pool{
	New: func() interface{} { return new(OpOpt) },
}

func borrowOpOpt() *OpOpt {
	return optPool.Get().(*OpOpt)
}

func returnOpOpt(oo *OpOpt) {
	optPool.Put(oo)
}
