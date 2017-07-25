// package perf is an internal package for managing memory pools.
// It's fundamentally a bunch of sync.Pools
package perf

import "sync"

const PoolSize = 16

/* INTS POOL */

var intsPool [PoolSize]sync.Pool

/* BOOLS POOL */

var boolsPool [PoolSize]sync.Pool

// Init function
func init() {
	for i := range intsPool {
		size := i
		intsPool[i].New = func() interface{} { return make([]int, size) }
	}

	for i := range boolsPool {
		size := i
		boolsPool[i].New = func() interface{} { return make([]bool, size) }
	}
}

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
