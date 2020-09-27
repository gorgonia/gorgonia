package shapes

import "unsafe"

func prodInts(a []int) int {
	if len(a) == 0 {
		return 0
	}
	retVal := a[0]
	for i := 1; i < len(a); i++ {
		retVal *= a[i]
	}
	return retVal
}

func axesToInts(a Axes) []int {
	return *(*[]int)(unsafe.Pointer(&a))
}
