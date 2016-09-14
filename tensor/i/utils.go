package tensori

import "math/rand"

func RandomInt(size int) []int {
	r := make([]int, size)
	for i := range r {
		r[i] = rand.Int()
	}
	return r
}

// RangeFloat is inclusive of Start AND End
func RangeInt(start, end int) []int {
	size := end - start
	incr := true
	if start > end {
		incr = false
		size = start - end
	}

	if size < 0 {
		panic("Cannot create a float range that is negative in size")
	}

	r := make([]int, size)
	for i, v := 0, int(start); i < size; i++ {
		r[i] = v

		if incr {
			v++
		} else {
			v--
		}
	}
	return r
}

func reduce(f func(a, b int) int, def int, l ...int) (retVal int) {
	retVal = def
	if len(l) == 0 {
		return
	}

	for _, v := range l {
		retVal = f(retVal, v)
	}
	return
}

func zeroAll(a []int) {
	for i := range a {
		a[i] = 0
	}
}

func boolsToInts(a []bool) []int {
	retVal := make([]int, len(a))
	for i, v := range a {
		if v {
			retVal[i] = int(1)
		} else {
			retVal[i] = int(0)
		}
	}
	return retVal
}

func argmax(a []int) int {
	var f int
	var max int
	var set bool
	for i, v := range a {
		if !set {
			f = v
			max = i
			set = true

			continue
		}

		// TODO: Maybe error instead of this?

		if v > f {
			max = i
			f = v
		}
	}
	return max
}
