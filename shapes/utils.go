package shapes

import "unsafe"

type exprtup struct {
	a, b Expr
}

func (t exprtup) freevars() varset {
	retVal := t.a.freevars()
	retVal = append(retVal, t.b.freevars()...)
	return unique(retVal)
}

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

func arrowToTup(a *Arrow) *exprtup {
	return (*exprtup)(unsafe.Pointer(a))
}
