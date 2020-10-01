package shapes

import (
	"unsafe"

	"github.com/pkg/errors"
)

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

// IsMonotonicInts returns true if the slice of ints is monotonically increasing. It also returns true for incr1 if every succession is a succession of 1
func IsMonotonicInts(a []int) (monotonic bool, incr1 bool) {
	var prev int
	incr1 = true
	for i, v := range a {
		if i == 0 {
			prev = v
			continue
		}

		if v < prev {
			return false, false
		}
		if v != prev+1 {
			incr1 = false
		}
		prev = v
	}
	monotonic = true
	return
}

func UnsafePermute(pattern []int, xs ...[]int) (err error) {
	if len(xs) == 0 {
		err = errors.New("Permute requres something to permute")
		return
	}

	dims := -1
	patLen := len(pattern)
	for _, x := range xs {
		if dims == -1 {
			dims = len(x)
			if patLen != dims {
				err = errors.Errorf(dimMismatch, len(x), len(pattern))
				return
			}
		} else {
			if len(x) != dims {
				err = errors.Errorf(dimMismatch, len(x), len(pattern))
				return
			}
		}
	}

	// check that all the axes are < nDims
	// and that there are no axis repeated
	seen := make(map[int]struct{})
	for _, a := range pattern {
		if a >= dims {
			err = errors.Errorf(invalidAxis, a, dims)
			return
		}

		if _, ok := seen[a]; ok {
			err = errors.Errorf(repeatedAxis, a)
			return
		}

		seen[a] = struct{}{}
	}

	// no op really... we did the checks for no reason too. Maybe move this up?
	if monotonic, incr1 := IsMonotonicInts(pattern); monotonic && incr1 {
		err = noopError{}
		return
	}

	switch dims {
	case 0, 1:
	case 2:
		for _, x := range xs {
			x[0], x[1] = x[1], x[0]
		}
	default:
		for i := 0; i < dims; i++ {
			to := pattern[i]
			for to < i {
				to = pattern[to]
			}
			for _, x := range xs {
				x[i], x[to] = x[to], x[i]
			}
		}
	}
	return nil
}
