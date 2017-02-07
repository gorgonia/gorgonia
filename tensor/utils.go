package tensor

import "github.com/pkg/errors"

const AllAxes int = -1

// MinInt returns the lowest between two ints. If both are the  same it returns the first
func MinInt(a, b int) int {
	if a <= b {
		return a
	}
	return b
}

// MaxInt returns the highest between two ints. If both are the same, it  returns the first
func MaxInt(a, b int) int {
	if a >= b {
		return a
	}
	return b
}

// MaxInts returns the max of a slice of ints.
func MaxInts(is ...int) (retVal int) {
	for _, i := range is {
		if i > retVal {
			retVal = i
		}
	}
	return
}

// SumInts sums a slice of ints
func SumInts(a []int) (retVal int) {
	for _, v := range a {
		retVal += v
	}
	return
}

// ProdInts returns the internal product of an int slice
func ProdInts(a []int) (retVal int) {
	if len(a) == 0 {
		return
	}
	retVal = 1
	for _, v := range a {
		retVal *= v
	}
	return
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

// Ltoi is Location to Index. Provide a shape, a strides, and a list of integers as coordinates, and returns the index at which the element is.
func Ltoi(shape Shape, strides []int, coords ...int) (at int, err error) {
	for i, coord := range coords {
		if i >= len(shape) {
			err = errors.Errorf(dimMismatch, len(shape), i)
			return
		}

		size := shape[i]

		if coord >= size {
			err = errors.Errorf(indexOOBAxis, i, coord, size)
			return
		}

		var stride int
		if shape.IsRowVec() {
			if i == 0 && len(coords) == 2 {
				continue
			}
			stride = strides[0]
		} else if shape.IsColVec() {
			if i == 1 && len(coords) == 2 {
				continue
			}
			stride = strides[0]
		} else {
			if i >= len(strides) {
				err = errors.Errorf(dimMismatch, len(strides), i)
				return
			}
			stride = strides[i]
		}
		at += stride * coord
	}
	return at, nil
}

// Itol is Index to Location.
func Itol(i int, shape Shape, strides []int) (coords []int, err error) {
	dims := len(strides)

	for d := 0; d < dims; d++ {
		var coord int
		coord, i = divmod(i, strides[d])

		if coord >= shape[d] {
			err = errors.Errorf(indexOOBAxis, d, coord, shape[d])
			// return
		}

		coords = append(coords, coord)
	}
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

func Permute(pattern []int, xs ...[]int) (retVal [][]int, err error) {
	if len(xs) == 0 {
		err = errors.New("Permute requires something to permute")
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
		retVal = xs
		err = noopError{}
		return
	}

	switch dims {
	case 0, 1:
		retVal = xs
	case 2:
		for _, x := range xs {
			rv := []int{x[1], x[0]}
			retVal = append(retVal, rv)
		}
	default:
		retVal = make([][]int, len(xs))
		for i := range retVal {
			retVal[i] = make([]int, dims)
		}

		for i, v := range pattern {
			for j, x := range xs {
				retVal[j][i] = x[v]
			}
		}
	}
	return
}

// CheckSlice checks a slice to see if it's sane
func CheckSlice(s Slice, size int) error {
	start := s.Start()
	end := s.End()
	step := s.Step()

	if start > end {
		return errors.Errorf(invalidSliceIndex, start, end)
	}

	if start < 0 {
		return errors.Errorf(invalidSliceIndex, start, 0)
	}

	if step == 0 && end-start > 1 {
		return errors.Errorf("Slice has 0 steps. Start is %d and end is %d", start, end)
	}

	if start >= size {
		return errors.Errorf("Start %d is greater than size %d", start, size)
	}

	return nil
}

// SliceDetails is a function that takes a slice and spits out its details. The whole reason for this is to handle the nil Slice, which is this: a[:]
func SliceDetails(s Slice, size int) (start, end, step int, err error) {
	if s == nil {
		start = 0
		end = size
		step = 1
	} else {
		if err = CheckSlice(s, size); err != nil {
			return
		}

		start = s.Start()
		end = s.End()
		step = s.Step()

		if end > size {
			end = size
		}
	}
	return
}

// reuseDenseCheck checks a reuse tensor, and reshapes it to be the correct one
func reuseDenseCheck(reuse *Dense, as *Dense) (err error) {
	if reuse.DataSize() != as.Size() {
		err = errors.Errorf("Reused Tensor %p does not have expected shape %v. Got %v instead. Reuse Size: %v, as Size %v (real: %d)", reuse, as.Shape(), reuse.Shape(), reuse.DataSize(), as.Size(), as.DataSize())
		return
	}
	return reuseCheckShape(reuse, as.Shape())

}

// reuseCheckShape  checks the shape and reshapes it to be correct if the size fits but the shape doesn't.
func reuseCheckShape(reuse *Dense, s Shape) (err error) {
	throw := BorrowInts(len(s))
	copy(throw, s)

	if err = reuse.reshape(throw...); err != nil {
		err = errors.Wrapf(err, reuseReshapeErr, s, reuse.DataSize())
		return
	}

	// clean up any funny things that may be in the reuse
	if reuse.old != nil {
		ReturnAP(reuse.old)
		reuse.old = nil
	}

	if reuse.transposeWith != nil {
		ReturnInts(reuse.transposeWith)
	}

	if reuse.viewOf != nil {
		reuse.viewOf = nil
	}
	return nil
}
