package types

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

// MaxInts returns the of a slice of ints.
func MaxInts(is ...int) (retVal int) {
	for _, i := range is {
		if i > retVal {
			retVal = i
		}
	}
	return
}

func SumInts(a []int) (retVal int) {
	for _, v := range a {
		retVal += v
	}
	return
}

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
			err = DimMismatchErr(len(shape), i)
			return
		}

		size := shape[i]

		if coord >= size {
			err = IndexErr(i, coord, size)
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
				err = DimMismatchErr(len(strides), i)
				return
			}
			stride = strides[i]
		}
		at += stride * coord
	}
	return at, nil
}

func Itol(i int, shape Shape, strides []int) (coords []int, err error) {
	dims := len(strides)

	for d := 0; d < dims; d++ {
		var coord int
		coord, i = Divmod(i, strides[d])

		if coord >= shape[d] {
			err = IndexErr(d, coord, shape[d])
			// return
		}

		coords = append(coords, coord)
	}
	return
}

func UnsafePermute(pattern []int, xs ...[]int) (err error) {
	if len(xs) == 0 {
		err = NewError(OpError, "Permute requires something to permute")
		return
	}

	dims := -1
	patLen := len(pattern)
	for _, x := range xs {
		if dims == -1 {
			dims = len(x)
			if patLen != dims {
				err = DimMismatchErr(len(x), len(pattern))
				return
			}
		} else {
			if len(x) != dims {
				err = DimMismatchErr(len(x), len(pattern))
				return
			}
		}
	}

	// check that all the axes are < nDims
	// and that there are no axis repeated
	seen := make(map[int]struct{})
	for _, a := range pattern {
		if a >= dims {
			err = AxisErr("Invalid axis %d for this ndarray. Dims: %d", a, dims)
			return
		}

		if _, ok := seen[a]; ok {
			err = AxisErr("Repeated axis %d in permutation pattern", a)
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
		err = NewError(OpError, "Permute requires something to permute")
		return
	}

	dims := -1
	patLen := len(pattern)
	for _, x := range xs {
		if dims == -1 {
			dims = len(x)
			if patLen != dims {
				err = DimMismatchErr(len(x), len(pattern))
				return
			}
		} else {
			if len(x) != dims {
				err = DimMismatchErr(len(x), len(pattern))
				return
			}
		}
	}

	// check that all the axes are < nDims
	// and that there are no axis repeated
	seen := make(map[int]struct{})
	for _, a := range pattern {
		if a >= dims {
			err = AxisErr("Invalid axis %d for this ndarray. Dims: %d", a, dims)
			return
		}

		if _, ok := seen[a]; ok {
			err = AxisErr("Repeated axis %d in permutation pattern", a)
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

func sliceSanity(s Slice, size int) error {
	start := s.Start()
	end := s.End()
	step := s.Step()

	if start > end {
		return NewError(IndexError, "Start %d > End %d", start, end)
	}

	if start < 0 {
		return NewError(IndexError, "Start %d < 0", start)
	}

	if step == 0 && end-start > 1 {
		return NewError(IndexError, "Slice has 0 steps, but start is %d and end is %d", start, end)
	}

	if start >= size {
		return NewError(IndexError, "Start %d > size %d", start, size)
	}

	return nil
}
