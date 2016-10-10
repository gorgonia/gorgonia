package types

import "fmt"

func ScalarShape() Shape { return Shape{} }

type Shape []int
type Strides []int

func (s Shape) TotalSize() int {
	return ProdInts([]int(s))
}

func (s Shape) CalcStrides() []int {
	// retVal := make([]int, len(s))
	retVal := BorrowInts(len(s))

	if s.IsScalar() {
		return nil
	}

	if s.IsVector() {
		retVal[0] = 1
		retVal = retVal[:1]
		return retVal
	}

	acc := 1
	for i := len(s) - 1; i >= 0; i-- {
		retVal[i] = acc
		d := s[i]
		if d < 0 {
			panic("negative dimension size does not make sense")
		}
		acc *= d
	}
	return retVal
}

func (s Shape) Eq(other Shape) bool {
	if s.IsScalar() && other.IsScalar() {
		return true
	}

	if s.IsVector() && other.IsVector() {
		switch {
		case len(s) == 2 && len(other) == 1:
			if (s.IsColVec() && s[0] == other[0]) || (s.IsRowVec() && s[1] == other[0]) {
				return true
			}
			return false
		case len(s) == 1 && len(other) == 2:
			if (other.IsColVec() && other[0] == s[0]) || (other.IsRowVec() && other[1] == s[0]) {
				return true
			}
			return false
			// case s.IsColVec() && other.IsRowVec():
			// 	return false
			// case s.IsRowVec() && other.IsColVec():
			// 	return false
		}
	}

	if len(s) != len(other) {
		return false
	}

	for i, v := range s {
		if other[i] != v {
			return false
		}
	}
	return true
}

func (s Shape) Clone() Shape {
	retVal := make(Shape, len(s))
	copy(retVal, s)
	return retVal
}

func (s Shape) IsScalar() bool {
	return len(s) == 0 || (len(s) == 1 && s[0] == 1)
}

func (s Shape) IsVector() bool {
	return s.IsColVec() || s.IsRowVec() || (len(s) == 1 && s[0] > 1)
}

func (s Shape) IsColVec() bool {
	return len(s) == 2 && (s[1] == 1 && s[0] > 1)
}

func (s Shape) IsRowVec() bool {
	return len(s) == 2 && (s[0] == 1 && s[1] > 1)
}

func (s Shape) Dims() int {
	switch {
	case s.IsScalar():
		return 0
	case s.IsVector():
		return 1
	default:
		return len(s)
	}
	panic("Unreachable")
}

// S gives the new shape after a shape has been sliced. It's repeated from the AP S() method mainly because there are other functions in Gorgonia that uses only shape
func (s Shape) S(slices ...Slice) (retVal Shape, err error) {
	opDims := len(s)
	if len(slices) > opDims {
		err = DimMismatchErr(opDims, len(slices))
		return
	}

	retVal = s.Clone()

	for d, size := range s {
		var sl Slice // default is a nil Slice
		if d <= len(slices)-1 {
			sl = slices[d]
		}

		var start, end, step int
		if start, end, step, err = SliceDetails(sl, size); err != nil {
			return
		}

		if step > 0 {
			retVal[d] = (end - start) / step

			//fix
			if retVal[d] <= 0 {
				retVal[d] = 1
			}
		} else {
			retVal[d] = (end - start)
		}

	}

	// drop any dimension with size 1, except the last dimension
	dims := s.Dims()
	for d := 0; d < dims; d++ {
		if retVal[d] == 1 /*&& d != t.dims-1  && dims > 2*/ {
			retVal = append(retVal[:d], retVal[d+1:]...)
			d--
			dims--
		}
	}

	if retVal.IsScalar() {
		ReturnInts(retVal)
		return ScalarShape(), nil
	}

	return
}

func (s Shape) Format(st fmt.State, r rune) {
	switch r {
	case 'v', 's':
		st.Write([]byte("("))
		for i, v := range s {
			fmt.Fprintf(st, "%d", v)
			if i < len(s)-1 {
				st.Write([]byte(", "))
			}
		}
		st.Write([]byte(")"))
	default:
		fmt.Fprintf(st, "%v", []int(s))
	}
}
