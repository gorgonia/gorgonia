package tensor

import (
	"fmt"

	"github.com/pkg/errors"
)

var scalarShape = Shape{}

// ScalarShape represents a scalar. It has no dimensions, no sizes
func ScalarShape() Shape { return scalarShape }

// Shape represents the dimensions of a Tensor. A (2,3) matrix has a shape of (2,3) - 2 rows, 3 columns.
// Likewise, a shape of (2,3,4) means a Tensor has 3 dimensions: 2 layers, 3 rows, 4 columns.
//
// Vectors are of particular note. This package defines a shape of (x, 1) as a column vector and
// a (1, x) as a row vector. Row vectors and column vectors are matrices as well. It is important to note that
// row and column vectors and vanilla vectors are comparable under some circumstances
type Shape []int

// TotalSize returns the number of elements expected in a Tensor of a certain shape
func (s Shape) TotalSize() int {
	return ProdInts([]int(s))
}

func (s Shape) calcStrides() []int {
	if s.IsScalar() {
		return nil
	}

	retVal := BorrowInts(len(s))
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

// calcStridesWithMask is similar to calcStrides, except that it has an argument, masks. It is used to mask out given dimensions
// during calculation of stride
func (s Shape) calcStridesWithMask(mask []bool) []int {
	if s.IsScalar() {
		return nil
	}

	retVal := BorrowInts(len(s))
	if s.IsVector() {
		retVal[0] = 1
		retVal = retVal[:1]
		return retVal
	}

	if len(mask) != s.Dims() {
		panic("mask length must be equal to number of shape dimensions")
	}
	acc := 1
	for i := len(s) - 1; i >= 0; i-- {
		if mask[i] {
			retVal[i] = acc
		} else {
			retVal[i] = 0
		}
		d := s[i]
		if d < 0 {
			panic("negative dimension size does not make sense")
		}
		if mask[i] {
			acc *= d
		}
	}

	return retVal
}

func (s Shape) calcStridesColMajor() []int {
	if s.IsScalar() {
		return nil
	}

	retVal := BorrowInts(len(s))
	if s.IsVector() {
		retVal[0] = 1
		retVal = retVal[:1]
		return retVal
	}

	acc := 1
	for i := 0; i < len(s); i++ {
		retVal[i] = acc
		d := s[i]
		if d < 0 {
			panic("negative dimension size does not make sense")
		}
		acc *= d
	}
	return retVal
}

// Eq indicates if a shape is equal with another. There is a soft concept of equality when it comes to vectors.
//
// If s is a column vector and other is a vanilla vector, they're considered equal if the size of the column dimension is the same as the vector size;
// if s is a row vector and other is a vanilla vector, they're considered equal if the size of the row dimension is the same as the vector size
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

// Clone clones a shape.
func (s Shape) Clone() Shape {
	retVal := BorrowInts(len(s))
	// retVal := make(Shape, len(s), len(s))
	copy(retVal, s)
	return retVal
}

// IsScalar returns true if the access pattern indicates it's a scalar value
func (s Shape) IsScalar() bool { return len(s) == 0 || (len(s) == 1 && s[0] == 1) }

// IsVector returns whether the access pattern falls into one of three possible definitions of vectors:
//		vanilla vector (not a row or a col)
//		column vector
//		row vector
func (s Shape) IsVector() bool { return s.IsColVec() || s.IsRowVec() || (len(s) == 1 && s[0] > 1) }

// IsColVec returns true when the access pattern has the shape (x, 1)
func (s Shape) IsColVec() bool { return len(s) == 2 && (s[1] == 1 && s[0] > 1) }

// IsRowVec returns true when the access pattern has the shape (1, x)
func (s Shape) IsRowVec() bool { return len(s) == 2 && (s[0] == 1 && s[1] > 1) }

// IsMatrix returns true if it's a matrix. This is mostly a convenience method. RowVec and ColVecs are also considered matrices
func (s Shape) IsMatrix() bool { return len(s) == 2 }

// Dims returns the number of dimensions in the shape
func (s Shape) Dims() int { return len(s) }

func (s Shape) DimSize(d int) (size int, err error) {
	if (s.IsScalar() && d != 0) || (!s.IsScalar() && d >= len(s)) {
		err = errors.Errorf(dimMismatch, len(s), d)
		return
	}

	switch {
	case s.IsScalar():
		return 0, nil
	default:
		return s[d], nil
	}
}

// S gives the new shape after a shape has been sliced. It's repeated from the AP S() method mainly because there are other functions in Gorgonia that uses only shape
func (s Shape) S(slices ...Slice) (retVal Shape, err error) {
	opDims := len(s)
	if len(slices) > opDims {
		err = errors.Errorf(dimMismatch, opDims, len(slices))
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

// Repeat returns the expected new shape given the repetition parameters.
func (s Shape) Repeat(axis int, repeats ...int) (newShape Shape, finalRepeats []int, size int, err error) {
	switch {
	case axis == AllAxes:
		size = s.TotalSize()
		newShape = Shape{size}
		axis = 0
	case s.IsScalar():
		size = 1
		// special case for row vecs
		if axis == 1 {
			newShape = Shape{1, 0}
		} else {
			// otherwise it will be repeated into a vanilla vector
			newShape = Shape{0}
		}
	case s.IsVector() && !s.IsRowVec() && !s.IsColVec() && axis == 1:
		size = 1
		newShape = s.Clone()
		newShape = append(newShape, 1)
	default:
		if axis >= len(s) {
			// error
			err = errors.Errorf(invalidAxis, axis, s.Dims())
			return
		}
		size = s[axis]
		newShape = s.Clone()
	}

	// special case to allow generic repeats
	if len(repeats) == 1 {
		rep := repeats[0]
		repeats = make([]int, size)
		for i := range repeats {
			repeats[i] = rep
		}
	}
	reps := len(repeats)
	if reps != size {
		err = errors.Errorf(broadcastError, size, reps)
		return
	}

	newSize := SumInts(repeats)
	newShape[axis] = newSize
	finalRepeats = repeats
	return
}

// Concat returns the expected new shape given the concatenation parameters
func (s Shape) Concat(axis int, ss ...Shape) (newShape Shape, err error) {
	dims := s.Dims()

	// check that all the concatenates have the same dimensions
	for _, shp := range ss {
		if shp.Dims() != dims {
			err = errors.Errorf(dimMismatch, dims, shp.Dims())
			return
		}
	}

	// special case
	if axis == AllAxes {
		axis = 0
	}

	// nope... no negative indexing here.
	if axis < 0 {
		err = errors.Errorf(invalidAxis, axis, len(s))
		return
	}

	if axis >= dims {
		err = errors.Errorf(invalidAxis, axis, len(s))
		return
	}

	newShape = Shape(BorrowInts(dims))
	copy(newShape, s)

	for _, shp := range ss {
		for d := 0; d < dims; d++ {
			if d == axis {
				newShape[d] += shp[d]
			} else {
				// validate that the rest of the dimensions match up
				if newShape[d] != shp[d] {
					err = errors.Errorf(dimMismatch, newShape[d], shp[d])
					return
				}
			}
		}
	}
	return
}

// Format implements fmt.Formatter, and formats a shape nicely
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
