package shapes

import "fmt"

type Shape []int

// Cons is an associative construction of shapes
func (s Shape) Cons(other Conser) Conser {
	switch ot := other.(type) {
	case Shape:
		return append(s, ot...)
	case Abstract:
		retVal := make(Abstract, 0, len(s)+len(ot))
		for i := range s {
			retVal = append(retVal, Size(s[i]))
		}
		retVal = append(retVal, ot...)
		return retVal
	}
	panic("Unreachable")
}

func (s Shape) isConser() {}

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

// Dims returns the number of dimensions in the shape
func (s Shape) Dims() int { return len(s) }

// IsScalar returns true if the access pattern indicates it's a scalar value
func (s Shape) IsScalar() bool { return len(s) == 0 }

// IsScalarEquiv returns true if the access pattern indicates it's a scalar-like value
func (s Shape) IsScalarEquiv() bool {
	if s.IsScalar() {
		return true
	}
	p := prodInts([]int(s))
	return p == 1 || p == 0
}

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

func (s Shape) TotalSize() int {
	panic("not implemented") // TODO: Implement
}

func (s Shape) DimSize(dim int) (Sizelike, error) {
	panic("not implemented") // TODO: Implement
}

func (s Shape) T(axes ...Axis) (newShape Shapelike, err error) {
	panic("not implemented") // TODO: Implement
}

func (s Shape) S(slices ...Slice) (newShape Shapelike, err error) {
	panic("not implemented") // TODO: Implement
}

func (s Shape) Repeat(axis Axis, repeats ...int) (newShape Shapelike, finalRepeats []int, size int, err error) {
	panic("not implemented") // TODO: Implement
}

func (s Shape) Concat(axis Axis, others ...Shapelike) (newShape Shapelike, err error) {
	panic("not implemented") // TODO: Implement
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

// apply doesn't apply any substitutions to Shape because there will not be anything to substitution.
func (s Shape) apply(_ substitutions) substitutable { return s }

// freevar returns nil because there are no free variables in a Shape.
func (s Shape) freevars() varset { return nil }

func (s Shape) isExpr() {}

// subExprs returns the shape as a slice of Expr (specifically, it becomes a slice of Size)
func (s Shape) subExprs() (retVal []substitutableExpr) {
	retVal = make([]substitutableExpr, 0, len(s))
	for i := range s {
		retVal = append(retVal, Size(s[i]))
	}
	return retVal
}
