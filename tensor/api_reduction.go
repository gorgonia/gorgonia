package tensor

// Sum sums a Tensor along the given axes
func Sum(t Tensor, along ...int) (retVal Tensor, err error) {
	switch T := t.(type) {
	case *Dense:
		return T.Sum(along...)
	}
	panic("Unreachable")
}

// Argmax finds the index of the max value along the axis provided
func Argmax(t Tensor, axis int) (retVal Tensor, err error) {
	switch T := t.(type) {
	case *Dense:
		return T.Argmax(axis)
	}
	panic("Unreachable")
}

// Argmin finds the index of the min value along the axis provided
func Argmin(t Tensor, axis int) (retVal Tensor, err error) {
	switch T := t.(type) {
	case *Dense:
		return T.Argmin(axis)
	}
	panic("Unreachable")
}
