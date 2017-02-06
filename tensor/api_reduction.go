package tensor

// Sum sums a Tensor along the given axes
func Sum(t Tensor, along ...int) (retVal Tensor, err error) {
	switch T := t.(type) {
	case *Dense:
		return T.Sum(along...)
	}
	panic("Unreachable")
}
