package tensor

// NoOpError is a useful for operations that have no op.
type NoOpError interface {
	NoOp() bool
}

type noopError struct{}

func (e noopError) NoOp() bool    { return true }
func (e noopError) Error() string { return "NoOp" }

const (
	emptyTensor       = "Tensor is uninitialized (no shape, no data)"
	dimMismatch       = "Dimension mismatch. Expected %d, got %d"
	dtypeMismatch     = "Dtype mismatch. Expected %v. Got %v"
	indexOOBAxis      = "Index %d is out of bounds for axis %d which has size %d"
	invalidAxis       = "Invalid axis %d for ndarray with %d dimensions"
	repeatedAxis      = "repeated axis %d in permutation pattern"
	invalidSliceIndex = "Invalid slice index. Start: %d, End: %d"
	broadcastError    = "Cannot broadcast together. Resulting shape will be at least (%d, 1). Repeats is (%d, 1)"
	lenMismatch       = "Cannot compare with differing lengths: %d and %d"
	typeMismatch      = "TypeMismatch: Op %q cannot be performed. a %T and b %T"
	shapeMismatch     = "Shape Mismatch. Coordinates has %d dimensions, ndarry has %d dimensions"
	sizeMismatch      = "Size Mismatch. %d and %d"
	reuseReshapeErr   = "Failed to reshape the reuse *Dense from into %v. Size was: %d"
	incrReshapeErr    = "Failed to reshape the incr *Dense into %v. Size was: %d"
	retValReshapeErr  = "Failed to reshape the retVal *Dense into %v. Size was: %d"

	methodNYI = "%q not yet implemented for %v"
)
