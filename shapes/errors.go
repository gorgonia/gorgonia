package shapes

const (
	dimsMismatch      = "Dimension mismatch. Expected %v. Got  %v instead."
	invalidAxis       = "Invalid axis %d for ndarray with %d dimensions."
	repeatedAxis      = "repeated axis %d in permutation pattern."
	invalidSliceIndex = "Invalid slice index. Start: %d, End: %d."
	unaryOpResolveErr = "Cannot resolve %v to a Size."
)

// NoOpError is a useful for operations that have no op.
type NoOpError interface {
	NoOp() bool
}

type noopError struct{}

func (e noopError) NoOp() bool    { return true }
func (e noopError) Error() string { return "NoOp" }

// func handleNoOp(err error) error {
// 	if err == nil {
// 		return nil
// 	}

// 	if _, ok := err.(NoOpError); !ok {
// 		return err
// 	}
// 	return nil
// }
