package tensori

import "github.com/chewxy/gorgonia/tensor/types"

const (
	reuseReshapeErr  = "Failed to reshape the reuse *Tensor into %v. Size was: %d"
	incrReshapeErr   = "Failed to reshape the incr *Tensor into %v. Size was: %d"
	retValReshapeErr = "Failed to reshape the retVal *Tensor into %v. Size was: %d"
)

func shapeMismatchError(expected, got types.Shape) error {
	return types.NewError(types.ShapeMismatch, "Shapes %v and %v are not aligned", expected, got)
}

func notyetimplemented(format string, attrs ...interface{}) error {
	return types.NewError(types.NotYetImplemented, format, attrs...)
}

type noopError struct{}

func (e noopError) NoOp() bool    { return true }
func (e noopError) Error() string { return "NoOp" }

type NoOpError interface {
	NoOp() bool
}
