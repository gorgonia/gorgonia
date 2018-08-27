package cuda

import "fmt"

// oomError represents an Out of tensor.Memory error. It is typically used for CUDA related machine work
type oomError struct {
	res       int64
	allocated int64
}

func (e oomError) Reserved() int64  { return e.res }
func (e oomError) Allocated() int64 { return e.allocated }
func (e oomError) Error() string    { return fmt.Sprintf("allocated/reserved: %v/%v", e.allocated, e.res) }

const (
	typeMismatch  = "TypeMismatch: a %T and b %T"
	shapeMismatch = "Shape mismatch. Expected %v. Got %v"
)
