package allocator

import "fmt"

// OOM represents an Out of tensor.Memory error. It is typically used for CUDA related machine work
type OOM struct {
	res       int64
	allocated int64
}

func (e OOM) Reserved() int64  { return e.res }
func (e OOM) Allocated() int64 { return e.allocated }
func (e OOM) Error() string    { return fmt.Sprintf("allocated/reserved: %v/%v", e.allocated, e.res) }
