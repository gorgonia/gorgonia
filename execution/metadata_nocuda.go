//go:build !cuda
// +build !cuda

package execution

import (
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/tensor"

	gerrors "gorgonia.org/gorgonia/internal/errors"
)

type ExternMetadata struct {
	tensor.Engine
}

func MakeEngine(e tensor.Engine) ExternMetadata {
	if e == nil {
		panic("Expected an engine to be passed in")
	}
	return ExternMetadata{e}
}

// Get allocates a memory of the size. In this build it returns a NoOpError.
func (m *ExternMetadata) Get(dev Device, size int64) (tensor.Memory, error) {
	return nil, gerrors.NoOp{}
}

// GetFromValue allocates a memory of the size of v. In this build it returns a NoOpError, and v itself.
func (m *ExternMetadata) GetFromValue(dev Device, v values.V) (tensor.Memory, error) {
	return v, gerrors.NoOp{}
}

// Put puts a previously allocated memory slab of the provided size back into the pool. Currently this is a No-op in this build.
func (m *ExternMetadata) Put(dev Device, mem tensor.Memory, size int64) {}

// PutValue puts a previously allocated value into the pool. In this build,  it is a noop.
func (m *ExternMetadata) PutValue(dev Device, v values.V) {}

// Transfer transfers a value from device to device. In this build, it's a noop, returning the input value, and a nil error
func (m *ExternMetadata) Transfer(toDev, fromDev Device, v values.V, synchronous bool) (retVal values.V, err error) {
	return v, nil
}
