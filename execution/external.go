package execution

import (
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/tensor"
)

// Arena is a representation of a pool of tensor.Memory
type Arena interface {
	Get(dev Device, size int64) (tensor.Memory, error)              // Get returns a NoOpError when it cannot get a memory. Please allocate
	GetFromValue(dev Device, v values.Value) (tensor.Memory, error) // Gets a memory and copies the values into the memory and returns it.
	Put(dev Device, mem tensor.Memory, size int64)                  // puts the memory back into the arena
	PutValue(dev Device, v values.Value)                            // puts the memory back into the arena

	// Transfers memory from device to device
	Transfer(toDev, fromDev Device, v values.Value, synchronous bool) (retVal values.Value, err error)
}

// External is a representation of an external device (cuda/cgo/openCL), conceptually modelled as a machine.
type External interface {
	Arena
	Signal() // signals the machine to do work
	Sync() chan struct{}
}
