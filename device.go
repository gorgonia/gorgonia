// +build !cuda

package gorgonia

import "gorgonia.org/tensor"

// Device represents the device where the code will be executed on. In this build, all code will run on the CPU
type Device int

const (
	// CPU the only device the graph will be executed on
	CPU Device = 0
)

// String implements fmt.Stringer and runtime.Stringer
func (d Device) String() string { return "CPU" }

// IsGPU will always return false in this build
func (d Device) IsGPU() bool { return false }

// Alloc allocates memory on the device. This is currently a NO-OP in this build
func (d Device) Alloc(extern External, size int64) (tensor.Memory, error) { return nil, nil }

// Free frees the memory on the device. This is currently a NO-OP in this build
func (d Device) Free(extern External, mem tensor.Memory, sie uint) error { return nil }
