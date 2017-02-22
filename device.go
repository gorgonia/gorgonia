// +build !cuda

package gorgonia

// Device represents the device where the code will be executed on. In this build, all code will run on the CPU
type Device int

const (
	CPU Device = -1 // CPU the only device the graph will be executed on
)

// String implements fmt.Stringer and runtime.Stringer
func (d Device) String() string { return "CPU" }

// IsGPU will always return false in this build
func (d Device) IsGPU() bool { return false }
