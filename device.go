// +build !cuda

package gorgonia

type Device int

const (
	CPU Device = -1
)

func (d Device) String() string { return "CPU" }
func (d Device) IsGPU() bool    { return false }
