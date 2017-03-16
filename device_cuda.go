// +build cuda

package gorgonia

import "github.com/chewxy/cu"

// Device represents the device where the code will be executed on. It can either be a GPU or CPU
type Device cu.Device

// CPU is the default the graph will be executed on.
const CPU = Device(cu.CPU)

// String implements fmt.Stringer and runtime.Stringer
func (d Device) String() string { return cu.Device(d).String() }
