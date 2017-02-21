// +build cuda

package gorgonia

import "github.com/chewxy/cu"

type Device cu.Device

const CPU = Device(cu.CPU)

func (d Device) String() string { return cu.Device(d).String() }
