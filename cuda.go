// +build cuda

package gorgonia

import "github.com/chewxy/cu"

// CUDAMachine is a representation of CUDA capable VMs.
type CUDAMachine interface {
	External
	Contexts() []cu.Context
	Modules() map[string][]cu.Module
}
