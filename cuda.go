// +build cuda

package gorgonia

import "github.com/chewxy/cu"

var cudaStdLib map[string]string

//go:generate cudagen

// CUDAMachine is a representation of CUDA capable VMs.
type CUDAMachine interface {
	External
	Contexts() []cu.Context
	Modules() map[string][]cu.Module
}
