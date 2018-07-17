package cuda

import (
	"gorgonia.org/cu"
	"gorgonia.org/cu/blas"
	"gorgonia.org/cu/dnn"
)

//  this file implements all the methods required to fulfil the External interface

var _ External = &Engine{}

// HasFunc returns true if the execution is external (cgo/cuda/openCL) AND the external device contains the function with the given name
func (e *Engine) HasFunc(name string) bool { _, ok := e.f[name]; return ok }

func (e *Engine) Sync() chan struct{} { return e.syncChan }

func (e *Engine) Signal() {
	if e.workAvailable != nil {
		e.signal()
		<-e.syncChan
	}
}

func (e *Engine) signal() { e.workAvailable <- true }

func (e *Engine) Context() cu.Context { return &e.c }

func (e *Engine) CUDNNContext() *cudnn.Context { return &e.n }

func (e *Engine) BLASContext() *cublas.Standard { return &e.b }

func (e *Engine) Modules() map[string]cu.Module { return e.m }

func (e *Engine) Functions() map[string]cu.Function { return e.f }

// elemGridSize calculates the gridsize for elementwise operations. n is the number of elements
func (e *Engine) ElemGridSize(n int) (gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ int) {
	maxThreads := e.mtpb
	maxGridX := e.mgdx
	maxGridY := e.mgdy
	maxGridZ := e.mgdz

	blockDimX = 1
	blockDimY = 1
	blockDimZ = 1
	gridDimX = 1
	gridDimY = 1
	gridDimZ = 1

	blocks := calcBlocks(n, maxThreads)
	switch {
	case blocks == 1:
		blockDimX = n
	case blocks >= maxGridX*maxGridY*maxGridZ:
		// what kind of monstrosity is this??!
	case blocks >= maxGridX*maxGridY:
		gridDimX = maxGridX
		gridDimY = maxGridY
		gridDimZ = calcBlocks(blocks%(maxGridX*maxGridY), maxGridZ)
		blockDimX = maxThreads
	case blocks >= maxGridX:
		gridDimX = maxGridX
		gridDimY = calcBlocks(blocks%(maxGridX), maxGridY)
		blockDimX = maxThreads
	default:
		gridDimX = blocks
		blockDimX = maxThreads
	}

	return
}

// blockThread is an easier version of calculating <<threads, blocks>> for CUDA. Useful for debugging
func (e *Engine) blockThread(n, dev int) (blocks, threads int) {
	switch {
	case n <= 32:
		threads = 32
	case n <= 64:
		threads = 64
	case n <= 128:
		threads = 128
	case n <= 256:
		threads = 256
	case n <= 512:
		threads = 512
	default:
		threads = 1024
	}

	blocks = (n + threads - 1) / threads
	if blocks < 0 || blocks > 128 {
		blocks = 128
	}
	return
}

// it's just a generic ceiling function. Added here to avoid mixing with any potential ceilInt operation
func calcBlocks(n, maxThreads int) int { return (n + maxThreads - 1) / maxThreads }
