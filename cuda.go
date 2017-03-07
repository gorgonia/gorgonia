// +build cuda

package gorgonia

// for non-cuda builds, look at noextern.go

import (
	"log"

	"github.com/chewxy/cu"
)

const CUDA = true

var cudaStdLib map[string]string

//go:generate cudagen

// CUDAMachine is a representation of CUDA capable VMs.
type CUDAMachine interface {
	External
	Contexts() []*cu.BatchedContext
	Modules() map[string][]cu.Module
	Functions() map[string][]cu.Function

	ElemGridSize(n, dev int) (gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ int)
}

// ExternMetadata holds any metadata for CUDA related stuff.
// The slices in there are indexed by deviceID
type ExternMetadata struct {
	warp []int // WarpSize
	mtpb []int // MaxThreadsPerBlock
	mgdx []int // MaxGridDimX
	mgdy []int // MaxGridDimY
	mgdz []int // MaxGridDimZ
	mbdx []int // MaxBlockDimX
	mbdy []int // MaxBlockDimY
	mbdz []int // MaxBlockDimZ

	b             batchedBLAS
	c             []*cu.BatchedContext
	hasWork       []bool
	workAvailable chan struct{}

	m map[string][]cu.Module
	f map[string][]cu.Function

	blasHasWork bool
	initialzed  bool
}

// elemGridSize calculates the gridsize for elementwise operations
func (md *ExternMetadata) ElemGridSize(n, dev int) (gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ int) {
	if dev > len(md.warp) {
		// error
	}

	maxThreads := md.mtpb[dev]
	maxGridX := md.mgdx[dev]
	maxGridY := md.mgdy[dev]
	maxGridZ := md.mgdz[dev]

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
func (md *ExternMetadata) blockThread(n, dev int) (blocks, threads int) {
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

func (m *ExternMetadata) WorkAvailable() <-chan struct{} { return m.workAvailable }

func (m *ExternMetadata) DoWork() {
	cudaLogf("DoWork() called")
	for i, hw := range m.hasWork {
		cudaLogf("Checking if %d has work %v", i, hw)
		if hw {
			m.c[i].Synchronize()
			m.c[i].DoWork()
		}
		m.hasWork[i] = false
	}

	if m.blasHasWork {
		m.b.DoWork()
		m.blasHasWork = false
	}
}
func (m *ExternMetadata) DoAllWork() {
	cudaLogf("DoAllWork")
	for _, c := range m.c {
		c.DoWork()
	}
	m.b.DoWork()
}

// HasFunc returns true if the execution is external (cgo/cuda/openCL) AND the external device contains the function with the given name
//
// Note that BLAS names will always return false, even if using a BLAS that requires cgo calls (like Intel MKL)
func (m *ExternMetadata) HasFunc(name string) bool {
	_, ok := m.f[name]
	return ok
}

// Contexts return a slice of contexts that is being used by this CUDAMachine
func (m *ExternMetadata) Contexts() []*cu.BatchedContext { return m.c }

// Modules returns a list of modules loaded (and referable by name) in this CUDAMachine
func (m *ExternMetadata) Modules() map[string][]cu.Module { return m.m }

// Functions returns a list of functions loaded (and refereable by name) in this CUDAMachine
func (m *ExternMetadata) Functions() map[string][]cu.Function { return m.f }

func (m *ExternMetadata) init() {
	if m.initialzed {
		return
	}

	devices, err := cu.NumDevices()
	if err != nil {
		cudaLogf("Failed to get number of devices: %v", err)
		return
	}

	if devices == 0 {
		cudaLogf("No devices found")
		return
	}

	m.c = make([]*cu.BatchedContext, devices)
	m.hasWork = make([]bool, devices)
	m.warp = make([]int, devices)
	m.mtpb = make([]int, devices)
	m.mgdx = make([]int, devices)
	m.mgdy = make([]int, devices)
	m.mgdz = make([]int, devices)
	m.mbdx = make([]int, devices)
	m.mbdy = make([]int, devices)
	m.mbdz = make([]int, devices)
	for i := range m.c {
		dev, err := cu.GetDevice(i)
		if err != nil {
			cudaLogf("Failed to get device %d: %v", i, err)
			m.cleanup()
			return
		}
		// ctx, err := dev.MakeContext(cu.SchedAuto)
		ctx, err := dev.MakeContext(cu.SchedBlockingSync) // for debugging
		if err != nil {
			if err == cu.OutOfMemory {
				var free, total int64
				if free, total, err = cu.MemInfo(); err != nil {
					cudaLogf("Error while getting mem info: %v", err)
				}
				cudaLogf("Out of memory. Free: %v, total %v", free, total)
				m.cleanup()
				return
			}
			cudaLogf("Failed to make context for device %d. Error: %v", i, err)
			m.cleanup()
			return
		}

		var attrs []int
		if attrs, err = dev.Attributes(cu.WarpSize, cu.MaxThreadsPerBlock, cu.MaxGridDimX, cu.MaxGridDimY, cu.MaxGridDimZ, cu.MaxBlockDimX, cu.MaxBlockDimY, cu.MaxBlockDimZ); err != nil {
			cudaLogf("Failed to get attributes for device %d. Error: %v", i, err)
			m.cleanup()
			return
		}

		m.warp[i] = attrs[0]
		m.mtpb[i] = attrs[1]
		m.mgdx[i] = attrs[2]
		m.mgdy[i] = attrs[3]
		m.mgdz[i] = attrs[4]
		m.mbdx[i] = attrs[5]
		m.mbdy[i] = attrs[6]
		m.mbdz[i] = attrs[7]

		m.c[i] = cu.NewBatchedContext(ctx, dev)
		go m.collectWork(i, m.c[i].WorkAvailable())
	}
	if len(m.c) > 0 {
		m.c[0].SetCurrent()
	}
	m.m = make(map[string][]cu.Module)
	m.f = make(map[string][]cu.Function)
	go m.collectBLASWork()

	m.initialzed = true
	cudaLogf("CUDA initialized. Contexts: %v", m.c)
}

func (m *ExternMetadata) cleanup() {
	cudaLogf("Cleanup")
	m.c = nil
	m.m = nil
	m.f = nil
}

func (m *ExternMetadata) collectWork(devID int, workAvailable <-chan struct{}) {
	for range workAvailable {
		cudaLogf("Device %d has work ", devID)
		m.hasWork[devID] = true
		m.workAvailable <- struct{}{}
	}
}

func (m *ExternMetadata) collectBLASWork() {
	if m.b != nil {
		for range m.b.WorkAvailable() {
			m.blasHasWork = true
			m.workAvailable <- struct{}{}
		}
	}
}

func init() {
	log.Println("Using CUDA build")
}

// it's just a generic ceiling function. Added here to avoid mixing with any potential ceilInt operation
func calcBlocks(n, maxThreads int) int {
	return (n + maxThreads - 1) / maxThreads
}

// AddToStdLib allows for custom ops to be included into the "stdlib" of CUDA functions, so that when the VMs are created, they're loaded automatically
// without having to specify extra loading.
func AddToStdLib(name, data string) {
	cudaStdLib[name] = data
}
