// +build cuda

package gorgonia

// for non-cuda builds, look at noextern.go

import (
	"log"

	"github.com/chewxy/cu"
)

const CUDA = true

const (
	// Any address of a variable residing in global memory or returned by one of the
	// memory allocation routines from the driver or runtime API is always aligned to at
	// least 256 bytes.
	//
	memalign    = 32
	scalarAlign = 8
)

var cudaStdLib map[string]string

//go:generate cudagen

// CUDAMachine is a representation of CUDA capable VMs.
type CUDAMachine interface {
	External
	Arena
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

	freeMem  []int64 // free memory available in this context
	totalMem []int64 // total memory available in this context

	a             []*bfc               // arena
	b             batchedBLAS          // blas
	c             []*cu.BatchedContext // context
	d             []cu.Device          // device
	hasWork       []bool
	workAvailable chan bool
	syncChan      chan struct{}

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

// WorkAvailable returns a channel of empty struct, which is used to signal to the VM when there is work available. The VM will then call the DoWork method
func (m *ExternMetadata) WorkAvailable() <-chan bool { return m.workAvailable }

func (m *ExternMetadata) Sync() chan struct{} { return m.syncChan }

// DoWork flushes any batched cgo calls. In this build it flushes any batched CUDA calls and any batched CBLAS calls.
func (m *ExternMetadata) DoWork() error {
	logf("DOWORK")
	// for i, hw := range m.hasWork {
	// if hw {
	// m.c[i].DoWork()
	// if err := m.c[i].Errors(); err != nil {
	// 	return err
	// }
	// m.hasWork[i] = false
	// }
	// }
	for _, c := range m.c {
		c.DoWork()
		if err := c.Errors(); err != nil {
			return err
		}
	}

	if m.blasHasWork {
		m.b.DoWork()
		m.blasHasWork = false
	}
	return nil
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

// Get gets a previously allocated memory slab of the provided size. If no memories of that size exist,
// it returns a NoOpError. The caller is then responsible for allocating the memory themselves.
func (m *ExternMetadata) Get(dev Device, size int64) (Memory, error) {
	d := int(dev)
	if d >= len(m.a) {
		return nil, noopError{} // this should not be a noopError
	}

	ptr, err := m.a[d].alloc(size)
	return cu.DevicePtr(ptr), err

	// if pool, ok := m.arena[d][size]; ok {
	// 	return pool.get()
	// }
	// return nil, noopError{}
}

// Put puts a previously allocated memory slab of the provided size back into the pool
func (m *ExternMetadata) Put(dev Device, mem Memory, size int64) {
	d := int(dev)
	if d >= len(m.a) {
		return // wat??
	}

	addr := uintptr(mem.Uintptr())
	m.a[d].free(addr)

	// pool, ok := m.arena[d][size]
	// if !ok {
	// 	pool = newMemoryQueue(size)
	// 	m.arena[d][size] = pool
	// }
	// pool.add(mem)
}

// Cleanup cleans up the ancillary allocations made during the calling of batched CUDA functions.
func (m *ExternMetadata) Cleanup() {
	for _, c := range m.c {
		c.Cleanup()
	}

	for _, a := range m.a {
		if a.start != 0 {
			cu.MemFree(cu.DevicePtr(a.start))
		}
	}
}

// Signal sends a signal down the workavailable channel, telling the VM to call the DoWork method. Signal is a synchronous method
func (m *ExternMetadata) Signal() { m.workAvailable <- true }

// Reset frees all the memories, and coalesces the allocator
func (m *ExternMetadata) Reset() {
	for _, a := range m.a {
		used := make([]uintptr, 0, len(a.used))
		for k := range a.used {
			used = append(used, k)
		}

		for _, ptr := range used {
			a.free(ptr + a.start)
		}

		a.coalesce()
	}
}

func (m *ExternMetadata) init(sizes []int64) {
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

	m.workAvailable = make(chan bool)
	m.syncChan = make(chan struct{})
	m.a = make([]*bfc, devices)
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

	m.freeMem = make([]int64, devices)
	m.totalMem = make([]int64, devices)

	for i := range m.c {
		dev, err := cu.GetDevice(i)
		if err != nil {
			cudaLogf("Failed to get device %d: %v", i, err)
			m.initFail()
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
				cudaLogf("Out of memory. ???! Free: %v, total %v", free, total)
				m.initFail()
				return
			}
			cudaLogf("Failed to make context for device %d. Error: %v", i, err)
			m.initFail()
			return
		}

		var attrs []int
		if attrs, err = dev.Attributes(cu.WarpSize, cu.MaxThreadsPerBlock, cu.MaxGridDimX, cu.MaxGridDimY, cu.MaxGridDimZ, cu.MaxBlockDimX, cu.MaxBlockDimY, cu.MaxBlockDimZ); err != nil {
			cudaLogf("Failed to get attributes for device %d. Error: %v", i, err)
			m.initFail()
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

		free, total, err := cu.MemInfo()
		if err != nil {
			cudaLogf("Failed to get free and total mem for device %d", i)
			m.initFail()
			return
		}
		m.freeMem[i] = free
		m.totalMem[i] = total
		m.a[i] = newBFC(memalign)

		var allocsize int64 = 2*sizes[i] + minAllocSize
		if allocsize > free {
			allocsize = free
		}
		ptr, err := cu.MemAllocManaged(allocsize, cu.AttachGlobal)
		m.a[i].reserve(uintptr(ptr), allocsize)

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

func (m *ExternMetadata) initFail() {
	cudaLogf("Cleanup")
	m.c = nil
	m.m = nil
	m.f = nil

	if m.workAvailable != nil {
		close(m.workAvailable)
	}
	m.workAvailable = nil
}

// collectWork is a muxer for all the channels for the different devices
func (m *ExternMetadata) collectWork(devID int, workAvailable <-chan struct{}) {
	for range workAvailable {
		m.hasWork[devID] = true
		m.workAvailable <- false
	}
}

// collectBLASWork is a muxer for CBLAS/CuBLAS (if any) and the devices
func (m *ExternMetadata) collectBLASWork() {
	if m.b != nil {
		for range m.b.WorkAvailable() {
			m.blasHasWork = true
			m.workAvailable <- false
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
