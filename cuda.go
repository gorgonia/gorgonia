// +build cuda

package gorgonia

// for non-cuda builds, look at noextern.go

import (
	"log"
	"sync"

	"github.com/pkg/errors"
	"gorgonia.org/cu"
	cudnn "gorgonia.org/cu/dnn"
	"gorgonia.org/gorgonia/cuda"
	"gorgonia.org/tensor"
)

// CUDA tells the package that CUDA is used
const CUDA = true

var (
	_ External    = &ExternMetadata{}
	_ CUDAMachine = &tapeMachine{}
	_ CUDAMachine = &lispMachine{}
)

const (
	// Any address of a variable residing in global memory or returned by one of the
	// memory allocation routines from the driver or runtime API is always aligned to at
	// least 256 bytes.
	//
	memalign    = 32
	scalarAlign = 8
)

var cudaStdLib map[string]string
var cudaStdFuncs map[string][]string

//go:generate cudagen

// CUDAMachine is a representation of CUDA capable VMs.
type CUDAMachine interface {
	External
	Engines() []cuda.Engine
	Contexts() []*cu.BatchedContext
	CUDNNContexts() []*cudnn.Context

	ElemGridSize(n, dev int) (gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ int)
}

// ExternMetadata holds any metadata for CUDA related stuff.
// The slices in there are indexed by deviceID
type ExternMetadata struct {
	tensor.Engine
	sync.Mutex

	// operational stuff
	u cu.Device   // device currently in use
	b batchedBLAS // UNUSED

	engines       []cuda.Engine
	workAvailable chan bool
	syncChan      chan struct{}
	initialized   bool
}

// ElemGridSize calculates the gridsize for elementwise operations
func (m *ExternMetadata) ElemGridSize(n, dev int) (gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ int) {
	if dev >= len(m.engines) {
		// error
	}
	return m.engines[dev].ElemGridSize(n)
}

// WorkAvailable returns a channel of empty struct, which is used to signal to the VM when there is work available. The VM will then call the DoWork method
func (m *ExternMetadata) WorkAvailable() <-chan bool { return m.workAvailable }

// Sync the channels
func (m *ExternMetadata) Sync() chan struct{} { return m.syncChan }

// DoWork flushes any batched cgo calls. In this build it flushes any batched CUDA calls and any batched CBLAS calls.
func (m *ExternMetadata) DoWork() error {
	for _, e := range m.engines {
		if err := e.DoWork(); err != nil {
			return err
		}
	}
	return nil
}

// Engines ...
func (m *ExternMetadata) Engines() []cuda.Engine { return m.engines }

// Contexts return a slice of contexts that is being used by this CUDAMachine
func (m *ExternMetadata) Contexts() []*cu.BatchedContext {
	retVal := make([]*cu.BatchedContext, 0, len(m.engines))
	for _, e := range m.engines {
		retVal = append(retVal, e.Context())
	}
	return retVal
}

// CUDNNContexts returns the CUDNN context
func (m *ExternMetadata) CUDNNContexts() []*cudnn.Context {
	retVal := make([]*cudnn.Context, 0, len(m.engines))
	for _, e := range m.engines {
		retVal = append(retVal, e.CUDNNContext())
	}
	return retVal
}

// Get gets a previously allocated memory slab of the provided size. If no memories of that size exist,
// it returns a NoOpError. The caller is then responsible for allocating the memory themselves.
func (m *ExternMetadata) Get(dev Device, size int64) (tensor.Memory, error) {
	d := int(dev)
	if d >= len(m.engines) {
		return nil, noopError{} // this should not be a noopError
	}
	return m.engines[dev].Get(size)
}

// GetFromValue allocates a memory on the GPU, and then copies the data over. v MUST be on CPU.
func (m *ExternMetadata) GetFromValue(dev Device, v Value) (tensor.Memory, error) {
	d := int(dev)
	if d >= len(m.engines) {
		return nil, noopError{}
	}
	memsize := calcMemSize(v.Dtype(), v.Shape())

	mem, err := m.engines[dev].Get(memsize)
	if err != nil {
		return nil, err
	}
	ptr := cu.DevicePtr(mem.Uintptr())
	ctx := m.engines[dev].Context()
	ctx.MemcpyHtoD(ptr, v.Pointer(), memsize)
	return cu.DevicePtr(ptr), nil
}

// Put puts a previously allocated memory slab of the provided size back into the pool
func (m *ExternMetadata) Put(dev Device, mem tensor.Memory, size int64) {
	d := int(dev)
	if d >= len(m.engines) {
		return // wat??
	}

	m.engines[dev].Put(mem, size)
}

// PutValue puts a previously allocated memory slab back into the pool
func (m *ExternMetadata) PutValue(dev Device, v Value) {
	d := int(dev)
	if d >= len(m.engines) {
		return
	}
	memsize := calcMemSize(v.Dtype(), v.Shape())
	m.engines[dev].Put(v, memsize)
}

// Transfer transfers data from device to device.
func (m *ExternMetadata) Transfer(toDev, fromDev Device, v Value, synchronous bool) (retVal Value, err error) {
	defer func() {
		if synchronous {
			m.Signal()
		}
	}()

	memsize := calcMemSize(v.Dtype(), v.Shape())
	switch {
	case fromDev == CPU && toDev != CPU:
		d := int(toDev)
		if d > len(m.engines) {
			return nil, errors.Errorf("No context for ToDev")
		}

		ctx := m.engines[d].Context()
		var mem tensor.Memory
		if mem, err = m.Get(toDev, memsize); err != nil {
			return
		}
		ctx.MemcpyHtoD(cu.DevicePtr(mem.Uintptr()), v.Pointer(), memsize)
		return makeValueFromMem(TypeOf(v), v.Shape(), mem)

	case fromDev != CPU && toDev == CPU:
		d := int(fromDev)
		if d > len(m.engines) {
			return nil, errors.Errorf("No context for FromDev")
		}

		ctx := m.engines[d].Context()
		if retVal, err = makeValue(TypeOf(v), v.Shape()); err != nil {
			return
		}
		ctx.MemcpyDtoH(retVal.Pointer(), cu.DevicePtr(v.Uintptr()), memsize)
		return
	case fromDev == toDev:
		return v, nil
	case fromDev != toDev && fromDev != CPU && toDev != CPU:

	}
	panic("Unreachable")
}

// Signal sends a signal down the workavailable channel, telling the VM to call the DoWork method. Signal is a synchronous method
func (m *ExternMetadata) Signal() {
	if m.workAvailable != nil {
		m.signal()
		<-m.syncChan
	}
}

// Reset frees all the memories, and coalesces the allocator
func (m *ExternMetadata) Reset() {
	for i := range m.engines {
		m.engines[i].ResetAllocator()
	}
}

func (m *ExternMetadata) init(sizes []int64) (err error) {
	m.Lock()
	initialized := m.initialized
	m.Unlock()
	if initialized {
		return nil
	}
	devices, err := cu.NumDevices()
	if err != nil {
		return errors.Wrapf(err, "Failed to get number of devices")
	}

	if devices == 0 {
		return errors.New("No Devices Found")
	}

	cudaLogf("Creating Engines")
	m.Lock()
	defer m.Unlock()
	m.engines = make([]cuda.Engine, len(sizes))
	for i := range m.engines {
		e := &m.engines[i]
		dev, err := cu.GetDevice(i)
		if err != nil {
			return errors.Wrapf(err, "Failed to get device %d", i)
		}

		if err = e.Init(dev, sizes[i]); err != nil {
			return err
		}
		ctx := e.Context()
		go m.collectWork(i, ctx.WorkAvailable())
	}

	m.initialized = true
	cudaLogf("CUDA initialized. Engines: %v", m.engines)
	return nil
}

func (m *ExternMetadata) initFail() {
	cudaLogf("Cleanup")
	m.engines = nil

	if m.workAvailable != nil {
		close(m.workAvailable)
	}
	m.workAvailable = nil
}

// cleanup cleans up the ancillary allocations made during the calling of batched CUDA functions.
func (m *ExternMetadata) cleanup() {
	for _, e := range m.engines {
		e.Close()
	}
}

// collectWork is a muxer for all the channels for the different devices
func (m *ExternMetadata) collectWork(devID int, workAvailable <-chan struct{}) {
	for range workAvailable {
		m.workAvailable <- false
	}
}

// collectBLASWork is a muxer for CBLAS/CuBLAS (if any) and the devices
func (m *ExternMetadata) collectBLASWork() {}

func (m *ExternMetadata) signal() { m.workAvailable <- true }

// it's just a generic ceiling function. Added here to avoid mixing with any potential ceilInt operation
func calcBlocks(n, maxThreads int) int {
	return (n + maxThreads - 1) / maxThreads
}

func (m *ExternMetadata) setEngine(e tensor.Engine) {}

// AddToStdLib allows for custom ops to be included into the "stdlib" of CUDA functions, so that when the VMs are created, they're loaded automatically
// without having to specify extra loading.
func AddToStdLib(name, data string) {
	cudaStdLib[name] = data
}

func init() {
	log.Println("Using CUDA build")
}

// ValueOnDevice gets the value of the node as a Value but on the desired device. If the node's valud is not on the same device
// as the desired device, a copy will be made.
func (n *Node) ValueOnDevice(toDev Device, extern External) (retVal Value, allocOnExtern bool, err error) {
	if n.dataOn == toDev {
		return n.Value(), false, nil
	}
	v := n.Value()
	fromDev := n.Device()

	var synchronous bool
	if toDev == CPU {
		synchronous = true
	}
	if toDev != fromDev && toDev != CPU {
		allocOnExtern = true
	}
	retVal, err = extern.Transfer(toDev, fromDev, v, synchronous)
	return
}

// GradOnDevice gets the gradient value of the node as a Value but on the desired device. If the node's valud is not on the same device
// as the desired device, a copy will be made.
func (n *Node) GradOnDevice(toDev Device, extern External) (retVal Value, allocOnExtern bool, err error) {
	if n.dataOn == toDev {
		retVal, err = n.Grad()
		return
	}

	var d Value
	if dv, ok := n.boundTo.(*dualValue); ok {
		d = dv.d
	} else if n.deriv != nil {
		return n.deriv.ValueOnDevice(toDev, extern)
	} else {
		return nil, false, errors.Errorf("No gradient node/value found for %v", n)
	}
	if d == nil {
		return nil, false, errors.Errorf("No gradient node/value found for %v", n)
	}

	fromDev := n.Device()

	var synchronous bool
	if toDev == CPU {
		synchronous = true
	}
	if toDev != CPU && toDev != fromDev {
		allocOnExtern = true
	}
	retVal, err = extern.Transfer(toDev, fromDev, d, synchronous)
	return
}
