// +build !cuda

package gorgonia

import "gorgonia.org/tensor"

// CUDA indicates if this build is using CUDA
const CUDA = false

var _ tensor.Engine = ExternMetadata{}

// ExternMetadata is used to hold metadata about external execution devices.
// In this build, it's an empty struct because the default build doesn't use external devices to execute the graph on
type ExternMetadata struct {
	tensor.Engine
	b             batchedBLAS
	workAvailable chan bool
	syncChan      chan struct{}
}

func (m *ExternMetadata) init() error {
	m.syncChan = make(chan struct{})
	if m.b != nil {
		m.workAvailable = make(chan bool)
		go m.collectBLASWork()
	}
	return nil
}

// initFail is a no-op
func (m *ExternMetadata) initFail() {}

// HasFunc will always return false in this build
func (m ExternMetadata) HasFunc(name string) bool { return false }

// WorkAvailable returns a channel of empty struct, which is used to signal to the VM when there is work available. The VM will then call the DoWork method.
func (m *ExternMetadata) WorkAvailable() <-chan bool {
	if m.b != nil {
		return m.workAvailable
	}

	return nil
}

// Sync returns the sync channel
func (m *ExternMetadata) Sync() chan struct{} { return m.syncChan }

// DoWork flushes any batched cgo calls. In this build it only flushes the batched BLAS calls.
func (m *ExternMetadata) DoWork() error {
	if m.b != nil {
		m.b.DoWork()
	}
	return nil
}

// Get allocates a memory of the size. In this build it returns a NoOpError.
func (m *ExternMetadata) Get(dev Device, size int64) (tensor.Memory, error) { return nil, noopError{} }

// GetFromValue allocates a memory of the size of v. In this build it returns a NoOpError, and v itself
func (m *ExternMetadata) GetFromValue(dev Device, v Value) (tensor.Memory, error) {
	return v, noopError{}
}

// Put puts a previously allocated memory slab of the provided size back into the pool. Currently this is a No-op in this build.
func (m *ExternMetadata) Put(dev Device, mem tensor.Memory, size int64) {}

// PutValue puts a previously allocated value into the pool. In this build,  it is a noop.
func (m *ExternMetadata) PutValue(dev Device, v Value) {}

// Transfer transfers a value from device to device. In this build, it's a noop, returning the input value, and a nil error
func (m *ExternMetadata) Transfer(toDev, fromDev Device, v Value, synchronous bool) (retVal Value, err error) {
	return v, nil
}

// Reset is a noop function for compatibility with the Cuda build
func (m *ExternMetadata) Reset() {}

// Cleanup cleans up the ancillary allocations made during the calling of batched external device function.
//
// The reason for this method is due to the fact that there is currently no way to free memory while the context is still running without causing
// some weirdness to the CUDA calls.
//
// This is a No-op in this build
func (m *ExternMetadata) Cleanup() {}

// Signal sends a signal down the workavailable channel, telling the VM to call the DoWork method. Signal is a synchronous method
func (m *ExternMetadata) Signal() {
	m.signal()
	if m.workAvailable != nil {
		<-m.syncChan
	}
}

// collectBLASWork is a muxer for CBLAS/CuBLAS (if any) and the devices
func (m *ExternMetadata) collectBLASWork() {
	if m.b != nil {
		for range m.b.WorkAvailable() {
			m.workAvailable <- false
		}
	}
}

func (m *ExternMetadata) signal() {
	if m.workAvailable != nil {
		m.workAvailable <- true
	}
}

func (m *ExternMetadata) setEngine(e tensor.Engine) { m.Engine = e }

// ValueOnDevice gets the value of the node as a Value but on the desired device. In this build the device is always CPU, so it's equivalent to calling .Value()
func (n *Node) ValueOnDevice(dev Device, extern External) (retVal Value, allocOnExtern bool, err error) {
	return n.Value(), false, nil
}

// GradOnDevice gets the gradient value of the node as a Value but on the desired device. In this build the device is always CPU, so it's equivalent to calling .Grad()
func (n *Node) GradOnDevice(dev Device, extern External) (retVal Value, allocOnExtern bool, err error) {
	retVal, err = n.Grad()
	return retVal, false, err
}
