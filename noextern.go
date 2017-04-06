// +build !cuda

package gorgonia

// CUDA indicates if this build is using CUDA
const CUDA = false

// ExternMetadata is used to hold metadata about external execution devices.
// In this build, it's an empty struct because the default build doesn't use external devices to execute the graph on
type ExternMetadata struct {
	b             batchedBLAS
	workAvailable chan bool
	syncChan      chan struct{}
}

func (m *ExternMetadata) init() {
	m.syncChan = make(chan struct{})
	if m.b != nil {
		m.workAvailable = make(chan bool)
		go m.collectBLASWork()
	}
}

// HasFunc will always return false in this build
func (m ExternMetadata) HasFunc(name string) bool { return false }

// WorkAvailable returns a channel of empty struct, which is used to signal to the VM when there is work available. The VM will then call the DoWork method.
func (m *ExternMetadata) WorkAvailable() <-chan bool {
	if m.b != nil {
		return m.workAvailable
	}

	return nil
}
func (m *ExternMetadata) Sync() chan struct{} { return m.syncChan }

// DoWork flushes any batched cgo calls. In this build it only flushes the batched BLAS calls.
func (m *ExternMetadata) DoWork() error {
	if m.b != nil {
		m.b.DoWork()
	}
	return nil
}

// Get gets a previously allocated memory slab of the provided size. If no memories of that size exist,
// it returns a NoOpError. The caller is then responsible for allocating the memory themselves.
func (m *ExternMetadata) Get(dev Device, size int64) (Memory, error) { return nil, noopError{} }

// Put puts a previously allocated memory slab of the provided size back into the pool. Currently this is a No-op in this build.
func (m *ExternMetadata) Put(dev Device, mem Memory, size int64) {}

func (m *ExternMetadata) Reset() {}

// Cleanup cleans up the ancillary allocations made during the calling of batched external device function.
//
// The reason for this method is due to the fact that there is currently no way to free memory while the context is still running without causing
// some weirdness to the CUDA calls.
//
// This is a No-op in this build
func (m *ExternMetadata) Cleanup() {}

// Signal sends a signal down the workavailable channel, telling the VM to call the DoWork method. Signal is a synchronous method
func (m *ExternMetadata) Signal() { m.signal(); <-m.syncChan }

// collectBLASWork is a muxer for CBLAS/CuBLAS (if any) and the devices
func (m *ExternMetadata) collectBLASWork() {
	if m.b != nil {
		for range m.b.WorkAvailable() {
			m.workAvailable <- false
		}
	}
}

func (m *ExternMetadata) signal() { m.workAvailable <- true }
