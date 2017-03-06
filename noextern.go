// +build !cuda

package gorgonia

// CUDA indicates if this build is using CUDA
const CUDA = false

// ExternMetadata is used to hold metadata about external execution devices.
// In this build, it's an empty struct because the default build doesn't use external devices to execute the graph on
type ExternMetadata struct {
	b batchedBLAS
}

// HasFunc will always return false in this build
func (m ExternMetadata) HasFunc(name string) bool { return false }
func (m *ExternMetadata) WorkAvailable() <-chan struct{} {
	if m.b != nil {
		return m.b.WorkAvailable()
	}

	return nil
}
func (m *ExternMetadata) DoWork() {
	if m.b != nil {
		m.b.DoWork()
	}
}
func (m *ExternMetadata) DoAllWork() { m.DoWork() }
