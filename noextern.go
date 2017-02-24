// +build !cuda

package gorgonia

// CUDA indicates if this build is using CUDA
const CUDA = false

// ExternMetadata is used to hold metadata about external execution devices.
// In this build, it's an empty struct because the default build doesn't use external devices to execute the graph on
type ExternMetadata struct{}

// HasFunc will always return false in this build
func (m ExternMetadata) HasFunc(name string) bool { return false }
