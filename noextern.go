// +build !cuda

package gorgonia

// CUDA indicates if this build is using CUDA
const CUDA = false

type ExternMetadata struct{}

func (m ExternMetadata) HasFunc(name string) bool                  { return false }
func (m ExternMetadata) Function(name string) (interface{}, error) { return nil, nil }
