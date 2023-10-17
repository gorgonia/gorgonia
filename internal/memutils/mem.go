// package memutils provides memory management utilities
package memutils

import (
	"gorgonia.org/dtype"
	"gorgonia.org/tensor"
)

// MemSize returns the size of the memory required
func MemSize(dt dtype.Dtype, s tensor.Shape) int64 {
	var elemSize int64
	if s.IsScalar() {
		elemSize = 1
	} else {
		elemSize = int64(s.TotalSize())
	}
	dtSize := int64(dt.Size())
	return elemSize * dtSize
}
