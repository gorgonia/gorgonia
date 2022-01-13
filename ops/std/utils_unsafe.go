// +build !nounsafe

package stdops

import (
	"gorgonia.org/dtype"
	_ "unsafe"
)

//go:linkname one values.nativeOne
func one(dt dtype.Dtype) interface{}
