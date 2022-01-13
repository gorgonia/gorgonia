// +build !nounsafe

package stdops

import (
	"gorgonia.org/dtype"
	_ "gorgonia.org/gorgonia/values"
	_ "unsafe"
)

//go:linkname one gorgonia.org/gorgonia/values.nativeOne
func one(dt dtype.Dtype) interface{}
