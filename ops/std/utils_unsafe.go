// +build !nounsafe

package stdops

import (
	"unsafe"

	"gorgonia.org/dtype"
	_ "gorgonia.org/gorgonia/values"
	"gorgonia.org/shapes"
)

//go:linkname one gorgonia.org/gorgonia/values.nativeOne
func one(dt dtype.Dtype) interface{}

func ints2axes(is []int) shapes.Axes { return *(*shapes.Axes)(unsafe.Pointer(&is)) }

//go:linkname axesToInts gorgonia.org/shapes.axesToInts
func axesToInts(a shapes.Axes) []int
