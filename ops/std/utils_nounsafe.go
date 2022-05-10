// +build nounsafe

package stdops

import (
	"gorgonia.org/dtype"
	"gorgonia.org/shapes"
)

// keepsync: gorgonia.org/gorgonia/values.oneNative
func one(dt dtype.Dtype) interface{} {
	switch dt {
	case dtype.Float64:
		return float64(1)
	case dtype.Float32:
		return float32(1)
	case dtype.Int:
		return int(1)
	case dtype.Int32:
		return int32(1)
	case dtype.Int64:
		return int64(1)
	case dtype.Byte:
		return byte(1)
	case dtype.Bool:
		return true
	default:
		panic("Unhandled dtype")
	}
}

func ints2axes(is []int) shapes.Axes {
	retVal := make(shapes.Axes, 0, len(is))
	for _, i := range is {
		retVal = append(retVal, shapes.Axis(i))
	}
	return retVal
}

func axesToInts(a shapes.Axes) []int {
	retVal := make([]int, len(a))
	for i := range a {
		retVal[i] = int(a[i])
	}
	return retVal
}
