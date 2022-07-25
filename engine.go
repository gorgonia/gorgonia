package gorgonia

import (
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

// StandardEngine is the default CPU engine for gorgonia
type StandardEngine struct {
	tensor.StdEng
}

// Transpose tensor a according to expStrides
func (e StandardEngine) Transpose(a tensor.Tensor, expStrides []int) error {
	if !a.IsNativelyAccessible() {
		return errors.Errorf("Cannot Transpose() on non-natively accessible tensor")
	}

	size := a.DataSize()
	it := a.Iterator()
	var i int
	switch a.Dtype() {
	case tensor.Float64:
		tmp := make([]float64, size)
		data := a.Data().([]float64)
		data = data[:size]
		nexts := make([]int, 0, size)
		for next, err := it.Next(); err == nil; next, err = it.Next() {
			//tmp[i] = data[next]
			//i++
			nexts = append(nexts, next)
		}
		maxNexts := nexts[len(nexts)-1] + 1 // this is possible because first and last elems don't change in a transpose

		data = data[:maxNexts] // bounds check
		tmp = tmp[:maxNexts]   // bounds check

		// you might feel like this loop can be parallelized.
		// you would be wrong.
		for i := 0; i < len(nexts); i++ {
			next := nexts[i]
			tmp[i] = data[next]
		}

		copy(data, tmp)
	case tensor.Float32:
		tmp := make([]float32, size)
		data := a.Data().([]float32)
		for next, err := it.Next(); err == nil; next, err = it.Next() {
			tmp[i] = data[next]
			i++
		}
		copy(data, tmp)
	default:
		return e.StdEng.Transpose(a, expStrides)
	}
	return nil

}
