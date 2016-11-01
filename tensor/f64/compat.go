package tensorf64

import (
	"github.com/chewxy/gorgonia/tensor/types"
	"github.com/gonum/matrix/mat64"
)

// FromMat64 converts a *"gonum/matrix/mat64".Dense into a *tensorf64.Tensor.
// toCopy indicates if the values should be copied, otherwise it will share the same backing as the *mat64.Dense
func FromMat64(m *mat64.Dense, toCopy bool) *Tensor {
	r, c := m.Dims()

	var backing []float64
	if toCopy {
		backing = make([]float64, len(m.RawMatrix().Data))
		copy(backing, m.RawMatrix().Data)
	} else {
		backing = m.RawMatrix().Data
	}

	return NewTensor(WithBacking(backing), WithShape(r, c))
}

// ToMat64 converts a *Tensor to a "gonum/matrix/mat64".Dense.
// toCopy indicates if the values should be copied over, otherwise, the gonum matrix will share the same backing as the Tensor
//
// Does not work on IsMaterializable() *Tensors yet
func ToMat64(t *Tensor, toCopy bool) (retVal *mat64.Dense, err error) {
	// checks:
	if !t.IsMatrix() {
		// error
		err = types.NewError(types.IOError, "Cannot convert *Tensor to *mat64.Dense. Expected number of dimensions: <=2, T has got %d dimensions (Shape: %v)", t.Opdims(), t.Shape())
		return
	}

	// fix dims
	r := t.Shape()[0]
	c := t.Shape()[1]

	var data []float64

	switch {
	case toCopy && !t.IsMaterializable():
		data = make([]float64, len(t.data))
		copy(data, t.data)
	case !t.IsMaterializable():
		data = t.data
	default:
		it := types.NewFlatIterator(t.AP)
		var next int
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			if _, noop := err.(NoOpError); err != nil && !noop {
				return
			}
			data = append(data, float64(t.data[next]))
		}
		err = nil
	}

	retVal = mat64.NewDense(r, c, data)
	return
}
