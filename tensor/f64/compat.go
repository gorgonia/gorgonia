package tensorf64

import (
	"github.com/chewxy/gorgonia/tensor/types"
	"github.com/gonum/matrix/mat64"
)

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
	if toCopy {
		data = make([]float64, len(t.data))
		copy(data, t.data)
	} else {
		data = t.data
	}

	retVal = mat64.NewDense(r, c, data)
	return
}
