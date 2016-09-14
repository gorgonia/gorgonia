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
	// fix dims
	var r, c int
	switch t.Dims() {
	case 2:
		r = t.Shape()[0]
		c = t.Shape()[1]
	case 1:
		if t.Shape().IsColVec() {
			r = t.Shape()[0]
			c = 1
		} else {
			r = 1
			c = t.Shape()[1]
		}
	default:
		err = types.NewError(types.IOError, "Cannot convert *Tensor to *mat64.Dense. Expected number of dimensions: <=2, T has got %d dimensions", t.Dims())
		return
	}

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
