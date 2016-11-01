package tensorf32

import (
	"math"

	"github.com/chewxy/gorgonia/tensor/types"
	"github.com/chewxy/math32"
	"github.com/gonum/matrix/mat64"
)

func toFloat64s(data []float32) (retVal []float64) {
	retVal = make([]float64, len(data))
	for i, v := range data {
		switch {
		case math32.IsNaN(v):
			retVal[i] = math.NaN()
		case math32.IsInf(v, 1):
			retVal[i] = math.Inf(1)
		case math32.IsInf(v, -1):
			retVal[i] = math.Inf(-1)
		default:
			retVal[i] = float64(v)
		}
	}
	return
}

func FromMat64(m *mat64.Dense) *Tensor {
	r, c := m.Dims()

	backing := make([]float32, len(m.RawMatrix().Data))
	for i, v := range m.RawMatrix().Data {
		switch {
		case math.IsNaN(v):
			backing[i] = math32.NaN()
		case math.IsInf(v, 1):
			backing[i] = math32.Inf(1)
		case math.IsInf(v, -1):
			backing[i] = math32.Inf(-1)
		default:
			backing[i] = float32(v)
		}
	}

	return NewTensor(WithBacking(backing), WithShape(r, c))
}

func ToMat64(t *Tensor) (retVal *mat64.Dense, err error) {
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

	data := toFloat64s(t.data)
	retVal = mat64.NewDense(r, c, data)
	return
}
