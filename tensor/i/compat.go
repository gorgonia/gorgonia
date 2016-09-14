package tensori

import (
	"github.com/chewxy/gorgonia/tensor/types"
	"github.com/gonum/matrix/mat64"
)

func toFloat64s(data []int) (retVal []float64) {
	retVal = make([]float64, len(data))
	for i, v := range data {
		retVal[i] = float64(v)
	}
	return
}

func toFloat32s(data []int) (retVal []float32) {
	retVal = make([]float32, len(data))
	for i, v := range data {
		retVal[i] = float32(v)
	}
	return
}

func toBools(data []int) (retVal []bool) {
	retVal = make([]bool, len(data))
	for i, v := range data {
		if v != 0 {
			retVal[i] = true
		}
	}
	return
}

func FromMat64(m *mat64.Dense) *Tensor {
	r, c := m.Dims()

	backing := make([]int, len(m.RawMatrix().Data))
	for i, v := range m.RawMatrix().Data {
		backing[i] = int(v)
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
