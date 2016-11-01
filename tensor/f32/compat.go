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

func fromFloat64s(data []float64, r []float32) (retVal []float32) {
	if r == nil {
		retVal = make([]float32, len(data))
	} else {
		if len(r) != len(data) {
			panic("Differing lengths!")
		}

		retVal = r
	}

	for i, v := range data {
		switch {
		case math.IsNaN(v):
			retVal[i] = math32.NaN()
		case math.IsInf(v, 1):
			retVal[i] = math32.Inf(1)
		case math.IsInf(v, -1):
			retVal[i] = math32.Inf(-1)
		default:
			retVal[i] = float32(v)
		}
	}
	return
}

// FromMat64 converts a *"gonum/matrix/mat64".Dense into a *tensorf32.Tensor
func FromMat64(m *mat64.Dense) *Tensor {
	r, c := m.Dims()

	backing := make([]float32, len(m.RawMatrix().Data))
	backing = fromFloat64s(m.RawMatrix().Data, backing)

	return NewTensor(WithBacking(backing), WithShape(r, c))
}

// ToMat64 converts a *Tensor to a *"gonum/matrix/mat64".Dense. All the values are converted into float64s.
// This function will only convert matrices. Anything *Tensor with dimensions larger than 2 will cause an error.
//
// Does not work on IsMaterializable() *Tensors yet
func ToMat64(t *Tensor) (retVal *mat64.Dense, err error) {
	// checks:
	if !t.IsMatrix() {
		// error
		err = types.NewError(types.IOError, "Cannot convert *Tensor to *mat64.Dense. Expected number of dimensions: <=2, T has got %d dimensions (Shape: %v)", t.Opdims(), t.Shape())
		return
	}

	// fix dims
	r := t.Shape()[0]
	c := t.Shape()[1]

	data := toFloat64s(t.data)
	retVal = mat64.NewDense(r, c, data)
	return
}
