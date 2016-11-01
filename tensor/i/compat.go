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

// FromMat64 converts a *"gonum/matrix/mat64".Dense into a *tensorf64.Tensor.
//
// Special cases of Inf, -Inf and NaN() are unhandled, and will be returned as the values that int(math.NaN()) or int(math.Inf(1)) or int(math.Inf(-1)) returns
func FromMat64(m *mat64.Dense) *Tensor {
	r, c := m.Dims()

	backing := make([]int, len(m.RawMatrix().Data))
	for i, v := range m.RawMatrix().Data {
		backing[i] = int(v)
	}

	return NewTensor(WithBacking(backing), WithShape(r, c))
}

// ToMat64 converts a *Tensor to a "gonum/matrix/mat64".Dense. All values are converted from int to float64
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
