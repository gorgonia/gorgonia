package tensor

import (
	"github.com/gonum/matrix/mat64"
	"github.com/pkg/errors"
)

// FromMat64 converts a *"gonum/matrix/mat64".Dense into a *tensorf64.Tensor.
// toCopy indicates if the values should be copied, otherwise it will share the same backing as the *mat64.Dense
func FromMat64(m *mat64.Dense, toCopy bool) *Dense {
	r, c := m.Dims()

	var backing []float64
	if toCopy {
		backing = make([]float64, len(m.RawMatrix().Data))
		copy(backing, m.RawMatrix().Data)
	} else {
		backing = m.RawMatrix().Data
	}

	return New(WithBacking(backing), WithShape(r, c))
}

// ToMat64 converts a *Tensor to a "gonum/matrix/mat64".Dense.
// toCopy indicates if the values should be copied over, otherwise, the gonum matrix will share the same backing as the Tensor
// toCopy only works on []float64. Other types will have copy ops.
//
// Does not work on IsMaterializable() *Tensors yet
func ToMat64(t Tensor, toCopy bool) (retVal *mat64.Dense, err error) {
	switch tt := t.(type) {
	case *Dense:
		// checks:
		if !tt.IsMatrix() {
			// error
			err = errors.Errorf("Cannot convert *Dense to *mat64.Dense. Expected number of dimensions: <=2, T has got %d dimensions (Shape: %v)", t.Dims(), t.Shape())
			return
		}
		// fix dims
		r := t.Shape()[0]
		c := t.Shape()[1]

		var data []float64
		switch {
		case toCopy && !tt.IsMaterializable():
			data = make([]float64, tt.data.Len())
			switch tdata := t.Data().(type) {
			case []float64:
				copy(data, tdata)
			case []float32:
				for i, v := range tdata {
					data[i] = float64(v)
				}
			case []int:
				for i, v := range tdata {
					data[i] = float64(v)
				}
			case []int64:
				for i, v := range tdata {
					data[i] = float64(v)
				}
			case []int32:
				for i, v := range tdata {
					data[i] = float64(v)
				}
			case []byte:
				for i, v := range tdata {
					data[i] = float64(v)
				}
			case []bool:
				for i, v := range tdata {
					if v {
						data[i] = 1
					} else {
						data[i] = 0
					}
				}
			}
		case !tt.IsMaterializable():
			switch tdata := t.Data().(type) {
			case []float64:
				data = tdata
			case []float32:
				data = make([]float64, len(tdata))
				for i, v := range tdata {
					data[i] = float64(v)
				}
			case []int:
				data = make([]float64, len(tdata))
				for i, v := range tdata {
					data[i] = float64(v)
				}
			case []int64:
				data = make([]float64, len(tdata))
				for i, v := range tdata {
					data[i] = float64(v)
				}
			case []int32:
				data = make([]float64, len(tdata))
				for i, v := range tdata {
					data[i] = float64(v)
				}
			case []byte:
				data = make([]float64, len(tdata))
				for i, v := range tdata {
					data[i] = float64(v)
				}
			case []bool:
				data = make([]float64, len(tdata))
				for i, v := range tdata {
					if v {
						data[i] = 1
					} else {
						data[i] = 0
					}
				}

			}
		default:
			it := NewFlatIterator(tt.AP)
			var next int
			for next, err = it.Next(); err == nil; next, err = it.Next() {
				if _, noop := err.(NoOpError); err != nil && !noop {
					return
				}
				v := tt.data.Get(next)
				switch vt := v.(type) {
				case float64:
					data = append(data, vt)
				case float32:
					data = append(data, float64(vt))
				case int:
					data = append(data, float64(vt))
				case int64:
					data = append(data, float64(vt))
				case int32:
					data = append(data, float64(vt))
				case byte:
					data = append(data, float64(vt))
				case bool:
					if vt {
						data = append(data, 1)
					} else {
						data = append(data, 0)
					}
				}
			}
			err = nil
		}

		retVal = mat64.NewDense(r, c, data)
	default:
		err = errors.Errorf(methodNYI, "ToMat64", t)
	}
	return
}
