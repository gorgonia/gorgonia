package tensor

import (
	"log"
	"math"

	"github.com/chewxy/math32"
	"github.com/gonum/matrix/mat64"
	"github.com/pkg/errors"
)

func toFloat64s(data Array) (retVal []float64) {
	retVal = make([]float64, data.Len())
	switch arr := data.(type) {
	case Float32ser:
		for i, v := range arr.Float32s() {
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
	case Intser:
		for i, v := range arr.Ints() {
			retVal[i] = float64(v)
		}
	case Int64ser:
		for i, v := range arr.Int64s() {
			retVal[i] = float64(v)
		}
	case Int32ser:
		for i, v := range arr.Int32s() {
			retVal[i] = float64(v)
		}
	case Byteser:
		for i, v := range arr.Bytes() {
			retVal[i] = float64(v)
		}
	case Boolser:
		for i, v := range arr.Bools() {
			if v {
				retVal[i] = 1
			} else {
				retVal[i] = 0
			}
		}

	}
	return
}

func fromFloat64s(data []float64, of Dtype) (retVal Array) {
	switch of {
	case Float64:
		return f64s(data)
	case Float32:
		r := make(f32s, len(data))
		for i, v := range data {
			switch {
			case math.IsNaN(v):
				r[i] = math32.NaN()
			case math.IsInf(v, 1):
				r[i] = math32.Inf(1)
			case math.IsInf(v, -1):
				r[i] = math32.Inf(-1)
			default:
				r[i] = float32(v)
			}
		}
		return r
	case Int:
		r := make(ints, len(data))
		for i, v := range data {
			switch {
			case math.IsNaN(v), math.IsInf(v, 0):
				r[i] = 0
			default:
				r[i] = int(v)
			}
		}
		return r
	case Int64:
		r := make(i64s, len(data))
		for i, v := range data {
			switch {
			case math.IsNaN(v), math.IsInf(v, 0):
				r[i] = 0
			default:
				r[i] = int64(v)
			}
		}
		return r
	case Int32:
		r := make(i32s, len(data))
		for i, v := range data {
			switch {
			case math.IsNaN(v), math.IsInf(v, 0):
				r[i] = 0
			default:
				r[i] = int32(v)
			}
		}
		return r
	case Byte:
		r := make(u8s, len(data))
		for i, v := range data {
			switch {
			case math.IsNaN(v), math.IsInf(v, 0):
				r[i] = 0
			default:
				r[i] = byte(v)
			}
		}
		return r
	case Bool:
		r := make(bs, len(data))
		for i, v := range data {
			switch {
			case math.IsNaN(v), math.IsInf(v, 0):
				r[i] = false
			case v == 0:
				r[i] = false
			default:
				r[i] = true
			}
		}
		return r
	default:
		log.Printf("%T", of)
		panic("Not handled yet")
	}
	return
}

// FromMat64 converts a *"gonum/matrix/mat64".Dense into a *tensorf64.Tensor.
// toCopy indicates if the values should be copied, otherwise it will share the same backing as the *mat64.Dense
func FromMat64(m *mat64.Dense, of Dtype, toCopy bool) *Dense {
	r, c := m.Dims()

	if of == Float64 {
		var backing []float64
		if toCopy {
			backing = make([]float64, len(m.RawMatrix().Data))
			copy(backing, m.RawMatrix().Data)
		} else {
			backing = m.RawMatrix().Data
		}

		return New(WithBacking(backing), WithShape(r, c))
	}

	backing := fromFloat64s(m.RawMatrix().Data, of)
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
			if f, ok := tt.data.(Float64ser); ok {
				data = make([]float64, tt.data.Len())
				copy(data, f.Float64s())
			} else {
				data = toFloat64s(tt.data)
			}
		case !tt.IsMaterializable():
			if f, ok := tt.data.(Float64ser); ok {
				data = f.Float64s()
			} else {
				data = toFloat64s(tt.data)
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
