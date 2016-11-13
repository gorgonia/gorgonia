package gorgonia

import (
	"fmt"
	"math"

	tf32 "github.com/chewxy/gorgonia/tensor/f32"
	tf64 "github.com/chewxy/gorgonia/tensor/f64"
	ti "github.com/chewxy/gorgonia/tensor/i"
	"github.com/chewxy/gorgonia/tensor/types"
	"github.com/chewxy/math32"
	"github.com/gonum/graph"
	"github.com/pkg/errors"
)

func graphNodeToNode(in []graph.Node) (out Nodes) {
	out = make(Nodes, len(in))
	for i, n := range in {
		out[i] = n.(*Node) // will panic if not. which is a good thng
	}
	return
}

func nodeToGraphNode(in []*Node) (out []graph.Node) {
	out = make([]graph.Node, len(in))
	for i, n := range in {
		out[i] = n
	}
	return
}

func dtypeToDtype(t types.Dtype) Dtype {
	if t >= types.MAXDTYPE || Dtype(t) >= Ptr {
		panic("Unsupported Dtype")
	}
	return Dtype(t)
}

func dtypeToTensorDtype(t Dtype) types.Dtype {
	if t >= Ptr || types.Dtype(t) >= types.MAXDTYPE {
		panic("Unsupported Dtype")
	}
	return types.Dtype(t)
}

func tensorInfo(t types.Tensor) (dt Dtype, dim int) {
	tdt := t.Dtype()
	dt = dtypeToDtype(tdt)
	dim = t.Dims()
	return
}

func cloneNodes(node Nodes, replacements map[*Node]*Node) Nodes {
	return nil
}

func anyToValue(any interface{}) (val Value, err error) {
	switch a := any.(type) {
	case float64, float32, int, int64, int32, byte, bool:
		return NewScalarValue(any), nil
	case types.Tensor:
		return Tensor{Tensor: a}, nil
	case Value:
		return a, nil
	default:
		return nil, errors.Errorf("value %v of %T not yet handled", any, any)
	}
	panic("Unreachable")
}

// valuesToInts will FORCIBLY cast floats to ints.
func valuesToInts(values []Value) (retVal []int, err error) {
	retVal = make([]int, len(values))
	for i, v := range values {
		sv, ok := v.(Scalar)
		if !ok {
			return nil, errors.Errorf("Expected values to be all Scalar Value. Got %v of %T instead", v, v)
		}

		var intV int
		switch vt := sv.v.(type) {
		case float64:
			intV = int(vt)
		case float32:
			intV = int(vt)
		case int:
			intV = vt
		default:
			return nil, errors.Errorf("Expected ScalarValue to have Int type. Got %v of %v(%T) instead", sv.v, sv.t, sv.v)

		}

		retVal[i] = intV
	}
	return
}

func intRange(start, end int) []int {
	size := end - start
	incr := true
	if start > end {
		incr = false
		size = start - end
	}

	if size < 0 {
		panic("Cannot create an int range that is somehow negative in size")
	}

	retVal := make([]int, size)

	for i, v := 0, start; i < size; i++ {
		retVal[i] = v
		if incr {
			v++
		} else {
			v--
		}
	}
	return retVal
}

func ones(dt Dtype, sizes ...int) (retVal Value) {
	switch dt {
	case Float64:
		if len(sizes) == 0 {
			retVal = NewScalarValue(float64(1.0))
		} else {
			t := tf64.Ones(sizes...)
			retVal = FromTensor(t)
		}
	case Float32:
		if len(sizes) == 0 {
			retVal = NewScalarValue(float64(1.0))
		} else {
			t := tf32.Ones(sizes...)
			retVal = FromTensor(t)
		}
	case Int:
		if len(sizes) == 0 {
			retVal = NewScalarValue(float64(1.0))
		} else {
			t := ti.Ones(sizes...)
			retVal = FromTensor(t)
		}

	default:
		panic(fmt.Sprintf("Dtype of %v not yet implemented for ones()"))
	}
	return
}

func hasInf(v Value) bool {
	switch vt := v.(type) {
	case Tensor:
		switch vt.Dtype() {
		case Float64:
			T := vt.Tensor.(*tf64.Tensor)
			data := T.Data().([]float64)
			for _, datum := range data {
				if math.IsInf(datum, 0) {
					return true
				}
			}
			return false
		case Float32:
			T := vt.Tensor.(*tf32.Tensor)
			data := T.Data().([]float32)
			for _, datum := range data {
				if math32.IsInf(datum, 0) {
					return true
				}
			}
			return false
		default:
			err := nyi("hasInf", vt.Dtype())
			panic(err)
		}
	case Scalar:
		switch f := vt.v.(type) {
		case float32:
			return math32.IsInf(f, 0)
		case float64:
			return math.IsInf(f, 0)
		default:
			return false
		}
	case *dualValue:
		return hasInf(vt.Value) || hasInf(vt.d)
	default:
		err := nyi("hasInf", vt)
		panic(err)
	}

	panic("Unreachable")
}

func hasNaN(v Value) bool {
	switch vt := v.(type) {
	case Tensor:
		switch vt.Dtype() {
		case Float64:
			T := vt.Tensor.(*tf64.Tensor)
			data := T.Data().([]float64)
			for _, datum := range data {
				if math.IsNaN(datum) {
					return true
				}
			}
			return false
		case Float32:
			T := vt.Tensor.(*tf64.Tensor)
			data := T.Data().([]float32)
			for _, datum := range data {
				if math32.IsNaN(datum) {
					return true
				}
			}
			return false
		default:
			err := nyi("hasNaN", vt.Dtype())
			panic(err)
		}
	case Scalar:
		switch f := vt.v.(type) {
		case float32:
			return math32.IsNaN(f)
		case float64:
			return math.IsNaN(f)
		default:
			return false
		}
	case *dualValue:
		return hasNaN(vt.Value) || hasNaN(vt.d)
	default:
		err := nyi("hasNaN", vt)
		panic(err)
	}
	panic("Unreachable")
}

func setZero(val Value) (retVal Value) {
	switch v := val.(type) {
	case Zeroer:
		v.Zero()
		return v
	case Scalar:
		cloned, err := v.clone()
		if err != nil {
			panic(err)
		}

		s2 := cloned.(Scalar)

		switch v.t {
		case Float64:
			s2.v = 0.0
		case Float32:
			s2.v = float32(0.0)
		case Int:
			s2.v = 0
		case Int64:
			s2.v = int64(0)
		case Int32:
			s2.v = int32(0)
		case Byte:
			s2.v = byte(0)
		case Bool:
			s2.v = false
		}
		return s2
	default:
		panic(fmt.Sprintf("setZero not implemented yet for %T", v))
	}
	panic("unreachable")
}
