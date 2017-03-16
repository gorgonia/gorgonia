package gorgonia

import (
	"fmt"
	"math"

	"github.com/chewxy/gorgonia/tensor"
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

func tensorInfo(t tensor.Tensor) (dt tensor.Dtype, dim int) {
	dt = t.Dtype()
	dim = t.Dims()
	return
}

func cloneNodes(node Nodes, replacements map[*Node]*Node) Nodes {
	return nil
}

// valuesToInts will FORCIBLY cast floats to ints.
func valuesToInts(values []Value) (retVal []int, err error) {
	retVal = make([]int, len(values))
	for i, v := range values {
		var intV int
		switch sv := v.(type) {
		case *F64:
			intV = int(float64(*sv))
		case *F32:
			intV = int(float32(*sv))
		case *I:
			intV = int(*sv)
		case *I32:
			intV = int(int32(*sv))
		case *I64:
			intV = int(int64(*sv))
		case *U8:
			intV = int(byte(*sv))
		case Scalar:
			return nil, errors.Errorf(nyiTypeFail, "valueToInts", v)
		default:
			return nil, errors.Errorf("Expected values to be all Scalar Value. Got %v of %T instead", v, v)

		}
		retVal[i] = intV
	}
	return
}

func valuesToTensors(values []Value) (retVal []tensor.Tensor, err error) {
	retVal = make([]tensor.Tensor, len(values))
	for i, v := range values {
		if vt, ok := v.(tensor.Tensor); ok {
			retVal[i] = vt
			continue
		}
		return nil, errors.Errorf("Expected values to all be tensor.Tensor. Got %v of %T in %dth index of the slice", v, v, i)
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

func ones(dt tensor.Dtype, sizes ...int) (retVal Value) {
	if len(sizes) == 0 {
		return one(dt)
	}
	return tensor.Ones(dt, sizes...)
}

func hasInf(v Value) bool {
	switch vt := v.(type) {
	case *F64:
		return math.IsInf(float64(*vt), 0)
	case *F32:
		return math32.IsInf(float32(*vt), 0)
	case *tensor.Dense:
		dt := vt.Dtype()
		if dt != tensor.Float64 && dt != tensor.Float32 {
			return false
		}
		switch dt {
		case tensor.Float32:
			data := vt.Data().([]float32)
			for _, datum := range data {
				if math32.IsInf(datum, 0) {
					return true
				}
			}
		case tensor.Float64:
			data := vt.Data().([]float64)
			for _, datum := range data {
				if math.IsInf(datum, 0) {
					return true
				}
			}
		}
		return false
	case *dualValue:
		return hasInf(vt.Value) || hasInf(vt.d)
	default:
		err := nyi("hasInf", v)
		panic(err)
	}
}

func hasNaN(v Value) bool {
	switch vt := v.(type) {
	case *F64:
		return math.IsNaN(float64(*vt))
	case *F32:
		return math32.IsNaN(float32(*vt))
	case *tensor.Dense:
		dt := vt.Dtype()
		if dt != tensor.Float64 && dt != tensor.Float32 {
			return false
		}
		switch dt {
		case tensor.Float32:
			data := vt.Data().([]float32)
			for _, datum := range data {
				if math32.IsNaN(datum) {
					return true
				}
			}
		case tensor.Float64:
			data := vt.Data().([]float64)
			for _, datum := range data {
				if math.IsNaN(datum) {
					return true
				}
			}
		}
		return false
	case *dualValue:
		return hasNaN(vt.Value) || hasNaN(vt.d)
	default:
		err := nyi("hasNaN", vt)
		panic(err)
	}
}

func setZero(val Value) (retVal Value) {
	switch v := val.(type) {
	case Zeroer:
		v.Zero()
		return v
	case Scalar:
		return zero(v.Dtype())
	default:
		panic(fmt.Sprintf("setZero not implemented yet for %T", v))
	}
}

type arityer interface {
	Arity() int
}

func checkArity(op arityer, inputs int) error {
	if inputs != op.Arity() && op.Arity() >= 0 {
		return errors.Errorf("%v has an arity of %d. Got %d instead", op, op.Arity(), inputs)
	}
	return nil
}
