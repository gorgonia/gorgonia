package gorgonia

import (
	"fmt"
	"hash/fnv"
	"log"
	"math"

	"github.com/chewxy/math32"
	"github.com/pkg/errors"
	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/iterator"
	"gorgonia.org/tensor"
)

const (
	maxFloat32 = math32.MaxFloat32
	maxFloat64 = math.MaxFloat64
)

// NodesToValueGrads is a utility function that converts a Nodes to a slice of ValueGrad for the solvers
func NodesToValueGrads(in Nodes) (out []ValueGrad) {
	out = make([]ValueGrad, len(in))
	for i := range in {
		out[i] = in[i]
	}
	return out
}

func graphNodeToNode(in graph.Nodes) (out Nodes) {
	out = make(Nodes, in.Len())
	for i := 0; in.Next(); i++ {
		out[i] = in.Node().(*Node)
	}

	return
}

func sliceNodesToNodes(in []graph.Node) (out Nodes) {
	out = make(Nodes, len(in))
	for i := range in {
		out[i] = in[i].(*Node)
	}
	return
}

func nodeToGraphNode(in []*Node) graph.Nodes {
	nodes := make([]graph.Node, len(in))
	for i, n := range in {
		nodes[i] = n
	}
	return iterator.NewOrderedNodes(nodes)
}

func tensorInfo(t tensor.Tensor) (dt tensor.Dtype, dim int) {
	dt = t.Dtype()
	dim = t.Dims()
	return
}

// valuesToInts will FORCIBLY cast floats to ints.
func valuesToInts(values []Value) (retVal []int, err error) {
	retVal = tensor.BorrowInts(len(values))
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

func hasInf(v Value, dev Device) bool {
	switch vt := v.(type) {
	case *F64:
		return math.IsInf(float64(*vt), 0)
	case *F32:
		return math32.IsInf(float32(*vt), 0)
	case tensor.Tensor:
		if e, ok := vt.Engine().(tensor.InfChecker); ok {
			ok, _ := e.HasInf(vt) // BUG: errors not checked
			return ok
		}

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
		return hasInf(vt.Value, dev) || hasInf(vt.d, dev)
	default:
		err := nyi("hasInf", v)
		panic(err)
	}
}

func hasNaN(v Value, dev Device) bool {
	switch vt := v.(type) {
	case *F64:
		return math.IsNaN(float64(*vt))
	case *F32:
		return math32.IsNaN(float32(*vt))
	case tensor.Tensor:
		if e, ok := vt.Engine().(tensor.NaNChecker); ok {
			ok, _ := e.HasNaN(vt) // BUG: errors not checked
			return ok
		}
		log.Printf("Value's engine %T", vt.Engine())

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
		return hasNaN(vt.Value, dev) || hasNaN(vt.d, dev)
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

func checkArity(op arityer, inputs int) error {
	if inputs != op.Arity() && op.Arity() >= 0 {
		return errors.Errorf("%v has an arity of %d. Got %d instead", op, op.Arity(), inputs)
	}
	return nil
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func ceilDivInt(a, b int) int {
	return (a + b - 1) / b
}

func simpleHash(op hashWriter) uint32 {
	h := fnv.New32a()
	op.WriteHash(h)
	return h.Sum32()
}

func getDV(x, y *Node) (xdv, ydv *dualValue) {
	return x.boundTo.(*dualValue), y.boundTo.(*dualValue)
}

func getDV3(x, y, z *Node) (xdv, ydv, zdv *dualValue) {
	return x.boundTo.(*dualValue), y.boundTo.(*dualValue), z.boundTo.(*dualValue)
}

func getConst(x *Node, constant string) (retVal *Node, err error) {
	var dt tensor.Dtype
	if dt, err = dtypeOf(x.t); err != nil {
		return nil, errors.Wrap(err, dtypeOfFail)
	}

	if m, ok := constmap[constant]; ok {
		if n, ok := m[dt]; ok {
			return n, nil
		}
	}
	return nil, errors.Errorf("constant %v not provided for %v", constant, dt)
}

func scalarEquiv(s tensor.Shape) bool {
	if len(s) == 0 {
		return true
	}
	prod := 1
	for _, v := range s {
		prod *= v
	}

	return prod == 1
}
