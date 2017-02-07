package gorgonia

import (
	"fmt"
	"log"
	"runtime"

	"github.com/chewxy/gorgonia/tensor"
	"github.com/chewxy/hm"
	"github.com/pkg/errors"
)

type errorStacker interface {
	ErrorStack() string
}

const EPSILON64 float64 = 1e-10
const EPSILON32 float32 = 1e-5

func floatEquals64(a, b float64) bool {
	if (a-b) < EPSILON64 && (b-a) < EPSILON64 {
		return true
	}
	return false
}

func floatsEqual64(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}

	for i, v := range a {
		if !floatEquals64(v, b[i]) {
			return false
		}
	}
	return true
}

func floatEquals32(a, b float32) bool {
	if (a-b) < EPSILON32 && (b-a) < EPSILON32 {
		return true
	}
	return false
}

func floatsEqual32(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}

	for i, v := range a {
		if !floatEquals32(v, b[i]) {
			return false
		}
	}
	return true
}

func extractF64s(v Value) []float64 {
	return v.Data().([]float64)
}

func extractF64(v Value) float64 {
	switch vt := v.(type) {
	case F64:
		return float64(vt)
	case tensor.Tensor:
		if !vt.IsScalar() {
			panic("Got a non scalar result!")
		}
		pc, _, _, _ := runtime.Caller(1)
		log.Printf("Better watch it: %v called with a Scalar tensor", runtime.FuncForPC(pc).Name())
		return vt.ScalarValue().(float64)
	}
	panic(fmt.Sprintf("Unhandled types! Got %v of %T instead", v, v))
}

func extractF32s(v Value) []float32 {
	return v.Data().([]float32)
}

func extractF32(v Value) float32 {
	switch vt := v.(type) {
	case F32:
		return float32(vt)
	case tensor.Tensor:
		if !vt.IsScalar() {
			panic("Got a non scalar result!")
		}
		pc, _, _, _ := runtime.Caller(1)
		log.Printf("Better watch it: %v called with a Scalar tensor", runtime.FuncForPC(pc).Name())
		return vt.ScalarValue().(float32)
	}
	panic(fmt.Sprintf("Unhandled types! Got %v of %T instead", v, v))
}

func f64sTof32s(f []float64) []float32 {
	retVal := make([]float32, len(f))
	for i, v := range f {
		retVal[i] = float32(v)
	}
	return retVal
}

func simpleMatEqn() (g *ExprGraph, x, y, z *Node) {
	g = NewGraph()
	x = NewMatrix(g, Float64, WithName("x"), WithShape(2, 2))
	y = NewMatrix(g, Float64, WithName("y"), WithShape(2, 2))
	z = Must(Add(x, y))
	return
}

func simpleVecEqn() (g *ExprGraph, x, y, z *Node) {
	g = NewGraph()
	x = NewVector(g, Float64, WithName("x"), WithShape(2))
	y = NewVector(g, Float64, WithName("y"), WithShape(2))
	z = Must(Add(x, y))
	return
}

func simpleEqn() (g *ExprGraph, x, y, z *Node) {
	g = NewGraph()
	x = NewScalar(g, Float64, WithName("x"))
	y = NewScalar(g, Float64, WithName("y"))
	z = Must(Add(x, y))
	return
}

func simpleUnaryEqn() (g *ExprGraph, x, y *Node) {
	g = NewGraph()
	x = NewScalar(g, Float64, WithName("x"))
	y = Must(Square(x))
	return
}

func simpleUnaryVecEqn() (g *ExprGraph, x, y *Node) {
	g = NewGraph()
	x = NewVector(g, Float64, WithName("x"), WithShape(2))
	y = Must(Square(x))
	return
}

type malformed struct{}

func (t malformed) Name() string                   { return "malformed" }
func (t malformed) Format(state fmt.State, c rune) { fmt.Fprintf(state, "malformed") }
func (t malformed) String() string                 { return "malformed" }
func (t malformed) Apply(hm.Subs) hm.Substitutable { return t }
func (t malformed) FreeTypeVar() hm.TypeVarSet     { return nil }
func (t malformed) Eq(hm.Type) bool                { return false }
func (t malformed) Types() hm.Types                { return nil }
func (t malformed) Normalize(a, b hm.TypeVarSet) (hm.Type, error) {
	return nil, errors.Errorf("cannot normalize malformed")
}
