package gorgonia

import (
	"fmt"

	tf64 "github.com/chewxy/gorgonia/tensor/f64"
	"github.com/chewxy/hm"
	"github.com/pkg/errors"
)

type errorStacker interface {
	ErrorStack() string
}

const EPSILON float64 = 1e-10

func floatEquals(a, b float64) bool {
	if (a-b) < EPSILON && (b-a) < EPSILON {
		return true
	}
	return false
}

func floatsEqual(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}

	for i, v := range a {
		if !floatEquals(v, b[i]) {
			return false
		}
	}
	return true
}

func extractF64s(v Value) []float64 {
	var t *tf64.Tensor
	var ok bool
	if t, ok = v.(*tf64.Tensor); !ok {
		panic("Only works for tf64.Tensor")
	}

	return t.Data().([]float64)
}

func extractF64(v Value) float64 {
	if f, ok := v.(F64); ok {
		return float64(f)
	}
	panic("Only works for F64!")
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
	x = NewVector(g, Float64, WithName("x"), WithShape(2, 1))
	y = NewVector(g, Float64, WithName("y"), WithShape(2, 1))
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
	x = NewVector(g, Float64, WithName("x"), WithShape(2, 1))
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
