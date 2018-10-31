package gorgonia

import (
	"fmt"
	"log"
	"math/rand"
	"reflect"
	"runtime"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/dawson"
	"gorgonia.org/tensor"
)

type errorStacker interface {
	ErrorStack() string
}

func floatsEqual64(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}

	for i, v := range a {
		if !dawson.CloseF64(v, b[i]) {
			return false
		}
	}
	return true
}

func floatsEqual32(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}

	for i, v := range a {
		if !dawson.CloseF32(v, b[i]) {
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
	case *F64:
		return float64(*vt)
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
	case *F32:
		return float32(*vt)
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
	z = Must(Add(x, y, 0))
	return
}

func simpleVecEqn() (g *ExprGraph, x, y, z *Node) {
	g = NewGraph()
	x = NewVector(g, Float64, WithName("x"), WithShape(2))
	y = NewVector(g, Float64, WithName("y"), WithShape(2))
	z = Must(Add(x, y, 0))
	return
}

func simpleEqn() (g *ExprGraph, x, y, z *Node) {
	g = NewGraph()
	x = NewScalar(g, Float64, WithName("x"))
	y = NewScalar(g, Float64, WithName("y"))
	z = Must(Add(x, y, 0))
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

type assertState struct {
	*assert.Assertions
	cont bool
}

func newAssertState(a *assert.Assertions) *assertState { return &assertState{a, true} }

func (a *assertState) Equal(expected interface{}, actual interface{}, msgAndArgs ...interface{}) {
	if !a.cont {
		return
	}
	a.cont = a.Assertions.Equal(expected, actual, msgAndArgs...)
}

func (a *assertState) True(value bool, msgAndArgs ...interface{}) {
	if !a.cont {
		return
	}
	a.cont = a.Assertions.True(value, msgAndArgs...)
}

func deepNodeEq(a, b *Node) bool {
	if a == b {
		return true
	}

	if a.isInput() {
		if !b.isInput() {
			return false
		}

		if a.name != b.name {
			return false
		}
		if !ValueEq(a.boundTo, b.boundTo) {
			return false
		}
		return true
	}

	if b.isInput() {
		return false
	}

	if a.name != b.name {
		return false
	}

	if a.group != b.group {
		return false
	}

	if a.id != b.id {
		return false
	}

	if a.hash != b.hash {
		return false
	}

	if a.hashed != b.hashed {
		return false
	}

	if a.inferredShape != b.inferredShape {
		return false
	}

	if a.unchanged != b.unchanged {
		return false
	}

	if a.isStmt != b.isStmt {
		return false
	}

	if a.ofInterest != b.ofInterest {
		return false
	}

	if a.dataOn != b.dataOn {
		return false
	}

	if !a.t.Eq(b.t) {
		return false
	}
	if !a.shape.Eq(b.shape) {
		return false
	}

	if a.op.Hashcode() != b.op.Hashcode() {
		return false
	}

	if !ValueEq(a.boundTo, b.boundTo) {
		return false
	}

	if len(a.children) != len(b.children) {
		return false
	}

	if len(a.derivOf) != len(b.derivOf) {
		return false
	}

	if a.deriv != nil {
		if b.deriv == nil {
			return false
		}
		if a.deriv.Hashcode() != b.deriv.Hashcode() {
			return false
		}
	}

	for i, c := range a.children {
		if c.Hashcode() != b.children[i].Hashcode() {
			return false
		}
	}

	for i, c := range a.derivOf {
		if c.Hashcode() != b.derivOf[i].Hashcode() {
			return false
		}
	}
	return true
}

// TensorGenerator only generates Dense tensors for now
type TensorGenerator struct {
	ShapeConstraint tensor.Shape // [0, 6, 0] implies that the second dimension is the constraint. 0 is any.
	DtypeConstraint tensor.Dtype
}

func (g TensorGenerator) Generate(r *rand.Rand, size int) reflect.Value {
	// shape := g.ShapeConstraint
	// of := g.DtypeConstraint

	// if g.ShapeConstraint == nil {
	// 	// generate
	// } else {
	// 	// generate for 0s in constraints
	// }

	// if g.DtypeConstraint == (tensor.Dtype{}) {
	// 	of = g.DtypeConstraint
	// }
	var retVal Value

	return reflect.ValueOf(retVal)
}

type ValueGenerator struct {
	ShapeConstraint tensor.Shape // [0, 6, 0] implies that the second dimension is the constraint. 0 is any.
	DtypeConstraint tensor.Dtype
}

func (g ValueGenerator) Generate(r *rand.Rand, size int) reflect.Value {
	// generate scalar or tensor
	ri := r.Intn(2)
	if ri == 0 {
		gen := TensorGenerator{
			ShapeConstraint: g.ShapeConstraint,
			DtypeConstraint: g.DtypeConstraint,
		}
		return gen.Generate(r, size)

	}
	var retVal Value
	// of := acceptableDtypes[r.Intn(len(acceptableDtypes))]

	return reflect.ValueOf(retVal)
}

type NodeGenerator struct{}

func (g NodeGenerator) Generate(r *rand.Rand, size int) reflect.Value {
	var n *Node
	return reflect.ValueOf(n)
}
