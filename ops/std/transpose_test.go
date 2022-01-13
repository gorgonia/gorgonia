package stdops

import (
	"math/rand"
	"reflect"
	"testing"
	"testing/quick"

	"github.com/chewxy/hm"
	"gorgonia.org/gorgonia/types"
	"gorgonia.org/shapes"
)

func (op transposeOp) Generate(rand *rand.Rand, size int) reflect.Value {
	d := rand.Intn(12) // we're only going to test up to 12 dims
	pattern := make(shapes.Axes, 0, d)
	for i := 0; i < d; i++ {
		pattern = append(pattern, shapes.Axis(i))
	}
	rand.Shuffle(d, func(i, j int) { pattern[i], pattern[j] = pattern[j], pattern[i] })
	op2 := transposeOp{pattern: pattern}
	return reflect.ValueOf(op2)
}

func TestTranspose_Basic(t *testing.T) {
	// Arity
	arity := func(a transposeOp) bool {
		return a.Arity() == 1
	}
	if err := quick.Check(arity, nil); err != nil {
		t.Error(err)
	}

	// type
	typ := func(a transposeOp) bool {
		d := a.pattern.Dims()
		v := hm.TypeVariable('a')
		tt := types.MakeTensorType(d, v)
		ret := types.MakeDependent(tt, v)
		correct := hm.NewFnType(tt, ret)
		return correct.Eq(a.Type())
	}
	if err := quick.Check(typ, nil); err != nil {
		t.Error(err)
	}

	// ShapeExpr
	shp := func(a transposeOp) bool {
		compExpr, ok := a.ShapeExpr().(shapes.Compound)
		if !ok {
			return ok
		}
		arr, ok := compExpr.Expr.(shapes.Arrow)
		if !ok {
			return ok
		}

		arrB, ok := arr.B.(shapes.Arrow)
		if !ok {
			return ok
		}
		arrBB, ok := arrB.B.(shapes.TransposeOf)
		if !ok {
			return ok
		}
		return reflect.DeepEqual(a.pattern, arrBB.Axes) && reflect.DeepEqual(arr.A, arrBB.A)
	}
	if err := quick.Check(shp, nil); err != nil {
		t.Error(err)
	}
}
