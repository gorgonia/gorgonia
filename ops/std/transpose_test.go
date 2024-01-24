package stdops

import (
	"context"
	"math/rand"
	"reflect"
	"testing"
	"testing/quick"

	"github.com/chewxy/hm"
	"gorgonia.org/gorgonia/internal/datatypes"
	"gorgonia.org/gorgonia/types"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/shapes"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/dense"
)

func genTrans[DT any, T values.Value[DT]](d int) transposeOp[DT, T] {
	pattern := make(shapes.Axes, 0, d)
	for i := 0; i < d; i++ {
		pattern = append(pattern, shapes.Axis(i))
	}
	for i := 0; i < d; i++ {
		rand.Shuffle(d, func(i, j int) { pattern[i], pattern[j] = pattern[j], pattern[i] })
	}
	return transposeOp[DT, T]{pattern: pattern}

}

func (op transposeOp[DT, T]) Generate(rand *rand.Rand, size int) reflect.Value {
	d := rand.Intn(12)
	return reflect.ValueOf(genTrans[DT, T](d))
}

func TestTranspose_Basic(t *testing.T) {
	// Arity
	arity := func(a transposeOp[float64, *dense.Dense[float64]]) bool {
		return a.Arity() == 1
	}
	if err := quick.Check(arity, nil); err != nil {
		t.Error(err)
	}

	// type
	typ := func(a transposeOp[float64, *dense.Dense[float64]]) bool {
		d := a.pattern.Dims()
		v := hm.TypeVariable('a')
		tt := types.MakeTensorType(d, v)
		correct := types.NewFunc(tt, tt)
		return correct.Eq(a.Type())
	}
	if err := quick.Check(typ, nil); err != nil {
		t.Error(err)
	}

	// ShapeExpr
	shp := func(a transposeOp[float64, *dense.Dense[float64]]) bool {
		compExpr, ok := a.ShapeExpr().(shapes.Compound)
		if !ok {
			return ok
		}
		arr, ok := compExpr.Expr.(shapes.Arrow)
		if !ok {
			return ok
		}

		arrBB, ok := arr.B.(shapes.TransposeOf)
		if !ok {
			return ok
		}
		return reflect.DeepEqual(a.pattern, arrBB.Axes) && reflect.DeepEqual(arr.A, arrBB.A)
	}
	if err := quick.Check(shp, nil); err != nil {
		t.Error(err)
	}

	do := func(tt tTensor[float64]) bool {
		a := tt.Dense
		d := a.Dims()
		op := genTrans[float64, *dense.Dense[float64]](d)

		expectedType, err := typecheck(op, a)
		if err != nil {
			t.Errorf("%v failed typechecking. Error: %v", op, err)
			return false
		}

		inferred, err := shapes.InferApp(op.ShapeExpr(), a.Shape())
		if err != nil {
			t.Logf("%v @ %v ⇒ %v", op.ShapeExpr(), a.Shape(), inferred)
			t.Errorf("%v", err)
			return false
		}

		expectedShape, err := shapecheck(op, a)
		if err != nil {
			t.Errorf("%v failed shapecheck. Error: %v. ", op, err)
			return false
		}

		b, err := op.Do(context.Background(), a)
		if err != nil {
			t.Errorf("Expected %v to work correctly. Error: %v", op, err)
			return false
		}

		return b.Shape().Eq(expectedShape) && datatypes.TypeOf(b).Eq(expectedType)
	}
	if err := quick.Check(do, nil); err != nil {
		t.Error(err)
	}

}

func TestTransposeScalar(t *testing.T) {
	s := dense.New[float64](tensor.FromScalar(1337.0))
	op := genTrans[float64, *dense.Dense[float64]](s.Dims())

	expectedType, err := typecheck(op, s)
	if err != nil {
		t.Errorf("%v failed typechecking. Error: %v", op, err)
	}

	t.Logf("op.Type %T, s.Type %T", op.Type().Types()[0], datatypes.TypeOf(s))
	t.Logf("expected %v", expectedType)
}
