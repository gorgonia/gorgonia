package stdops

import (
	"context"
	"testing"

	"github.com/chewxy/hm"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/gorgonia/internal/datatypes"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/shapes"
	"gorgonia.org/tensor"
)

// Code generated by genops, which is a ops generation tool for Gorgonia. DO NOT EDIT.

func Test_addVV(t *testing.T) {
	op := addVV{}
	// basic test
	assert.Equal(t, 2, op.Arity())

	/* Do (using tensor-tensor) */

	// set up
	var a, b, c values.Value
	var expectedType hm.Type
	var expectedShape shapes.Shape
	var err error

	a = tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))
	b = tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{10, 20, 30, 40, 50, 60}))

	// type and shape checks
	if expectedType, err = typecheck(op, a, b); err != nil {
		t.Fatalf("Expected addVV{} to pass type checking. Err: %v", err)
	}
	if expectedShape, err = shapecheck(op, a, b); err != nil {
		t.Fatalf("Expected addVV{} to pass shape checking. Err: %v", err)
	}

	// actually doing and testing
	if c, err = op.Do(context.Background(), a, b); err != nil {
		t.Fatalf("Expected addVV{} to work correctly. Err: %v", err)
	}
	assert.Equal(t, expectedType, datatypes.TypeOf(c))
	assert.True(t, expectedShape.Eq(c.Shape()))
	correct := []float64{11, 22, 33, 44, 55, 66}
	assert.Equal(t, correct, c.Data())

	/* PreallocDo (using scalar-scalar to make sure things don't go wrong) */

	// set up
	a = tensor.New(tensor.WithShape(), tensor.WithBacking([]float64{1}))
	b = tensor.New(tensor.WithShape(), tensor.WithBacking([]float64{2}))
	c = tensor.New(tensor.WithShape(), tensor.WithBacking([]float64{-1}))

	// type and shape checks
	if expectedType, err = typecheck(op, a, b); err != nil {
		t.Fatalf("Expected addVV{} to pass type checking. Err: %v", err)
	}
	if expectedShape, err = shapecheck(op, a, b); err != nil {
		t.Fatalf("Expected addVV{} to pass shape checking. Err: %v", err)
	}

	// actually PreallocDo-ing and testing
	c, err = op.PreallocDo(context.Background(), c, a, b)
	if err != nil {
		t.Fatalf("Expected addVV{}'s Prealloc to work. Err: %v", err)
	}
	assert.Equal(t, expectedType, datatypes.TypeOf(c))
	assert.True(t, expectedShape.Eq(c.Shape()))
	correctScalar := 3.0
	assert.Equal(t, correctScalar, c.Data())

	// bad cases: fails  typecheck and shapecheck
	a = tensor.New(tensor.WithShape(2, 3), tensor.Of(tensor.Float64))
	b = tensor.New(tensor.WithShape(), tensor.Of(tensor.Float64))
	if expectedType, err = typecheck(op, a, b); err == nil {
		t.Fatalf("Expected addVV{} to NOT pass type checking. Got ~(%v %v) =  %v ", datatypes.TypeOf(a), datatypes.TypeOf(b), expectedType)
	}
	if expectedShape, err = shapecheck(op, a, b); err == nil {
		t.Fatalf("Expected addVV{} to NOT pass shape checking. Got expectedShape = %v", expectedShape)
	}

}

func Test_addVS(t *testing.T) {
	op := addVS{}
	// basic test
	assert.Equal(t, 2, op.Arity())

	/* Do */

	// set up
	var a, b, c values.Value
	var expectedType hm.Type
	var expectedShape shapes.Shape
	var err error

	a = tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))
	b = tensor.New(tensor.WithShape(), tensor.WithBacking([]float64{100}))

	// type and shape checks
	if expectedType, err = typecheck(op, a, b); err != nil {
		t.Fatalf("Expected addVS{} to pass type checking. Err: %v", err)
	}
	if expectedShape, err = shapecheck(op, a, b); err != nil {
		t.Fatalf("Expected addVS{} to pass shape checking. Err: %v", err)
	}

	// actually doing and test
	if c, err = op.Do(context.Background(), a, b); err != nil {
		t.Fatalf("Expected addVS{} to work correctly. Err: %v", err)
	}
	assert.Equal(t, expectedType, datatypes.TypeOf(c))
	assert.True(t, expectedShape.Eq(c.Shape()))
	correct := []float64{101, 102, 103, 104, 105, 106}
	assert.Equal(t, correct, c.Data())

	/* PreallocDo */

	// set up - create a new preallocated result
	c = tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{-1, -1, -1, -1, -1, -1}))

	// actually PreallocDo-ing and checking
	c, err = op.PreallocDo(context.Background(), c, a, b)
	if err != nil {
		t.Fatalf("Expected addVS{}'s Prealloc to work. Err: %v", err)
	}
	assert.Equal(t, expectedType, datatypes.TypeOf(c))
	assert.True(t, expectedShape.Eq(c.Shape()))
	assert.Equal(t, correct, c.Data())

	/* bad cases: addVS{} on tensor-tensor */

	b = tensor.New(tensor.WithShape(2, 3), tensor.Of(tensor.Float64))
	// we won't type check because the type system is not a dependent type system, thus
	// addVS : (a → b → a) will always type check without errors
	if expectedShape, err = shapecheck(op, a, b); err == nil {
		t.Fatalf("Expected addVS{} to NOT pass shape checking. Got %v ~ (%v, %v) = %v", op.ShapeExpr(), a.Shape(), b.Shape(), expectedShape)
	}
}

func Test_addSV(t *testing.T) {
	op := addSV{}
	// basic test
	assert.Equal(t, 2, op.Arity())

	/* Do */

	// set up
	var a, b, c values.Value
	var expectedType hm.Type
	var expectedShape shapes.Shape
	var err error

	a = tensor.New(tensor.WithShape(), tensor.WithBacking([]float64{100}))
	b = tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))

	// type and shape checks
	if expectedType, err = typecheck(op, a, b); err != nil {
		t.Fatalf("Expected addSV{} to pass type checking. Err: %v", err)
	}
	if expectedShape, err = shapecheck(op, a, b); err != nil {
		t.Fatalf("Expected addSV{} to pass shape checking. Err: %v", err)
	}

	// actually doing and test
	if c, err = op.Do(context.Background(), a, b); err != nil {
		t.Fatalf("Expected addSV{} to work correctly. Err: %v", err)
	}
	assert.Equal(t, expectedType, datatypes.TypeOf(c))
	assert.True(t, expectedShape.Eq(c.Shape()))
	correct := []float64{101, 102, 103, 104, 105, 106}
	assert.Equal(t, correct, c.Data())

	/* PreallocDo */

	// set up - create a new preallocated result
	c = tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{-1, -1, -1, -1, -1, -1}))

	// actually PreallocDo-ing and checking
	c, err = op.PreallocDo(context.Background(), c, a, b)
	if err != nil {
		t.Fatalf("Expected addVS{}'s Prealloc to work. Err: %v", err)
	}
	assert.Equal(t, expectedType, datatypes.TypeOf(c))
	assert.True(t, expectedShape.Eq(c.Shape()))
	assert.Equal(t, correct, c.Data())

	/* bad cases: addSV{} on tensor-tensor */

	a = tensor.New(tensor.WithShape(2, 3), tensor.Of(tensor.Float64))
	// we won't type check because the type system is not a dependent type system, thus
	// addSV : (a → b → b) will always type check without errors
	if expectedShape, err = shapecheck(op, a, b); err == nil {
		t.Fatalf("Expected addSV{} to NOT pass shape checking. Got %v ~ (%v, %v) = %v", op.ShapeExpr(), a.Shape(), b.Shape(), expectedShape)
	}
}