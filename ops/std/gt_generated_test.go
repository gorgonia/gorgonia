package stdops

import (
	"context"
	"testing"

	"github.com/chewxy/hm"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/gorgonia/internal/datatypes"
	"gorgonia.org/shapes"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/dense"
)

// Code generated by genops, which is a ops generation tool for Gorgonia. DO NOT EDIT.

func Test_gtVV(t *testing.T) {
	op := gtVV[float64, *dense.Dense[float64]]{}
	// basic test
	assert.Equal(t, 2, op.Arity())

	/* Do (using tensor-tensor) */

	// set up
	var a, b, c *dense.Dense[float64]
	var expectedType hm.Type
	var expectedShape shapes.Shape
	var err error

	a = dense.New[float64](tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))
	b = dense.New[float64](tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 40, 50, 60}))

	// type and shape checks
	if expectedType, err = typecheck(op, a, b); err != nil {
		t.Fatalf("Expected gtVV{} to pass type checking. Err: %v", err)
	}
	if expectedShape, err = shapecheck(op, a, b); err != nil {
		t.Fatalf("Expected gtVV{} to pass shape checking. Err: %v", err)
	}

	// actually doing and testing
	if c, err = op.Do(context.Background(), a, b); err != nil {
		t.Fatalf("Expected gtVV{} to work correctly. Err: %v", err)
	}
	assert.Equal(t, expectedType, datatypes.TypeOf[float64](c))
	assert.True(t, expectedShape.Eq(c.Shape()))
	correct := []bool{false, false, false, false, false, false}
	assert.Equal(t, correct, c.Data())

	/* PreallocDo (using scalar-scalar to make sure things don't go wrong) */

	// set up
	a = dense.New[float64](tensor.WithShape(), tensor.WithBacking([]float64{1}))
	b = dense.New[float64](tensor.WithShape(), tensor.WithBacking([]float64{2}))
	c = dense.New[float64](tensor.WithShape(), tensor.WithBacking([]bool{false}))

	// type and shape checks
	if expectedType, err = typecheck(op, a, b); err != nil {
		t.Fatalf("Expected gtVV{} to pass type checking. Err: %v", err)
	}
	if expectedShape, err = shapecheck(op, a, b); err != nil {
		t.Fatalf("Expected gtVV{} to pass shape checking. Err: %v", err)
	}

	// actually PreallocDo-ing and testing
	c, err = op.PreallocDo(context.Background(), c, a, b)
	if err != nil {
		t.Fatalf("Expected gtVV{}'s Prealloc to work. Err: %v", err)
	}
	assert.Equal(t, expectedType, datatypes.TypeOf[float64](c))
	assert.True(t, expectedShape.Eq(c.Shape()))
	correctScalar := false
	assert.Equal(t, correctScalar, c.Data())

	// bad cases: fails  typecheck and shapecheck
	a = dense.New[float64](tensor.WithShape(2, 3))
	b = dense.New[float64](tensor.WithShape())
	if expectedType, err = typecheck(op, a, b); err == nil {
		t.Fatalf("Expected gtVV{} to NOT pass type checking. Got ~(%v %v) =  %v ", datatypes.TypeOf[float64](a), datatypes.TypeOf[float64](b), expectedType)
	}
	if expectedShape, err = shapecheck(op, a, b); err == nil {
		t.Fatalf("Expected gtVV{} to NOT pass shape checking. Got expectedShape = %v", expectedShape)
	}

}

func Test_gtVS(t *testing.T) {
	op := gtVS[float64, *dense.Dense[float64]]{}
	// basic test
	assert.Equal(t, 2, op.Arity())

	/* Do */

	// set up
	var a, b, c *dense.Dense[float64]
	var expectedType hm.Type
	var expectedShape shapes.Shape
	var err error

	a = dense.New[float64](tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))
	b = dense.New[float64](tensor.WithShape(), tensor.WithBacking([]float64{100}))

	// type and shape checks
	if expectedType, err = typecheck(op, a, b); err != nil {
		t.Fatalf("Expected gtVS{} to pass type checking. Err: %v", err)
	}
	if expectedShape, err = shapecheck(op, a, b); err != nil {
		t.Fatalf("Expected gtVS{} to pass shape checking. Err: %v", err)
	}

	// actually doing and test
	if c, err = op.Do(context.Background(), a, b); err != nil {
		t.Fatalf("Expected gtVS{} to work correctly. Err: %v", err)
	}
	assert.Equal(t, expectedType, datatypes.TypeOf[float64](c))
	assert.True(t, expectedShape.Eq(c.Shape()))
	correct := []bool{false, false, false, false, false, false}
	assert.Equal(t, correct, c.Data())

	/* PreallocDo */

	// set up - create a new preallocated result
	c = dense.New[float64](tensor.WithShape(2, 3), tensor.WithBacking([]bool{false, false, false, false, false, false}))

	// actually PreallocDo-ing and checking
	c, err = op.PreallocDo(context.Background(), c, a, b)
	if err != nil {
		t.Fatalf("Expected gtVS{}'s Prealloc to work. Err: %v", err)
	}
	assert.Equal(t, expectedType, datatypes.TypeOf[float64](c))
	assert.True(t, expectedShape.Eq(c.Shape()))
	assert.Equal(t, correct, c.Data())

	/* bad cases: gtVS{} on tensor-tensor */

	b = dense.New[float64](tensor.WithShape(2, 3))
	// we won't type check because the type system is not a dependent type system, thus
	// gtVS : (a → b → a) will always type check without errors
	if expectedShape, err = shapecheck(op, a, b); err == nil {
		t.Fatalf("Expected gtVS{} to NOT pass shape checking. Got %v ~ (%v, %v) = %v", op.ShapeExpr(), a.Shape(), b.Shape(), expectedShape)
	}
}

func Test_gtSV(t *testing.T) {
	op := gtSV[float64, *dense.Dense[float64]]{}
	// basic test
	assert.Equal(t, 2, op.Arity())

	/* Do */

	// set up
	var a, b, c *dense.Dense[float64]
	var expectedType hm.Type
	var expectedShape shapes.Shape
	var err error

	a = dense.New[float64](tensor.WithShape(), tensor.WithBacking([]float64{100}))
	b = dense.New[float64](tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))

	// type and shape checks
	if expectedType, err = typecheck(op, a, b); err != nil {
		t.Fatalf("Expected gtSV{} to pass type checking. Err: %v", err)
	}
	if expectedShape, err = shapecheck(op, a, b); err != nil {
		t.Fatalf("Expected gtSV{} to pass shape checking. Err: %v", err)
	}

	// actually doing and test
	if c, err = op.Do(context.Background(), a, b); err != nil {
		t.Fatalf("Expected gtSV{} to work correctly. Err: %v", err)
	}
	assert.Equal(t, expectedType, datatypes.TypeOf[float64](c))
	assert.True(t, expectedShape.Eq(c.Shape()))
	correct := []bool{true, true, true, true, true, true}
	assert.Equal(t, correct, c.Data())

	/* PreallocDo */

	// set up - create a new preallocated result
	c = dense.New[float64](tensor.WithShape(2, 3), tensor.WithBacking([]bool{false, false, false, false, false, false}))

	// actually PreallocDo-ing and checking
	c, err = op.PreallocDo(context.Background(), c, a, b)
	if err != nil {
		t.Fatalf("Expected gtVS{}'s Prealloc to work. Err: %v", err)
	}
	assert.Equal(t, expectedType, datatypes.TypeOf[float64](c))
	assert.True(t, expectedShape.Eq(c.Shape()))
	assert.Equal(t, correct, c.Data())

	/* bad cases: gtSV{} on tensor-tensor */

	a = dense.New[float64](tensor.WithShape(2, 3))
	// we won't type check because the type system is not a dependent type system, thus
	// gtSV : (a → b → b) will always type check without errors
	if expectedShape, err = shapecheck(op, a, b); err == nil {
		t.Fatalf("Expected gtSV{} to NOT pass shape checking. Got %v ~ (%v, %v) = %v", op.ShapeExpr(), a.Shape(), b.Shape(), expectedShape)
	}
}

func Test_gtVV_RetSame(t *testing.T) {
	op := gtVV[float64, *dense.Dense[float64]]{gtOp[float64, *dense.Dense[float64]]{retSame: true}, binopVV{}}
	// basic test
	assert.Equal(t, 2, op.Arity())

	/* Do (using tensor-tensor) */

	// set up
	var a, b, c *dense.Dense[float64]
	var expectedType hm.Type
	var expectedShape shapes.Shape
	var err error

	a = dense.New[float64](tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))
	b = dense.New[float64](tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 40, 50, 60}))

	// type and shape checks
	if expectedType, err = typecheck(op, a, b); err != nil {
		t.Fatalf("Expected gtVV{} to pass type checking. Err: %v", err)
	}
	if expectedShape, err = shapecheck(op, a, b); err != nil {
		t.Fatalf("Expected gtVV{} to pass shape checking. Err: %v", err)
	}

	// actually doing and testing
	if c, err = op.Do(context.Background(), a, b); err != nil {
		t.Fatalf("Expected gtVV{} to work correctly. Err: %v", err)
	}
	assert.Equal(t, expectedType, datatypes.TypeOf[float64](c))
	assert.True(t, expectedShape.Eq(c.Shape()))
	correct := []float64{0, 0, 0, 0, 0, 0}
	assert.Equal(t, correct, c.Data())

	/* PreallocDo (using scalar-scalar to make sure things don't go wrong) */

	// set up
	a = dense.New[float64](tensor.WithShape(), tensor.WithBacking([]float64{1}))
	b = dense.New[float64](tensor.WithShape(), tensor.WithBacking([]float64{2}))
	c = dense.New[float64](tensor.WithShape(), tensor.WithBacking([]float64{0}))

	// type and shape checks
	if expectedType, err = typecheck(op, a, b); err != nil {
		t.Fatalf("Expected gtVV{} to pass type checking. Err: %v", err)
	}
	if expectedShape, err = shapecheck(op, a, b); err != nil {
		t.Fatalf("Expected gtVV{} to pass shape checking. Err: %v", err)
	}

	// actually PreallocDo-ing and testing
	c, err = op.PreallocDo(context.Background(), c, a, b)
	if err != nil {
		t.Fatalf("Expected gtVV{}'s Prealloc to work. Err: %v", err)
	}
	assert.Equal(t, expectedType, datatypes.TypeOf[float64](c))
	assert.True(t, expectedShape.Eq(c.Shape()))
	correctScalar := 0.0
	assert.Equal(t, correctScalar, c.Data())

	// bad cases: fails  typecheck and shapecheck
	a = dense.New[float64](tensor.WithShape(2, 3))
	b = dense.New[float64](tensor.WithShape())
	if expectedType, err = typecheck(op, a, b); err == nil {
		t.Fatalf("Expected gtVV{} to NOT pass type checking. Got ~(%v %v) =  %v ", datatypes.TypeOf[float64](a), datatypes.TypeOf[float64](b), expectedType)
	}
	if expectedShape, err = shapecheck(op, a, b); err == nil {
		t.Fatalf("Expected gtVV{} to NOT pass shape checking. Got expectedShape = %v", expectedShape)
	}

}

func Test_gtVS_RetSame(t *testing.T) {
	op := gtVS[float64, *dense.Dense[float64]]{gtOp[float64, *dense.Dense[float64]]{retSame: true}, binopVS{}}
	// basic test
	assert.Equal(t, 2, op.Arity())

	/* Do */

	// set up
	var a, b, c *dense.Dense[float64]
	var expectedType hm.Type
	var expectedShape shapes.Shape
	var err error

	a = dense.New[float64](tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))
	b = dense.New[float64](tensor.WithShape(), tensor.WithBacking([]float64{100}))

	// type and shape checks
	if expectedType, err = typecheck(op, a, b); err != nil {
		t.Fatalf("Expected gtVS{} to pass type checking. Err: %v", err)
	}
	if expectedShape, err = shapecheck(op, a, b); err != nil {
		t.Fatalf("Expected gtVS{} to pass shape checking. Err: %v", err)
	}

	// actually doing and test
	if c, err = op.Do(context.Background(), a, b); err != nil {
		t.Fatalf("Expected gtVS{} to work correctly. Err: %v", err)
	}
	assert.Equal(t, expectedType, datatypes.TypeOf[float64](c))
	assert.True(t, expectedShape.Eq(c.Shape()))
	correct := []float64{0, 0, 0, 0, 0, 0}
	assert.Equal(t, correct, c.Data())

	/* PreallocDo */

	// set up - create a new preallocated result
	c = dense.New[float64](tensor.WithShape(2, 3), tensor.WithBacking([]float64{0, 0, 0, 0, 0, 0}))

	// actually PreallocDo-ing and checking
	c, err = op.PreallocDo(context.Background(), c, a, b)
	if err != nil {
		t.Fatalf("Expected gtVS{}'s Prealloc to work. Err: %v", err)
	}
	assert.Equal(t, expectedType, datatypes.TypeOf[float64](c))
	assert.True(t, expectedShape.Eq(c.Shape()))
	assert.Equal(t, correct, c.Data())

	/* bad cases: gtVS{} on tensor-tensor */

	b = dense.New[float64](tensor.WithShape(2, 3))
	// we won't type check because the type system is not a dependent type system, thus
	// gtVS : (a → b → a) will always type check without errors
	if expectedShape, err = shapecheck(op, a, b); err == nil {
		t.Fatalf("Expected gtVS{} to NOT pass shape checking. Got %v ~ (%v, %v) = %v", op.ShapeExpr(), a.Shape(), b.Shape(), expectedShape)
	}
}

func Test_gtSV_RetSame(t *testing.T) {
	op := gtSV[float64, *dense.Dense[float64]]{gtOp[float64, *dense.Dense[float64]]{retSame: true}, binopSV{}}
	// basic test
	assert.Equal(t, 2, op.Arity())

	/* Do */

	// set up
	var a, b, c *dense.Dense[float64]
	var expectedType hm.Type
	var expectedShape shapes.Shape
	var err error

	a = dense.New[float64](tensor.WithShape(), tensor.WithBacking([]float64{100}))
	b = dense.New[float64](tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))

	// type and shape checks
	if expectedType, err = typecheck(op, a, b); err != nil {
		t.Fatalf("Expected gtSV{} to pass type checking. Err: %v", err)
	}
	if expectedShape, err = shapecheck(op, a, b); err != nil {
		t.Fatalf("Expected gtSV{} to pass shape checking. Err: %v", err)
	}

	// actually doing and test
	if c, err = op.Do(context.Background(), a, b); err != nil {
		t.Fatalf("Expected gtSV{} to work correctly. Err: %v", err)
	}
	assert.Equal(t, expectedType, datatypes.TypeOf[float64](c))
	assert.True(t, expectedShape.Eq(c.Shape()))
	correct := []float64{1, 1, 1, 1, 1, 1}
	assert.Equal(t, correct, c.Data())

	/* PreallocDo */

	// set up - create a new preallocated result
	c = dense.New[float64](tensor.WithShape(2, 3), tensor.WithBacking([]float64{0, 0, 0, 0, 0, 0}))

	// actually PreallocDo-ing and checking
	c, err = op.PreallocDo(context.Background(), c, a, b)
	if err != nil {
		t.Fatalf("Expected gtVS{}'s Prealloc to work. Err: %v", err)
	}
	assert.Equal(t, expectedType, datatypes.TypeOf[float64](c))
	assert.True(t, expectedShape.Eq(c.Shape()))
	assert.Equal(t, correct, c.Data())

	/* bad cases: gtSV{} on tensor-tensor */

	a = dense.New[float64](tensor.WithShape(2, 3))
	// we won't type check because the type system is not a dependent type system, thus
	// gtSV : (a → b → b) will always type check without errors
	if expectedShape, err = shapecheck(op, a, b); err == nil {
		t.Fatalf("Expected gtSV{} to NOT pass shape checking. Got %v ~ (%v, %v) = %v", op.ShapeExpr(), a.Shape(), b.Shape(), expectedShape)
	}
}
