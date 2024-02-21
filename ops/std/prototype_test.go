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

var (
	_ Abser[float64, *dense.Dense[float64]] = dense.StdFloat64Engine[*dense.Dense[float64]]{}
)

func TestAdd__(t *testing.T) {
	op := addVV[float64, *dense.Dense[float64]]{}
	// basic test
	assert.Equal(t, 2, op.Arity())

	// tensor-tensor / Do()

	var a, b, c *dense.Dense[float64]
	a = dense.New[float64](tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))
	b = dense.New[float64](tensor.WithShape(2, 3), tensor.WithBacking([]float64{10, 20, 30, 40, 50, 60}))

	var expectedType hm.Type
	var expectedShape shapes.Shape
	var err error

	if expectedType, err = typecheck(op, a, b); err != nil {
		t.Fatalf("Expected Add{} to pass type checking. Err: %v", err)
	}
	if expectedShape, err = shapecheck(op, a, b); err != nil {
		t.Fatalf("Expected Add{} to pass shape checking. Err: %v", err)
	}

	if c, err = op.Do(context.Background(), a, b); err != nil {
		t.Fatalf("Expected Add{} to work correctly. Err: %v", err)
	}
	assert.Equal(t, expectedType, datatypes.TypeOf(c))
	assert.True(t, expectedShape.Eq(c.Shape()))
	correct := []float64{11, 22, 33, 44, 55, 66}
	assert.Equal(t, correct, c.Data())

	// scalar-scalar / PreallocDo

	a = dense.New[float64](tensor.WithShape(), tensor.WithBacking([]float64{1}))
	b = dense.New[float64](tensor.WithShape(), tensor.WithBacking([]float64{2}))
	c = dense.New[float64](tensor.WithShape(), tensor.WithBacking([]float64{-1}))

	if expectedType, err = typecheck(op, a, b); err != nil {
		t.Fatalf("Expected Add{} to pass type checking. Err: %v", err)
	}
	if expectedShape, err = shapecheck(op, a, b); err != nil {
		t.Fatalf("Expected Add{} to pass shape checking. Err: %v", err)
	}

	c, err = op.PreallocDo(context.Background(), c, a, b)
	if err != nil {
		t.Fatalf("Expected Add{}'s Prealloc to work. Err: %v", err)
	}
	correctScalar := []float64{3.0}
	assert.Equal(t, expectedType, datatypes.TypeOf(c))
	assert.Equal(t, correctScalar, c.Data())
	assert.True(t, expectedShape.Eq(c.Shape()))

	// bad cases: fails  typecheck and shapecheck
	a = dense.New[float64](tensor.WithShape(2, 3))
	b = dense.New[float64](tensor.WithShape())
	if expectedType, err = typecheck(op, a, b); err == nil {
		t.Fatalf("Expected Add{} to NOT pass type checking. Got ~(%v %v) =  %v ", datatypes.TypeOf(a), datatypes.TypeOf(b), expectedType)
	}
	if expectedShape, err = shapecheck(op, a, b); err == nil {
		t.Fatalf("Expected Add{} to NOT pass shape checking. Got expectedShape = %v", expectedShape)
	}

}

func TestAddVS__(t *testing.T) {
	op := addVS[float64, *dense.Dense[float64]]{}

	// Do
	var a, b, c *dense.Dense[float64]
	a = dense.New[float64](tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))
	b = dense.New[float64](tensor.WithShape(), tensor.WithBacking([]float64{100}))

	var expectedType hm.Type
	var expectedShape shapes.Shape
	var err error

	if expectedType, err = typecheck(op, a, b); err != nil {
		t.Fatalf("Expected AddVS{} to pass type checking. Err: %v", err)
	}
	if expectedShape, err = shapecheck(op, a, b); err != nil {
		t.Fatalf("Expected AddVS{} to pass shape checking. Err: %v", err)
	}

	if c, err = op.Do(context.Background(), a, b); err != nil {
		t.Fatalf("Expected AddVS{} to work correctly. Err: %v", err)
	}
	assert.Equal(t, expectedType, datatypes.TypeOf(c))
	assert.True(t, expectedShape.Eq(c.Shape()))
	correct := []float64{101, 102, 103, 104, 105, 106}
	assert.Equal(t, correct, c.Data())

	// PreallocDo
	c = dense.New[float64](tensor.WithShape(2, 3), tensor.WithBacking([]float64{-1, -1, -1, -1, -1, -1}))

	c, err = op.PreallocDo(context.Background(), c, a, b)
	if err != nil {
		t.Fatalf("Expected addition operation to work. Err: %v", err)
	}
	assert.Equal(t, expectedType, datatypes.TypeOf(c))
	assert.True(t, expectedShape.Eq(c.Shape()))
	assert.Equal(t, correct, c.Data())

	// bad cases: AddVS{} on tensor-tensor
	b = dense.New[float64](tensor.WithShape(2, 3))
	// we won't type check because the type system is not a dependent type system, thus
	// AddVS : (a → b → a) will always type check without errors
	if expectedShape, err = shapecheck(op, a, b); err == nil {
		t.Fatalf("Expected Add{} to NOT pass shape checking. Got %v ~ (%v, %v) = %v", op.ShapeExpr(), a.Shape(), b.Shape(), expectedShape)
	}
}

func TestAddSV__(t *testing.T) {
	op := addSV[float64, *dense.Dense[float64]]{}

	// Do
	var a, b, c *dense.Dense[float64]
	a = dense.New[float64](tensor.WithShape(), tensor.WithBacking([]float64{100}))
	b = dense.New[float64](tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))

	var expectedType hm.Type
	var expectedShape shapes.Shape
	var err error

	if expectedType, err = typecheck(op, a, b); err != nil {
		t.Fatalf("Expected AddSV{} to pass type checking. Err: %v", err)
	}
	if expectedShape, err = shapecheck(op, a, b); err != nil {
		t.Fatalf("Expected AddSV{} to pass shape checking. Err: %v", err)
	}

	if c, err = op.Do(context.Background(), a, b); err != nil {
		t.Fatalf("Expected AddSV{} to work correctly. Err: %v", err)
	}
	assert.Equal(t, expectedType, datatypes.TypeOf(c))
	assert.True(t, expectedShape.Eq(c.Shape()))
	correct := []float64{101, 102, 103, 104, 105, 106}
	assert.Equal(t, correct, c.Data())

	// PreallocDo
	c = dense.New[float64](tensor.WithShape(2, 3), tensor.WithBacking([]float64{-1, -1, -1, -1, -1, -1}))

	c, err = op.PreallocDo(context.Background(), c, a, b)
	if err != nil {
		t.Fatalf("Expected addition operation to work. Err: %v", err)
	}
	assert.Equal(t, expectedType, datatypes.TypeOf(c))
	assert.True(t, expectedShape.Eq(c.Shape()))
	assert.Equal(t, correct, c.Data())

	// bad cases: AddSV{} on tensor-tensor
	a = dense.New[float64](tensor.WithShape(2, 3))
	// we won't type check because the type system is not a dependent type system, thus
	// AddSV : (a → b → b) will always type check without errors
	if expectedShape, err = shapecheck(op, a, b); err == nil {
		t.Fatalf("Expected Add{} to NOT pass shape checking. Got %v ~ (%v, %v) = %v", op.ShapeExpr(), a.Shape(), b.Shape(), expectedShape)
	}
}

func TestAbs__(t *testing.T) {
	op := Abs[float64, *dense.Dense[float64]]()

	// Do
	var a, b *dense.Dense[float64]
	a = dense.New[float64](tensor.WithShape(2, 3), tensor.WithBacking([]float64{-1, -2, -3, 4, 5, 6}))

	var expectedType hm.Type
	var expectedShape shapes.Shape
	var err error

	if expectedType, err = typecheck(op, a); err != nil {
		t.Fatalf("...")
	}

	if expectedShape, err = shapecheck(op, a); err != nil {
		t.Fatalf("...")
	}

	if b, err = op.Do(context.Background(), a); err != nil {
		t.Fatalf("Expected Abs{} to work correctly. Err: %v", err)
	}
	assert.Equal(t, expectedType, datatypes.TypeOf(b))
	assert.True(t, expectedShape.Eq(b.Shape()))
	correct := []float64{1, 2, 3, 4, 5, 6}
	assert.Equal(t, correct, b.Data())

	// PreallocDo
	b = dense.New[float64](tensor.WithShape(2, 3), tensor.WithBacking([]float64{-100, -100, -100, -100, -100, -100}))
	if b, err = op.PreallocDo(context.Background(), b, a); err != nil {
		t.Fatalf("Expected AddSV{} to work correctly. Err: %v", err)
	}
	assert.Equal(t, expectedType, datatypes.TypeOf(b))
	assert.True(t, expectedShape.Eq(b.Shape()))
	assert.Equal(t, correct, b.Data())
}
