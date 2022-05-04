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

func TestAt(t *testing.T) {
	op := &At{1, 1}

	// basic test
	assert.Equal(t, 1, op.Arity())

	// set up
	var a, b values.Value
	var expectedType hm.Type
	var expectedShape shapes.Shape
	var err error

	a = tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))

	// type and shape checks
	if expectedType, err = typecheck(op, a); err != nil {
		t.Fatalf("Expected At{} to pass type checking. Err: %v", err)
	}
	if expectedShape, err = shapecheck(op, a); err != nil {
		t.Fatalf("Expected At{} to pass shape checking. Err: %v", err)
	}

	// actually doing and testing
	if b, err = op.Do(context.Background(), a); err != nil {
		t.Fatalf("Expected At{} to work correctly. Err: %v", err)
	}
	assert.Equal(t, expectedType, datatypes.TypeOf(b))
	assert.True(t, expectedShape.Eq(b.Shape()))
	correct := 5.0
	assert.Equal(t, correct, b.Data())
}

func TestSlice(t *testing.T) {
	op := &Slice{Slices: shapes.Slices{shapes.S(1, 2), shapes.S(0, 2)}}

	// basic test
	assert.Equal(t, 1, op.Arity())

	// set up
	var a, b values.Value
	var expectedType hm.Type
	var expectedShape shapes.Shape
	var err error

	a = tensor.New(tensor.WithShape(2, 3, 2), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}))

	// type and shape checks
	if expectedType, err = typecheck(op, a); err != nil {
		t.Fatalf("Expected Slice{} to pass type checking. Err: %v", err)
	}
	if expectedShape, err = shapecheck(op, a); err != nil {
		t.Fatalf("Expected Slice{} to pass shape checking. Err: %v", err)
	}

	// actually doing and testing
	if b, err = op.Do(context.Background(), a); err != nil {
		t.Fatalf("Expected Slice{} to work correctly. Err: %v", err)
	}
	assert.Equal(t, expectedType, datatypes.TypeOf(b))
	assert.True(t, expectedShape.Eq(b.Shape()))
	correct := []float64{7, 8, 9, 10}
	assert.Equal(t, correct, b.Data())

	t.Logf("\n%v \n%v \n%v", a, op, b)

	/* PreallocDo */
	// setup - create preallocated result
	b = tensor.New(tensor.WithShape(expectedShape...), tensor.Of(tensor.Float64))
	if b, err = op.PreallocDo(context.Background(), b, a); err != nil {
		t.Fatalf("Expected Slice{}.PreallocDo to work correctly. Err: %v", err)
	}
	assert.Equal(t, expectedType, datatypes.TypeOf(b))
	assert.True(t, expectedShape.Eq(b.Shape()))
	assert.Equal(t, correct, b.Data())
}

func TestSize(t *testing.T) {
	op := Size(0)

	// basic test
	assert.Equal(t, 1, op.Arity())

	// set up
	var a, b values.Value
	var expectedType hm.Type
	var expectedShape shapes.Shape
	var err error

	a = tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))

	// type and shape checks
	if expectedType, err = typecheck(op, a); err != nil {
		t.Fatalf("Expected Size{} to pass type checking. Err: %v", err)
	}
	if expectedShape, err = shapecheck(op, a); err != nil {
		t.Fatalf("Expected Size{} to pass shape checking. Err: %v", err)
	}

	// actually doing and testing
	if b, err = op.Do(context.Background(), a); err != nil {
		t.Fatalf("Expected Size{} to work correctly. Err: %v", err)
	}
	assert.Equal(t, expectedType, datatypes.TypeOf(b))
	assert.True(t, expectedShape.Eq(b.Shape()), "Expected %v. Got %v", expectedShape, b.Shape())
	correct := 2.0
	assert.Equal(t, correct, b.Data())

	// alternative edition: different op, and different backing datatype
	op = Size(1)
	a = tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]int{1, 2, 3, 4, 5, 6}))
	if b, err = op.Do(context.Background(), a); err != nil {
		t.Fatalf("Expected Size{} to work correctly. Err: %v", err)
	}
	assert.True(t, expectedShape.Eq(b.Shape()), "Expected %v. Got %v", expectedShape, b.Shape())
	correctIntData := 3
	assert.Equal(t, correctIntData, b.Data())

	// alternative edition: op on scalar
	op = Size(5) // when the input is scalar, then size of any doesn't really matter does it
	a = tensor.New(tensor.WithShape(), tensor.WithBacking([]bool{true}))
	if b, err = op.Do(context.Background(), a); err != nil {
		t.Fatalf("Expected Size{} to work correctly. Err: %v", err)
	}
	assert.True(t, expectedShape.Eq(b.Shape()), "Expected %v. Got %v", expectedShape, b.Shape())
	correctBoolData := 1
	assert.Equal(t, correctBoolData, b.Data())

	// bad size
	a = tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))
	op = Size(5)
	if b, err = op.Do(context.Background(), a); err == nil {
		t.Fatalf("Expected Size{} to fail when the size is bad. Got value %v", b)
	}
}

func TestReshape(t *testing.T) {
	op := &Reshape{To: shapes.Shape{2, 3}}

	// basic test
	assert.Equal(t, 1, op.Arity())

	// set up
	var a, b values.Value
	var expectedType hm.Type
	var expectedShape shapes.Shape
	var err error

	a = tensor.New(tensor.WithShape(6), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))

	// type and shape checks
	if expectedType, err = typecheck(op, a); err != nil {
		t.Fatalf("Expected Reshape{} to pass type checking. Err: %v", err)
	}
	if expectedShape, err = shapecheck(op, a); err != nil {
		t.Fatalf("Expected Reshape{} to pass shape checking. Err: %v", err)
	}

	// actually doing and testing
	if b, err = op.Do(context.Background(), a); err != nil {
		t.Fatalf("Expected Reshape{} to work correctly. Err: %v", err)
	}
	assert.Equal(t, expectedType, datatypes.TypeOf(b))
	assert.True(t, expectedShape.Eq(b.Shape()))
	correct := []float64{1, 2, 3, 4, 5, 6}
	assert.Equal(t, correct, b.Data())

	// bad shape
	op.To = shapes.Shape{3, 3}
	/*
		// TODO: FUTURE when the op is augmented with shapes library's checking function
		if expectedShape, err = shapecheck(op, a); err == nil {
		    t.Errorf("Expected shapecheck to fail. Got expectedShape: %v", expectedShape)
		}
	*/

	if b, err = op.Do(context.Background(), a); err == nil {
		t.Fatalf("Expected Reshape{} to fail when there is a bad shape. b %v", b)
	}
}
