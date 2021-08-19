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

func TestOuter(t *testing.T) {
	assert := assert.New(t)
	op := Outer{}

	// basic test
	assert.Equal(2, op.Arity())

	/* Do */

	// set up
	var a, b, c values.Value
	var expectedType hm.Type
	var expectedShape shapes.Shape
	var err error

	a = tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))
	b = tensor.New(tensor.WithShape(5, 2, 5), tensor.WithBacking(tensor.Range(tensor.Float64, 1, 51)))

	// type and shape checks
	if expectedType, err = typecheck(op, a, b); err != nil {
		t.Fatalf("Expected %v to pass type checking. Err: %v", op, err)
	}
	if expectedShape, err = shapecheck(op, a, b); err != nil {
		t.Fatalf("Expected %v to pass shape checking. Err: %v", op, err)
	}

	// actually doing and testing
	if c, err = op.Do(context.Background(), a, b); err != nil {
		t.Fatalf("Expected %v to work correctly. Err: %v", op, err)
	}
	assert.Equal(expectedType, datatypes.TypeOf(c))
	assert.True(expectedShape.Eq(c.Shape()))
	correct := []float64{11, 22, 33, 44, 55, 66}
	assert.Equal(correct, c.Data())
}
