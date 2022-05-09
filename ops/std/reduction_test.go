package stdops

import (
	"testing"

	"github.com/chewxy/hm"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/shapes"
	"gorgonia.org/tensor"
)

func TestSum(t *testing.T) {
	op := &Sum{}

	// basic test
	assert.Equal(t, 1, op.Arity())

	// set up
	var a, b values.Value
	var expectedType hm.Type
	var expectedShape shapes.Shape
	var err error

	a = tensor.New(tensor.WithShape(6), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))
	t.Logf("%v", op.ShapeExpr())
	// type and shape checks
	if expectedType, err = typecheck(op, a); err != nil {
		t.Fatalf("Expected Sum{} to pass type checking. Err: %v", err)
	}
	if expectedShape, err = shapecheck(op, a); err != nil {
		t.Fatalf("Expected Sum{} to pass shape checking. Err: %v", err)
	}
	/*
		// actually doing and testing
		if b, err = op.Do(context.Background(), a); err != nil {
			t.Fatalf("Expected Sum{} to work correctly. Err: %v", err)
		}
		assert.Equal(t, expectedType, datatypes.TypeOf(b))
		assert.True(t, expectedShape.Eq(b.Shape()))
		correct := []float64{1, 2, 3, 4, 5, 6}
		assert.Equal(t, correct, b.Data())
	*/
	_ = b

	t.Logf("expected Type %v", expectedType)
	t.Logf("expected shape %v", expectedShape)
}
