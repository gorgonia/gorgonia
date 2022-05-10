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

var sumTests = []struct {
	shape        shapes.Shape
	backing      interface{}
	along        shapes.Axes
	correctShape shapes.Shape
	correctData  interface{}
}{
	{
		shapes.Shape{6},
		[]float64{1, 2, 3, 4, 5, 6},
		nil,
		shapes.ScalarShape(),
		21.0,
	},

	{
		shapes.Shape{2, 3},
		[]float64{1, 2, 3, 4, 5, 6},
		nil,
		shapes.ScalarShape(),
		21.0,
	},

	{
		shapes.Shape{2, 3},
		[]float64{1, 2, 3, 4, 5, 6},
		shapes.Axes{1},
		shapes.Shape{2},
		[]float64{6, 15},
	},

	{
		shapes.Shape{2, 3},
		[]float64{1, 2, 3, 4, 5, 6},
		shapes.Axes{0},
		shapes.Shape{3},
		[]float64{5, 7, 9},
	},

	{
		shapes.Shape{2, 3},
		[]float64{1, 2, 3, 4, 5, 6},
		shapes.Axes{0, 1},
		shapes.ScalarShape(),
		21.0,
	},
}

func TestSum(t *testing.T) {
	op := &Sum{}

	// basic test
	assert.Equal(t, 1, op.Arity())

	for _, tc := range sumTests {
		// set up
		var a, b values.Value
		var expectedType hm.Type
		var expectedShape shapes.Shape
		var err error

		a = tensor.New(tensor.WithShape(tc.shape...), tensor.WithBacking(tc.backing))
		op.along = tc.along

		t.Logf("%v \n %v", op.Type(), op.ShapeExpr())

		// type and shape checks
		if expectedType, err = typecheck(op, a); err != nil {
			t.Fatalf("Expected Sum{} to pass type checking. Err: %v", err)
		}
		if expectedShape, err = shapecheck(op, a); err != nil {
			t.Fatalf("Expected Sum{} to pass shape checking. Err: %v", err)
		}

		// actually doing and testing
		if b, err = op.Do(context.Background(), a); err != nil {
			t.Fatalf("Expected Sum{} to work correctly. Err: %v", err)
		}
		assert.Equal(t, expectedType, datatypes.TypeOf(b))
		assert.True(t, expectedShape.Eq(b.Shape()))
		assert.Equal(t, tc.correctData, b.Data())
	}

}
