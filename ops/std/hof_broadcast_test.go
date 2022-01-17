package stdops

import (
	"context"
	"testing"

	"gorgonia.org/gorgonia/ops"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/tensor"
)

func TestAutoBroadcast(t *testing.T) {
	testcases := []struct {
		name    string
		op      func(a, b ops.Operand) ops.PreallocOp
		a, b    values.Value
		correct values.Value
		willErr bool
	}{

		{
			"(2,3) × (2, 1)",
			Mul,
			tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6})),
			tensor.New(tensor.WithShape(2, 1), tensor.WithBacking([]float64{100, 100})),
			tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{100, 200, 300, 400, 500, 600})),
			false,
		},

		{
			"(2,3) × (1, 3)",
			Mul,
			tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6})),
			tensor.New(tensor.WithShape(1, 3), tensor.WithBacking([]float64{100, 200, 100})),
			tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{100, 400, 300, 400, 1000, 600})),
			false,
		},

		{
			"(1,3) × (2, 3)",
			Mul,
			tensor.New(tensor.WithShape(1, 3), tensor.WithBacking([]float64{100, 200, 100})),
			tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6})),
			tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{100, 400, 300, 400, 1000, 600})),
			false,
		},
		{
			"(2, 1) × (2, 3)",
			Mul,
			tensor.New(tensor.WithShape(2, 1), tensor.WithBacking([]float64{100, 200})),
			tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6})),
			tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{100, 200, 300, 800, 1000, 1200})),
			false,
		},
		{
			"(1, 2) × (2, 1)",
			Mul,
			tensor.New(tensor.WithShape(1, 2), tensor.WithBacking([]float64{100, 200})),
			tensor.New(tensor.WithShape(2, 1), tensor.WithBacking([]float64{1, 2})),
			tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{100, 200, 200, 400})),
			false,
		},
		{
			"(2, 1) × (1, 3)",
			Mul,
			tensor.New(tensor.WithShape(2, 1), tensor.WithBacking([]float64{100, 200})),
			tensor.New(tensor.WithShape(1, 3), tensor.WithBacking([]float64{1, 2, 3})),
			tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{100, 200, 300, 200, 400, 600})),
			false,
		},
		{
			"(2, 1, 4) + (2, 3, 4)",
			Add,
			tensor.New(tensor.WithShape(2, 1, 4), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6, 7, 8})),
			tensor.New(tensor.WithShape(2, 3, 4), tensor.WithBacking([]float64{
				10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120,
				10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120,
			})),
			tensor.New(tensor.WithShape(2, 3, 4), tensor.WithBacking([]float64{
				11, 22, 33, 44,
				51, 62, 73, 84,
				91, 102, 113, 124,

				15, 26, 37, 48,
				55, 66, 77, 88,
				95, 106, 117, 128,
			})),
			false,
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			//t.Parallel()
			op := tc.op(tc.a, tc.b)
			broadcastOp, err := Auto(op, tc.a, tc.b)
			if err != nil {
				t.Fatal(err)
			}
			c, err := broadcastOp.Do(context.TODO(), tc.a, tc.b)
			if err != nil {
				t.Fatal(err)
			}
			if !c.Eq(tc.correct) {
				t.Errorf("Expected c to be \n%v\nGot\n%v", tc.correct, c)
			}
		})
	}

}
