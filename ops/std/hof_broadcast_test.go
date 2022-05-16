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

/*
func TestBroadcast(t *testing.t) {
	eng := engines.NewStd()
	g := exprgraph.NewGraph(eng)
	x := exprgraph.NewNode(g, "x", tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))
	y := exprgraph.NewNode(g, "y", tensor.WithShape(1, 3), tensor.WithBacking([]float64{100, 1000, 10000}))
	z := exprgraph.NewNode(g, "z", tensor.WithShape(), tensor.Of(tensor.Float64))
	a := exprgraph.NewNode(g, "a", tensor.WithShape(), tensor.Of(tensor.Float64))

	op := Mul(x, y)

	const1 := exprgraph.NewNode(g, "const 1", tensor.WithShape(), tensor.WiithBacking([]float64{1}))
	outputs := []*exprgraph.Node{xypz}
	wrt := []*exprgraph.Node{x, y, z}
	gradOutputs := []*exprgraph.Node{const1}

	g2, err := Backpropagate(g, outputs, gradOutputs, wrt)
	if err != nil {
		t.Fatalf("Backprop %v", err)
	}
	_ = g2
}
*/

func TestDevBroadcast(t *testing.T) {
	a := tensor.New(tensor.WithShape(3, 1), tensor.WithBacking([]float64{1, 2, 3}))
	b := tensor.New(tensor.WithShape(1, 3), tensor.WithBacking([]float64{10, 100, 1000}))
	op := Mul(a, b)
	bcop, err := Auto(op, a, b)
	if err != nil {
		t.Fatal(err)
	}
	c, err := bcop.Do(context.TODO(), a, b)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%v", c)
}
