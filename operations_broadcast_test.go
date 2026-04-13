package gorgonia

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

type broadcastOpTest struct {
	name string
	a    Value
	b    Value

	// broadcast axes
	left, right []byte

	// results
	ab  Value
	err bool
}

var broadcastAddTests = []broadcastOpTest{
	{
		name:  "vec-mat",
		a:     tensor.New(tensor.WithShape(2), tensor.WithBacking([]float64{100, 200})),
		b:     tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{1, 2, 3, 4})),
		left:  []byte{1},
		right: nil,
		ab:    tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{101, 102, 203, 204})),
		err:   false,
	},

	{
		name:  "mat-vec",
		a:     tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{1, 2, 3, 4})),
		b:     tensor.New(tensor.WithShape(2), tensor.WithBacking([]float64{100, 200})),
		left:  nil,
		right: []byte{1},
		ab:    tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{101, 102, 203, 204})),
		err:   false,
	},
	{
		name:  "rowvec-mat",
		a:     tensor.New(tensor.WithShape(2, 1), tensor.WithBacking([]float64{100, 200})),
		b:     tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{1, 2, 3, 4})),
		left:  []byte{1},
		right: nil,
		ab:    tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{101, 102, 203, 204})),
		err:   false,
	},
	{
		name:  "mat-rowvec",
		a:     tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{1, 2, 3, 4})),
		b:     tensor.New(tensor.WithShape(2, 1), tensor.WithBacking([]float64{100, 200})),
		left:  nil,
		right: []byte{1},
		ab:    tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{101, 102, 203, 204})),
		err:   false,
	},
	{
		name:  "colvec-mat",
		a:     tensor.New(tensor.WithShape(1, 2), tensor.WithBacking([]float64{100, 200})),
		b:     tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{1, 2, 3, 4})),
		left:  []byte{0},
		right: nil,
		ab:    tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{101, 202, 103, 204})),
		err:   false,
	},
	{
		name:  "mat-colvec",
		a:     tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{1, 2, 3, 4})),
		b:     tensor.New(tensor.WithShape(1, 2), tensor.WithBacking([]float64{100, 200})),
		left:  nil,
		right: []byte{0},
		ab:    tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{101, 202, 103, 204})),
		err:   false,
	},
	/* // SKIPPED UNTIL WE CAN FIX BROADCAST SEMANTICS
	{name: "3col-3tensor",
		a:     tensor.New(tensor.WithShape(1, 1, 2), tensor.WithBacking([]float64{100, 200})),
		b:     tensor.New(tensor.WithShape(2, 2, 2), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6, 7, 8})),
		left:  []byte{0, 1},
		right: nil,
		ab:    tensor.New(tensor.WithShape(2, 2, 2), tensor.WithBacking([]float64{101, 202, 103, 204, 105, 206, 107, 208})),
		err:   false,
	},
	{name: "3vec-3tensor",
		a:     tensor.New(tensor.WithShape(2, 1, 1), tensor.WithBacking([]float64{100, 200})),
		b:     tensor.New(tensor.WithShape(2, 2, 2), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6, 7, 8})),
		left:  []byte{1, 2},
		right: nil,
		ab:    tensor.New(tensor.WithShape(2, 2, 2), tensor.WithBacking([]float64{101, 102, 103, 104, 205, 206, 207, 208})),
		err:   false,
	},
	{name: "colmat-3tensor",
		a:     tensor.New(tensor.WithShape(1, 2, 2), tensor.WithBacking([]float64{100, 200, 300, 400})),
		b:     tensor.New(tensor.WithShape(2, 2, 2), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6, 7, 8})),
		left:  []byte{0},
		right: nil,
		ab:    tensor.New(tensor.WithShape(2, 2, 2), tensor.WithBacking([]float64{101, 202, 303, 404, 105, 206, 307, 408})),
		err:   false,
	},
	{name: "3tensor-colmat",
		a:     tensor.New(tensor.WithShape(2, 2, 2), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6, 7, 8})),
		b:     tensor.New(tensor.WithShape(1, 2, 2), tensor.WithBacking([]float64{100, 200, 300, 400})),
		left:  nil,
		right: []byte{0},
		ab:    tensor.New(tensor.WithShape(2, 2, 2), tensor.WithBacking([]float64{101, 202, 303, 404, 105, 206, 307, 408})),
		err:   false,
	},
	{name: "rowmat-3tensor",
		a:     tensor.New(tensor.WithShape(2, 2, 1), tensor.WithBacking([]float64{100, 200, 300, 400})),
		b:     tensor.New(tensor.WithShape(2, 2, 2), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6, 7, 8})),
		left:  []byte{2},
		right: nil,
		ab:    tensor.New(tensor.WithShape(2, 2, 2), tensor.WithBacking([]float64{101, 102, 203, 204, 305, 306, 407, 408})),
		err:   false,
	},
	{name: "3tensor-rowmat",
		a:     tensor.New(tensor.WithShape(2, 2, 2), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6, 7, 8})),
		b:     tensor.New(tensor.WithShape(2, 2, 1), tensor.WithBacking([]float64{100, 200, 300, 400})),
		left:  nil,
		right: []byte{2},
		ab:    tensor.New(tensor.WithShape(2, 2, 2), tensor.WithBacking([]float64{101, 102, 203, 204, 305, 306, 407, 408})),
		err:   false,
	},
	{name: "vec-3tensor",
		a:     tensor.New(tensor.WithShape(2), tensor.WithBacking([]float64{100, 200})),
		b:     tensor.New(tensor.WithShape(2, 2, 2), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6, 7, 8})),
		left:  []byte{1, 2},
		right: nil,
		ab:    tensor.New(tensor.WithShape(2, 2, 2), tensor.WithBacking([]float64{101, 202, 103, 204, 105, 206, 107, 208})),
		err:   false,
	},
	*/
	// TODO (these would give coverage to all broadcast applications)
	// 	vec-3tensor
	// 	3tensor-vec
	// 	mat-3tensor
	// 	3-tensor-mat
	// and their corresponding errors

	// WILL ERR
	// {name: "vec-mat- wrong left pattern axis",
	// 	a:     tensor.New(tensor.WithShape(2), tensor.WithBacking([]float64{100, 200})),
	// 	b:     tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{1, 2, 3, 4})),
	// 	left:  []byte{0},
	// 	right: nil,
	// 	ab:    tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{101, 102, 203, 204})),
	// 	err:   true,
	// },
	{
		name:  "rowvec-mat: wrong axis",
		a:     tensor.New(tensor.WithShape(2, 1), tensor.WithBacking([]float64{100, 200})),
		b:     tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{1, 2, 3, 4})),
		left:  []byte{2},
		right: nil,
		ab:    tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{101, 102, 203, 204})),
		err:   true,
	},

	{
		name:  "impossible mat-mat",
		a:     tensor.New(tensor.WithShape(2, 4), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6, 7, 8})),
		b:     tensor.New(tensor.WithShape(1, 2), tensor.WithBacking([]float64{100, 200})),
		left:  nil,
		right: []byte{0, 1},
		ab:    tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{101, 102, 203, 204})),
		err:   true,
	},
}

func TestBroadcastAdd(t *testing.T) {
	assert := assert.New(t)
	for i, bat := range broadcastAddTests {
		// if bat.name != "impossible mat-mat" {
		//		continue
		//	}
		g := NewGraph()
		a := NodeFromAny(g, bat.a, WithName("a"))
		b := NodeFromAny(g, bat.b, WithName("b"))
		c, err := BroadcastAdd(a, b, bat.left, bat.right)
		if checkErr(t, bat.err, err, bat.name, i) {
			continue
		}
		machine := NewTapeMachine(g)

		if err = machine.RunAll(); err != nil {
			t.Errorf("Test %v(%d): %v", bat.name, i, err)
		}
		assert.Equal(bat.ab.Data(), c.Value().Data(), "Test %v(%v)", bat.name, i)
		machine.Close()
	}
}

var broadcastMulTests = []broadcastOpTest{
	{
		name:  "vec-mat",
		a:     tensor.New(tensor.WithShape(2), tensor.WithBacking([]float64{10, 20})),
		b:     tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{1, 2, 3, 4})),
		left:  []byte{1},
		right: nil,
		ab:    tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{10, 20, 60, 80})),
		err:   false,
	},

	{
		name:  "mat-vec",
		a:     tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{1, 2, 3, 4})),
		b:     tensor.New(tensor.WithShape(2), tensor.WithBacking([]float64{10, 20})),
		left:  nil,
		right: []byte{1},
		ab:    tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{10, 20, 60, 80})),
		err:   false,
	},
	{
		name:  "rowvec-mat",
		a:     tensor.New(tensor.WithShape(2, 1), tensor.WithBacking([]float64{10, 20})),
		b:     tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{1, 2, 3, 4})),
		left:  []byte{1},
		right: nil,
		ab:    tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{10, 20, 60, 80})),
		err:   false,
	},
	{
		name:  "mat-rowvec",
		a:     tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{1, 2, 3, 4})),
		b:     tensor.New(tensor.WithShape(2, 1), tensor.WithBacking([]float64{10, 20})),
		left:  nil,
		right: []byte{1},
		ab:    tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{10, 20, 60, 80})),
		err:   false,
	},
	{
		name:  "colvec-mat",
		a:     tensor.New(tensor.WithShape(1, 2), tensor.WithBacking([]float64{10, 20})),
		b:     tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{1, 2, 3, 4})),
		left:  []byte{0},
		right: nil,
		ab:    tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{10, 40, 30, 80})),
		err:   false,
	},
	{
		name:  "mat-colvec",
		a:     tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{1, 2, 3, 4})),
		b:     tensor.New(tensor.WithShape(1, 2), tensor.WithBacking([]float64{10, 20})),
		left:  nil,
		right: []byte{0},
		ab:    tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{10, 40, 30, 80})),
		err:   false,
	},
	{
		name:  "vec-3tensor",
		a:     tensor.New(tensor.WithShape(2), tensor.WithBacking([]float64{10, 20})),
		b:     tensor.New(tensor.WithShape(2, 2, 2), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6, 7, 8})),
		left:  []byte{0, 1},
		right: nil,
		ab:    tensor.New(tensor.WithShape(2, 2, 2), tensor.WithBacking([]float64{10, 40, 30, 80, 50, 120, 70, 160})),
		err:   false,
	},
	{
		name:  "3tensor-vec",
		a:     tensor.New(tensor.WithShape(2, 2, 2), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6, 7, 8})),
		b:     tensor.New(tensor.WithShape(2), tensor.WithBacking([]float64{10, 20})),
		left:  nil,
		right: []byte{0, 1},
		ab:    tensor.New(tensor.WithShape(2, 2, 2), tensor.WithBacking([]float64{10, 40, 30, 80, 50, 120, 70, 160})),
		err:   false,
	},
	{
		name:  "mat-3tensor",
		a:     tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{10, 20, 30, 40})),
		b:     tensor.New(tensor.WithShape(2, 2, 2), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6, 7, 8})),
		left:  []byte{0},
		right: nil,
		ab:    tensor.New(tensor.WithShape(2, 2, 2), tensor.WithBacking([]float64{10, 40, 90, 160, 50, 120, 210, 320})),
		err:   false,
	},
	{
		name:  "3-tensor-mat",
		a:     tensor.New(tensor.WithShape(2, 2, 2), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6, 7, 8})),
		b:     tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{10, 20, 30, 40})),
		left:  nil,
		right: []byte{0},
		ab:    tensor.New(tensor.WithShape(2, 2, 2), tensor.WithBacking([]float64{10, 40, 90, 160, 50, 120, 210, 320})),
		err:   false,
	},
	{
		name:  "mat-3tensor: missing pattern",
		a:     tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{10, 20, 30, 40})),
		b:     tensor.New(tensor.WithShape(2, 2, 2), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6, 7, 8})),
		left:  nil,
		right: nil,
		err:   true,
	},
	{
		name:  "3-tensor-mat: missing pattern",
		a:     tensor.New(tensor.WithShape(2, 2, 2), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6, 7, 8})),
		b:     tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{10, 20, 30, 40})),
		left:  nil,
		right: nil,
		err:   true,
	},

	// WILL ERR
	// {name: "vec-mat- wrong left pattern axis",
	// 	a:     tensor.New(tensor.WithShape(2), tensor.WithBacking([]float64{10, 20})),
	// 	b:     tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{1, 2, 3, 4})),
	// 	left:  []byte{0},
	// 	right: nil,
	// 	err:   true,
	// },
	{
		name:  "rowvec-mat: wrong axis",
		a:     tensor.New(tensor.WithShape(2, 1), tensor.WithBacking([]float64{10, 20})),
		b:     tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{1, 2, 3, 4})),
		left:  []byte{2},
		right: nil,
		err:   true,
	},

	{
		name:  "impossible mat-mat",
		a:     tensor.New(tensor.WithShape(2, 4), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6, 7, 8})),
		b:     tensor.New(tensor.WithShape(1, 2), tensor.WithBacking([]float64{10, 20})),
		left:  nil,
		right: []byte{0, 1},
		err:   true,
	},
}

func TestBroadcastHadamardProd(t *testing.T) {
	assert := assert.New(t)
	for i, bat := range broadcastMulTests {
		g := NewGraph()
		a := NodeFromAny(g, bat.a, WithName("a"))
		b := NodeFromAny(g, bat.b, WithName("b"))
		c, err := BroadcastHadamardProd(a, b, bat.left, bat.right)
		if checkErr(t, bat.err, err, bat.name, i) {
			continue
		}
		machine := NewTapeMachine(g)

		if err = machine.RunAll(); err != nil {
			t.Errorf("Test %v(%d): %v", bat.name, i, err)
		}
		assert.Equal(bat.ab.Data(), c.Value().Data(), "Test %v(%v)", bat.name, i)
		machine.Close()
	}
}

func runBroadcastBinaryOp(
	t *testing.T,
	name string,
	aVal, bVal Value,
	left, right []byte,
	wantErr bool,
	want Value,
	op func(a, b *Node, leftPattern, rightPattern []byte) (*Node, error),
) {
	t.Helper()
	g := NewGraph()
	a := NodeFromAny(g, aVal, WithName("a"))
	b := NodeFromAny(g, bVal, WithName("b"))
	c, err := op(a, b, left, right)
	if checkErr(t, wantErr, err, name, name) {
		return
	}
	machine := NewTapeMachine(g)
	defer machine.Close()
	if err = machine.RunAll(); err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	assert.Equal(t, want.Data(), c.Value().Data(), name)
}

func TestBroadcastOtherArithmeticOps(t *testing.T) {
	a := tensor.New(tensor.WithShape(2), tensor.WithBacking([]float64{10, 20}))
	b := tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{1, 2, 3, 4}))
	badA := tensor.New(tensor.WithShape(2, 1), tensor.WithBacking([]float64{10, 20}))

	runBroadcastBinaryOp(t, "sub vec-mat", a, b, []byte{1}, nil, false,
		tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{9, 8, 17, 16})),
		BroadcastSub)
	runBroadcastBinaryOp(t, "div vec-mat", a, b, []byte{1}, nil, false,
		tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{10, 5, 6.666666666666667, 5})),
		BroadcastHadamardDiv)
	runBroadcastBinaryOp(t, "pow vec-mat", tensor.New(tensor.WithShape(2), tensor.WithBacking([]float64{2, 3})),
		tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{1, 2, 3, 2})),
		[]byte{1}, nil, false,
		tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{2, 4, 27, 9})),
		BroadcastPow)

	runBroadcastBinaryOp(t, "sub vec-mat missing pattern", a, b, nil, nil, true, nil, BroadcastSub)
	runBroadcastBinaryOp(t, "div vec-mat missing pattern", a, b, nil, nil, true, nil, BroadcastHadamardDiv)
	runBroadcastBinaryOp(t, "pow vec-mat missing pattern", a, b, nil, nil, true, nil, BroadcastPow)

	runBroadcastBinaryOp(t, "sub invalid axis", badA, b, []byte{2}, nil, true, nil, BroadcastSub)
	runBroadcastBinaryOp(t, "div invalid axis", badA, b, []byte{2}, nil, true, nil, BroadcastHadamardDiv)
	runBroadcastBinaryOp(t, "pow invalid axis", badA, b, []byte{2}, nil, true, nil, BroadcastPow)
}

func TestBroadcastComparisonOps(t *testing.T) {
	type cmpCase struct {
		name string
		op   func(a, b *Node, retSame bool, leftPattern, rightPattern []byte) (*Node, error)
		want Value
	}
	aVal := tensor.New(tensor.WithShape(2), tensor.WithBacking([]float64{2, 1}))
	bVal := tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{1, 2, 3, 0}))
	cases := []cmpCase{
		{
			name: "lt",
			op:   BroadcastLt,
			want: tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{0, 0, 1, 0})),
		},
		{
			name: "gt",
			op:   BroadcastGt,
			want: tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{1, 0, 0, 1})),
		},
		{
			name: "lte",
			op:   BroadcastLte,
			want: tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{0, 1, 1, 0})),
		},
		{
			name: "gte",
			op:   BroadcastGte,
			want: tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{1, 1, 0, 1})),
		},
		{
			name: "eq",
			op:   BroadcastEq,
			want: tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{0, 1, 0, 0})),
		},
		{
			name: "ne",
			op:   BroadcastNe,
			want: tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{1, 0, 1, 1})),
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			g := NewGraph()
			a := NodeFromAny(g, aVal, WithName("a"))
			b := NodeFromAny(g, bVal, WithName("b"))
			c, err := tc.op(a, b, true, []byte{1}, nil)
			if err != nil {
				t.Fatalf("%s: %v", tc.name, err)
			}
			machine := NewTapeMachine(g)
			defer machine.Close()
			if err = machine.RunAll(); err != nil {
				t.Fatalf("%s: %v", tc.name, err)
			}
			assert.Equal(t, tc.want.Data(), c.Value().Data(), tc.name)
		})

		t.Run(tc.name+" invalid axis", func(t *testing.T) {
			g := NewGraph()
			a := NodeFromAny(g, tensor.New(tensor.WithShape(2, 1), tensor.WithBacking([]float64{2, 1})), WithName("a"))
			b := NodeFromAny(g, bVal, WithName("b"))
			_, err := tc.op(a, b, true, []byte{2}, nil)
			if err == nil {
				t.Fatalf("%s invalid axis: expected error, got nil", tc.name)
			}
		})
	}
}

// Broadcasts with nils in both left and right patterns will yield the original inputs.
func ExampleBroadcast_nils() {
	g := NewGraph()
	x := NewMatrix(g, Float64, WithShape(2, 3), WithName("x"))
	y := NewMatrix(g, Float64, WithShape(2, 3), WithName("y"))
	a, b, err := Broadcast(x, y, NewBroadcastPattern(nil, nil))
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	fmt.Printf("a == x %t; b == y %t", a == x, b == y)
	//  Output:
	// a == x true; b == y true
}
