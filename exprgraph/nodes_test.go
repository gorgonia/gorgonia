package exprgraph

import (
	"math/rand"
	"reflect"
	"testing"
	"testing/quick"

	"gonum.org/v1/gonum/graph"
	"gorgonia.org/dtype"
	"gorgonia.org/shapes"
	"gorgonia.org/tensor"
)

type testIterNodesFields struct {
	ns []*Node
	i  int
}

func (n *IterNodes) Generate(r *rand.Rand, size int) reflect.Value {
	ns := make([]*Node, size)
	i := r.Int()
	nn := &IterNodes{
		ns: ns,
		i:  i,
	}
	return reflect.ValueOf(nn)
}

func Test_nodeIDs_Contains(t *testing.T) {
	type args struct {
		a NodeID
	}
	tests := []struct {
		name string
		ns   NodeIDs
		args args
		want bool
	}{
		{
			"contains",
			[]NodeID{0, 1, 2},
			args{
				a: NodeID(1),
			},
			true,
		},
		{
			"Does not contain",
			[]NodeID{0, 1, 2},
			args{
				a: NodeID(3),
			},
			false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.ns.Contains(tt.args.a); got != tt.want {
				t.Errorf("nodeIDs.Contains() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestNodes_NodeSlice(t *testing.T) {
	n0 := &Node{id: 0}
	n1 := &Node{id: 1}
	n2 := &Node{id: 2}

	tests := []struct {
		name   string
		fields testIterNodesFields
		want   []*Node
	}{
		{
			"nil length Nodes",
			testIterNodesFields{
				ns: nil,
			},
			nil,
		},

		{
			"all Nodes",
			testIterNodesFields{
				ns: []*Node{n0, n1, n2},
				i:  -1,
			},
			[]*Node{n0, n1, n2},
		},
		{
			"all Nodes, used iterator",
			testIterNodesFields{
				ns: []*Node{n0, n1, n2},
				i:  0,
			},
			[]*Node{n1, n2},
		},
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			n := &IterNodes{
				ns: tt.fields.ns,
				i:  tt.fields.i,
			}
			if n.Next() {
				got := n.Nodes()
				if len(got) != len(tt.want) {
					t.Errorf("Nodes.NodeSlice() = %v, want %v", got, tt.want)
				}
			}

		})
	}
}

func TestNodes_Reset(t *testing.T) {
	f := func(n *IterNodes) bool {
		n.Reset()
		return n.i == -1
	}
	if err := quick.Check(f, nil); err != nil {
		t.Error(err)
	}
}

func TestNodes_Next(t *testing.T) {
	tests := []struct {
		name   string
		fields testIterNodesFields
		want   bool
	}{
		{
			"nil nodesbyedge",
			testIterNodesFields{
				ns: nil,
				i:  -1,
			},
			false,
		},
		{
			"exhausted iterator",
			testIterNodesFields{
				ns: make([]*Node, 2),
				i:  2,
			},
			false,
		},
		{
			"usual use",
			testIterNodesFields{
				ns: make([]*Node, 2),
				i:  -1,
			},
			true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			n := &IterNodes{
				ns: tt.fields.ns,
				i:  tt.fields.i,
			}
			if got := n.Next(); got != tt.want {
				t.Errorf("Nodes.Next() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestNodes_Node(t *testing.T) {
	n0 := &Node{id: 0, name: "foo"}
	n1 := &Node{id: 1, name: "bar"}
	tests := []struct {
		name   string
		fields testIterNodesFields
		want   graph.Node
	}{
		{
			"nil ns",
			testIterNodesFields{
				ns: nil,
				i:  -1,
			},
			nil,
		},

		{
			"usual",
			testIterNodesFields{
				ns: []*Node{n0, n1},
				i:  0, // this must be advanced by .Next(), but in this test case it's hard coded
			},
			n0,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			n := &IterNodes{
				ns: tt.fields.ns,
				i:  tt.fields.i,
			}
			if got := n.Node(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Nodes.Node() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestTensorsFromNodeIDs(t *testing.T) {
	g := NewGraph(nil)
	a := NewNode(g, "a", tensor.WithShape(2), tensor.Of(dtype.Float64))
	b, err := NewSymbolic(g, "b", dtype.Float64, shapes.Shape{2})
	if err != nil {
		t.Fatal(err)
	}
	ns := NodeIDs{a.NodeID(), b.NodeID()}
	ts := TensorsFromNodeIDs(g, ns)

	var correct = []Tensor{a, b}
	if !reflect.DeepEqual(ts, correct) {
		t.Errorf("Expected a slice of Tensors %v. Got %v instead", correct, ts)
	}
}
