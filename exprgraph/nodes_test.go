package exprgraph

import (
	"math/rand"
	"reflect"
	"testing"
	"testing/quick"

	"gonum.org/v1/gonum/graph"
)

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
	type fields struct {
		ns []*Node
		i  int
	}
	tests := []struct {
		name   string
		fields fields
		want   []*Node
	}{
		{
			"nil length Nodes",
			fields{
				ns: nil,
			},
			nil,
		},

		{
			"all Nodes",
			fields{
				ns: []*Node{n0, n1, n2},
				i:  -1,
			},
			[]*Node{n0, n1, n2},
		},
		{
			"all Nodes, used iterator",
			fields{
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
		if n.i != -1 {
			return false
		}
		return true
	}
	if err := quick.Check(f, nil); err != nil {
		t.Error(err)
	}
}

func TestNodes_Next(t *testing.T) {
	type fields struct {
		ns []*Node
		i  int
	}
	tests := []struct {
		name   string
		fields fields
		want   bool
	}{
		{
			"nil nodesbyedge",
			fields{
				ns: nil,
				i:  -1,
			},
			false,
		},
		{
			"exhausted iterator",
			fields{
				ns: make([]*Node, 2),
				i:  2,
			},
			false,
		},
		{
			"usual use",
			fields{
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
	type fields struct {
		ns []*Node
		i  int
	}
	tests := []struct {
		name   string
		fields fields
		want   graph.Node
	}{
		{
			"nil ns",
			fields{
				ns: nil,
				i:  -1,
			},
			nil,
		},

		{
			"usual",
			fields{
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
