package exprgraph

import (
	"reflect"
	"testing"

	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/iterator"
)

func Test_nodeIDs_Contains(t *testing.T) {
	type args struct {
		a NodeID
	}
	tests := []struct {
		name string
		ns   nodeIDs
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
		Nodes       *iterator.Nodes
		NodesByEdge *iterator.NodesByEdge
	}
	tests := []struct {
		name   string
		fields fields
		want   []*Node
	}{
		{
			"nil length Nodes",
			fields{
				Nodes: iterator.NewNodes(nil),
			},
			nil,
		},
		{
			"nil length Nodes By Weighted edge",
			fields{
				NodesByEdge: iterator.NewNodesByEdge(map[int64]graph.Node{
					0: n0,
					1: n1,
					2: n2,
				}, map[int64]graph.Edge{
					0: &WeightedEdge{
						F: n0,
						T: n1,
						W: 0,
					},
					1: &WeightedEdge{
						F: n0,
						T: n2,
						W: 1,
					},
				}),
			},
			[]*Node{n0, n1},
		},
		{
			"all Nodes",
			fields{
				Nodes: iterator.NewNodes(map[int64]graph.Node{
					0: n0,
					1: n1,
					2: n2,
				}),
			},
			[]*Node{n0, n1, n2},
		},
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			n := &Nodes{
				Nodes:       tt.fields.Nodes,
				NodesByEdge: tt.fields.NodesByEdge,
			}
			got := n.NodeSlice()
			if len(got) != len(tt.want) {
				t.Errorf("Nodes.NodeSlice() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestNodes_Reset(t *testing.T) {
	type fields struct {
		Nodes       *iterator.Nodes
		NodesByEdge *iterator.NodesByEdge
	}
	tests := []struct {
		name   string
		fields fields
	}{
		{
			"simple nodes",
			fields{
				Nodes: iterator.NewNodes(make(map[int64]graph.Node)),
			},
		},
		{
			"simple nodes with weighted edge",
			fields{
				NodesByEdge: iterator.NewNodesByEdge(
					make(map[int64]graph.Node),
					make(map[int64]graph.Edge),
				),
			},
		},
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			n := &Nodes{
				Nodes:       tt.fields.Nodes,
				NodesByEdge: tt.fields.NodesByEdge,
			}
			n.Reset()
		})
	}
}

func TestNewNodes(t *testing.T) {
	n0 := &Node{id: 0}
	n1 := &Node{id: 0}
	n2 := &Node{id: 0}
	n3 := &Node{id: 0}
	nn0 := graph.Node(n0)
	nn1 := graph.Node(n1)
	nn2 := graph.Node(n2)
	nn3 := graph.Node(n3)
	edges := map[int64]graph.WeightedEdge{
		0: &WeightedEdge{
			F: n0,
			T: n1,
			W: 0,
		},
		1: &WeightedEdge{
			F: n0,
			T: n2,
			W: 1,
		},
		2: &WeightedEdge{
			F: n0,
			T: n3,
			W: 2,
		},
	}
	type args struct {
		nodes map[int64]*Node
		edges map[int64]graph.WeightedEdge
	}
	tests := []struct {
		name string
		args args
		want *Nodes
	}{
		{
			"nil edge",
			args{
				nodes: map[int64]*Node{
					0: n0,
					1: n1,
					2: n2,
					3: n3,
				},
			},
			&Nodes{
				Nodes: iterator.NewNodes(map[int64]graph.Node{
					0: nn0,
					1: nn1,
					2: nn2,
					3: nn3,
				}),
			},
		},
		{
			"nil edge",
			args{
				nodes: map[int64]*Node{
					0: n0,
					1: n1,
					2: n2,
					3: n3,
				},
				edges: edges,
			},
			&Nodes{
				NodesByEdge: iterator.NewNodesByWeightedEdge(map[int64]graph.Node{
					0: nn0,
					1: nn1,
					2: nn2,
					3: nn3,
				},
					edges,
				),
			},
		},
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := NewNodes(tt.args.nodes, tt.args.edges)
			// TODO properly check equity
			if !reflect.DeepEqual(got.Len(), tt.want.Len()) {
				t.Errorf("NewNodes() = %v, want %v", got, tt.want)
			}
			if got.Nodes == nil && tt.want.Nodes != nil {
				t.Errorf("NewNodes() = %v, want %v", got, tt.want)
			}
			if got.NodesByEdge == nil && tt.want.NodesByEdge != nil {
				t.Errorf("NewNodes() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestNodes_Next(t *testing.T) {
	type fields struct {
		Nodes       *iterator.Nodes
		NodesByEdge *iterator.NodesByEdge
	}
	tests := []struct {
		name   string
		fields fields
		want   bool
	}{
		{
			"nil nodesbyedge",
			fields{
				Nodes: iterator.NewNodes(nil),
			},
			false,
		},
		{
			"nodesbyedge",
			fields{
				NodesByEdge: iterator.NewNodesByEdge(nil, nil),
			},
			false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			n := &Nodes{
				Nodes:       tt.fields.Nodes,
				NodesByEdge: tt.fields.NodesByEdge,
			}
			if got := n.Next(); got != tt.want {
				t.Errorf("Nodes.Next() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestNodes_Node(t *testing.T) {
	type fields struct {
		Nodes       *iterator.Nodes
		NodesByEdge *iterator.NodesByEdge
	}
	tests := []struct {
		name   string
		fields fields
		want   graph.Node
	}{
		{
			"nil nodesbyedge",
			fields{
				Nodes: iterator.NewNodes(nil),
			},
			nil,
		},
		{
			"nodesbyedge",
			fields{
				NodesByEdge: iterator.NewNodesByEdge(nil, nil),
			},
			nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			n := &Nodes{
				Nodes:       tt.fields.Nodes,
				NodesByEdge: tt.fields.NodesByEdge,
			}
			if got := n.Node(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Nodes.Node() = %v, want %v", got, tt.want)
			}
		})
	}
}
