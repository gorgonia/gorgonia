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
				NodesByEdge: iterator.NewNodesByEdge(nil, nil),
			},
			nil,
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
			if got := n.NodeSlice(); !reflect.DeepEqual(got, tt.want) {
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
