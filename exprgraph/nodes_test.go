package exprgraph

import (
	"testing"
)

func TestNodes_Len(t *testing.T) {
	type fields struct {
		nodes map[int64]*Node
		edges int
		iter  *mapIter
		pos   int
		curr  *Node
	}
	tests := []struct {
		name   string
		fields fields
		want   int
	}{
		{
			"simple",
			fields{
				edges: 2,
				pos:   1,
			},
			1,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			n := &Nodes{
				nodes: tt.fields.nodes,
				edges: tt.fields.edges,
				iter:  tt.fields.iter,
				pos:   tt.fields.pos,
				curr:  tt.fields.curr,
			}
			if got := n.Len(); got != tt.want {
				t.Errorf("Nodes.Len() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestNodes_Next(t *testing.T) {
	type fields struct {
		nodes map[int64]*Node
		edges int
		iter  *mapIter
		pos   int
		curr  *Node
	}
	tests := []struct {
		name   string
		fields fields
		want   bool
	}{
		{
			"no more node",
			fields{
				edges: 2,
				pos:   2,
			},
			false,
		},
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			n := &Nodes{
				nodes: tt.fields.nodes,
				edges: tt.fields.edges,
				iter:  tt.fields.iter,
				pos:   tt.fields.pos,
				curr:  tt.fields.curr,
			}
			if got := n.Next(); got != tt.want {
				t.Errorf("Nodes.Next() = %v, want %v", got, tt.want)
			}
		})
	}
}
