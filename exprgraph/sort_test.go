package exprgraph

import (
	"reflect"
	"testing"
)

func TestSort(t *testing.T) {
	anode := &Node{id: 0, name: "a"}
	bnode := &Node{id: 1, name: "b"}
	cnode := &Node{id: 2, name: "c"}

	anode2 := &Node{id: 5, name: "a2"}
	bnode2 := &Node{id: 4, name: "b2"}
	cnode2 := &Node{id: 3, name: "c2"}

	tests := []struct {
		name    string
		fields  testGraphFields
		want    []*Node
		willErr bool
	}{
		{
			"nil graph",
			testGraphFields{},
			nil,
			false,
		},
		{
			"simple",
			testGraphFields{
				nodes: map[int64]*Node{
					0: anode,
					1: bnode,
					2: cnode,
				},
				from: map[int64][]int64{
					2: {0, 1},
				},
				to: map[int64][]int64{
					0: {2},
					1: {2},
				},
			},
			[]*Node{cnode, bnode, anode},
			false,
		},

		{
			"simple, but something went wrong with the numbering of nodes",
			testGraphFields{
				nodes: map[int64]*Node{
					5: anode2,
					4: bnode2,
					3: cnode2,
				},
				from: map[int64][]int64{
					3: {5, 4},
				},
				to: map[int64][]int64{
					5: {3},
					4: {3},
				},
			},
			[]*Node{cnode2, anode2, bnode2},
			false,
		},

		{
			"two roots, disjoint subgraphs",
			testGraphFields{
				nodes: map[int64]*Node{
					0: anode,
					1: bnode,
					2: cnode,
					5: anode2,
					4: bnode2,
					3: cnode2,
				},
				from: map[int64][]int64{
					3: {5, 4},
					2: {0, 1},
				},
				to: map[int64][]int64{
					5: {3},
					4: {3},
					0: {2},
					1: {2},
				},
			},
			[]*Node{cnode2, anode2, bnode2, cnode, bnode, anode},
			false,
		},

		{
			"two roots",
			testGraphFields{
				nodes: map[int64]*Node{
					0: anode,
					1: bnode,
					2: cnode,
					3: cnode2,
				},
				from: map[int64][]int64{
					3: {0, 1},
					2: {0, 1},
				},
				to: map[int64][]int64{
					0: {2, 3},
					1: {2, 3},
				},
			},
			[]*Node{cnode2, cnode, bnode, anode},
			false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := graphFromFields(tt.fields)
			if got, err := Sort(g); !reflect.DeepEqual(got, tt.want) || ((err != nil) != tt.willErr) {
				t.Errorf("Sort = %v. Want %v | WillErr %v. Got err %v", got, tt.want, tt.willErr, err)
			}
		})
	}
}
