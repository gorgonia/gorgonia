package exprgraph

import (
	"fmt"
	"testing"

	"gorgonia.org/tensor/dense"
)

/*
func TestCons(t *testing.T) {
	tens := tensor.New(tensor.WithBacking([]int{1, 2, 3}))
	emptyGraph := NewGraph(&tensor.StdEng{})
	type args struct {
		g    *Graph
		name string
		t    tensor.Tensor
	}
	tests := []struct {
		name    string
		args    args
		want    *Node
		wantErr bool
	}{
		{
			"nil graph",
			args{
				nil,
				"test",
				tens,
			},
			&Node{
				Tensor: tens,
				name:   "test",
				id:     0,
				Op:     nil,
			},
			false,
		},
		{
			"empty graph",
			args{
				emptyGraph,
				"test",
				tens,
			},
			&Node{
				id:     0,
				Tensor: tens,
				name:   "test",
				Op:     nil,
			},
			false,
		},
		{
			"node collision in graph",
			args{
				&Graph{
					nodes: map[int64]*Node{
						1: {},
					},
				},
				"test",
				tens,
			},
			nil,
			true,
		},
		{
			"node exists with same name",
			args{
				&Graph{
					nodes: map[int64]*Node{
						1: {
							Tensor: tens,
							id:     1,
							name:   "test",
						},
					},
				},
				"test",
				tens,
			},
			&Node{
				Tensor: tens,
				id:     1,
				name:   "test",
			},
			false,
		},
		{
			"node exists with diffetent name",
			args{
				&Graph{
					nodes: map[int64]*Node{
						1: {
							Tensor: tens,
							id:     1,
							name:   "test",
						},
					},
				},
				"test2",
				tens,
			},
			nil,
			true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Cons(tt.args.g, tt.args.name, tt.args.t)
			if (err != nil) != tt.wantErr {
				t.Errorf("Cons() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Cons() = %v, want %v", got, tt.want)
			}
		})
	}
}
*/

func TestNode_Name(t *testing.T) {
	tests := []struct {
		name string
		n    Node
		want string
	}{
		{"simple Value", newVal(), "test"},
		{"simple Symbolic", newSym(), "test"},
		{"nil Value", (*Value[float64, *dense.Dense[float64]])(nil), "<nil>"},
		{"nil Symbolic", (*Symbolic[float64])(nil), "<nil>"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.n.Name(); got != tt.want {
				t.Errorf("Node.Name() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestNode_Format(t *testing.T) {
	tests := []struct {
		name   string
		n      Node
		format string
		want   string
	}{
		{
			name:   "display name - Value",
			n:      newVal(),
			format: "%s",
			want:   "test",
		},
		{
			name:   "display name - Symbolic",
			n:      newSym(),
			format: "%s",
			want:   "test",
		},
		{
			name:   "display - nil Value",
			n:      (*Value[float64, *dense.Dense[float64]])(nil),
			format: "%v",
			want:   "<nil>",
		},
		{
			name:   "display name - nil Symbolic",
			n:      (*Symbolic[float64])(nil),
			format: "%v",
			want:   "<nil>",
		},
		{
			name:   "display - Value with no Basic",
			n:      newNilVal(),
			format: "%v",
			want:   "node(1337,test,<nil>)",
		},
		{
			name:   "display  Symbolic",
			n:      newSym(),
			format: "%v",
			want:   "node(0,test)",
		},

		{
			name:   "display Value %2.2v",
			n:      newVal(),
			format: "%2.2v",
			want: `⎡1e+02  2e+02⎤
⎣  3.1      4⎦
`,
		},

		{
			name:   "display Value %2.2f",
			n:      newVal(),
			format: "%2.2f",
			want: `⎡100.00  200.00⎤
⎣  3.14    4.00⎦
`,
		},

		{
			name:   "display Value %#v",
			n:      newVal(),
			format: "%#v",
			want: `⎡    100      200⎤
⎣3.14159        4⎦
`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := fmt.Sprintf(tt.format, tt.n); got != tt.want {
				t.Errorf("Node.Format() =\n%v, want\n%v", got, tt.want)
			}

		})
	}
}

func TestNodeID_ID(t *testing.T) {
	tests := []struct {
		name string
		n    NodeID
		want int64
	}{
		{
			"simple",
			NodeID(0),
			0,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.n.ID(); got != tt.want {
				t.Errorf("NodeID.ID() = %v, want %v", got, tt.want)
			}
		})
	}
}
