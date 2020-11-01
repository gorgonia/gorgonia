package exprgraph

import (
	"reflect"
	"testing"

	"gorgonia.org/gorgonia"
	"gorgonia.org/gorgonia/ops"
	"gorgonia.org/tensor"
)

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
				id:     1,
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

func TestNode_GetName(t *testing.T) {
	type fields struct {
		Tensor gorgonia.Tensor
		id     int64
		name   string
		Op     ops.Op
	}
	tests := []struct {
		name   string
		fields fields
		want   string
	}{
		{
			"ok",
			fields{
				name: "test",
			},
			"test",
		},
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			n := &Node{
				Tensor: tt.fields.Tensor,
				id:     tt.fields.id,
				name:   tt.fields.name,
				Op:     tt.fields.Op,
			}
			if got := n.GetName(); got != tt.want {
				t.Errorf("Node.GetName() = %v, want %v", got, tt.want)
			}
		})
	}
}
