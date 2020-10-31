package exprgraph

import (
	"reflect"
	"testing"

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
