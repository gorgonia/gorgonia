package exprgraph

import (
	"reflect"
	"testing"

	"gorgonia.org/gorgonia/values/dual"
	"gorgonia.org/tensor"
)

func TestT2T(t *testing.T) {
	sampleTensor := tensor.NewDense(tensor.Float32, []int{1})
	sampleDV := dual.New(tensor.NewDense(tensor.Float32, []int{1}))
	type args struct {
		a Tensor
	}
	tests := []struct {
		name string
		args args
		want tensor.Tensor
	}{
		{
			"*Node with dual value",
			args{
				&Node{
					Tensor: sampleDV,
				},
			},
			sampleDV,
		},
		{
			"*Node",
			args{
				&Node{
					Tensor: sampleTensor,
				},
			},
			sampleTensor,
		},
		{
			"*Node without tensor",
			args{
				&Node{},
			},
			nil,
		},
		{
			"nil value",
			args{
				nil,
			},
			nil,
		},
		{
			"tensor value",
			args{
				sampleTensor,
			},
			sampleTensor,
		},
		{
			"dual value",
			args{
				sampleDV,
			},
			sampleDV,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := T2T(tt.args.a); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("T2T() = %v, want %v", got, tt.want)
			}
		})
	}
}
