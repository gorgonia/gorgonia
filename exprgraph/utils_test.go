package exprgraph

import (
	"reflect"
	"testing"

	"gorgonia.org/gorgonia/values/dual"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/dense"
)

func TestT2B(t *testing.T) {
	sampleTensor := dense.New[float32](tensor.WithShape(1))
	sampleDV := dual.New[float32, *dense.Dense[float32]](dense.New[float32](tensor.WithShape(1)))

	tests := []struct {
		name string
		n    Node
		want tensor.Basic[float32]
	}{
		{
			"*Node with dual value",
			&Value[float32, *dense.Dense[float32]]{
				Basic: sampleDV,
			},
			sampleDV,
		},
		{
			"*Node",
			&Value[float32, *dense.Dense[float32]]{
				Basic: sampleTensor,
			},
			sampleTensor,
		},
		{"Value without tensor", newNilVal(), nil},
		{"nil Node", nil, nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := T2B[float32](tt.n); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("T2T() = %v, want %v", got, tt.want)
			}
		})
	}
}
