package exprgraph

/*
import (
	"fmt"
	"reflect"
	"testing"

	"gorgonia.org/tensor"
)

func Test_NewSymbolic(t *testing.T) {
	g := NewGraph(&tensor.StdEng{})
	type args struct {
		g     *Graph
		dt    tensor.Dtype
		shape tensor.Shape
		name  string
	}
	tests := []struct {
		name string
		args args
		want *header
	}{
		{
			"simple",
			args{
				g:     g,
				dt:    tensor.Float32,
				shape: tensor.Shape{1, 2, 3},
				name:  "simple",
			},
			&header{
				AP: tensor.MakeAP(tensor.Shape{1, 2, 3}, tensor.CalcStrides(tensor.Shape{1, 2, 3}), 0, 0),
				dt: tensor.Float32,
				g:  g,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := newHeader(tt.args.g, tt.args.dt, tt.args.shape)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("newSymbolic() = %v, want %v", got, tt.want)
			}
			if got.DataSize() != 0 {
				t.Error("Datasize fail")
			}
			if got.Dtype() != tt.args.dt {
				t.Error("Dtype fail")
			}
			if got.Engine() != tt.args.g {
				t.Error("Engine")
			}
			if got.IsNativelyAccessible() != false {
				t.Error("IsNativelyAccessible")
			}
			if got.IsManuallyManaged() != true {
				t.Error("IsManuallyManaged")
			}
			if got.MemSize() != 0 {
				t.Error("MemSize")
			}
			if got.Uintptr() != 0 {
				t.Error("Uinptr")
			}
			if got.Pointer() != nil {
				t.Error("Pointer")
			}
			if got.ScalarValue() != nil {
				t.Error("ScalarValue")
			}
			if got.Data() != nil {
				t.Error("Data")
			}
			if out := fmt.Sprintf("%v", got); out != "" {
				t.Errorf("Format %q x", got)
			}
		})
	}
}
*/
