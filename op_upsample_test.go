package gorgonia

import (
	"fmt"
	"testing"

	"gorgonia.org/tensor"
)

func TestUpsample(t *testing.T) {
	tt := tensor.New(tensor.Of(tensor.Float64), tensor.WithShape(1, 1, 3, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9}))
	upop := newUpsampleOp(tt.Shape(), 2)
	fmt.Println(tt)
	out, err := upop.Do(tt)
	t.Log(out)
	if err != nil {
		panic(err)
	}

}
