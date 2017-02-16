// +build cuda

package gorgonia

import (
	"testing"

	"github.com/alecthomas/assert"
	"github.com/chewxy/gorgonia/tensor"
)

func TestCUDACube(t *testing.T) {
	assert := assert.New(t)
	xT := tensor.New(tensor.Of(tensor.Float32), tensor.WithBacking(tensor.Range(Float32, 0, 32)), tensor.WithShape(8, 4))

	g := NewGraph(WithGraphName("Test"))
	x := NewMatrix(g, tensor.Float32, WithName("x"), WithShape(8, 4), WithValue(xT))
	x3 := Must(Cube(x))

	prog, locMap, err := Compile(g)
	t.Logf("Prog: \n%v", prog)
	if err != nil {
		t.Fatal(err)
	}
	m := NewTapeMachine(prog, locMap)
	if err = m.LoadCUDAFunc("cube32", cube32PTX); err != nil {
		t.Fatal(err)
	}
	if err = m.RunAll(); err != nil {
		t.Error(err)
	}
	correct := []float32{0, 1, 8, 27, 64, 125, 216, 343, 512, 729, 1000, 1331, 1728, 2197, 2744, 3375, 4096, 4913, 5832, 6859, 8000, 9261, 10648, 12167, 13824, 15625, 17576, 19683, 21952, 24389, 27000, 29791}
	assert.Equal(correct, x3.Value().Data())

	correct = tensor.Range(tensor.Float32, 0, 32).([]float32)
	assert.Equal(correct, x.Value().Data())
}

func TestCUDAUnary(t *testing.T) {
	g := NewGraph(WithGraphName("CUDATEST"))
	x := NewVector(g, tensor.Float32, WithName("x"), WithShape(5))
	y := NewVector(g, tensor.Float32, WithName("y"), WithShape(5))
	x2 := Must(Square(x))
	x2py := Must(Add(x2, y))
	negx2py := Must(Neg(x2py))
	WithName("negx2py")(negx2py)
	// Must(Add(x2py, negx2py))

	prog, _, _ := Compile(g)
	t.Logf("%v", prog)

}
