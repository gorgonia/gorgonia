package gorgonia

import (
	"testing"

	"gorgonia.org/tensor"

	"github.com/pkg/errors"
)

func concat(xS, yS tensor.Shape) (Value, error) {

	g := NewGraph()
	x := NewTensor(g, Float32, xS.Dims(), WithName("x"), WithShape(xS...), WithInit(RangedFrom(0)))
	y := NewTensor(g, Float32, yS.Dims(), WithName("y"), WithShape(yS...), WithInit(RangedFrom(0)))

	z, err := Concat(1, x, y)
	if err != nil {
		return nil, errors.Wrap(err, "Concat")
	}

	m := NewLispMachine(g, ExecuteFwdOnly())
	if err = m.RunAll(); err != nil {
		return nil, errors.Wrap(err, "run Concat")
	}

	return z.Value(), nil
	//fmt.Print("x: \n", x.Value())
	//fmt.Print("\ny: \n", y.Value())
	//fmt.Print("\nConcat of x,y: \n", z.Value())
}

func runMul(xS, yS tensor.Shape) (Value, error) {

	g := NewGraph()
	x := NewMatrix(g, Float32, WithName("x"), WithShape(xS...), WithInit(RangedFrom(0)))
	y := NewMatrix(g, Float32, WithName("y"), WithShape(yS...), WithInit(RangedFrom(0)))

	z := Must(Mul(x, y))
	cost := Must(Sum(z))

	_, err := Grad(cost, x, y)
	if err != nil {
		return nil, errors.Wrap(err, "Grad")
	}

	m := NewTapeMachine(g)
	if err = m.RunAll(); err != nil {
		return nil, errors.Wrap(err, "run mul")
	}
	return cost.Value(), nil
}

func TestConcat_issue341(t *testing.T) {

	//success
	t.Run("2,2,2+ 2,3,2", func(t *testing.T) {
		_, err := concat(tensor.Shape{2, 2, 2}, tensor.Shape{2, 3, 2})
		if err != nil {
			t.Fatal(err)
		}
	})

	//run Concat: RunAll: Failed to bindVar: Failed to carry op.Do(): Failed to perform Concat:
	//Unable to assignArray in denseConcat: BroadcastStrides failed: Dimension mismatch. Expected 2, got 3
	t.Run("2,2,2+ 2,2,2", func(t *testing.T) {
		_, err := concat(tensor.Shape{2, 2, 2}, tensor.Shape{2, 1, 2})
		if err != nil {
			t.Fatal(err)
		}
	})
}

func TestMul_issue341(t *testing.T) {
	//it works, cost = 91
	t.Run("2,3x3,2", func(t *testing.T) {
		z, err := runMul(tensor.Shape{2, 3}, tensor.Shape{3, 2})
		if err != nil {
			t.Fatal(err)
		}
		v, ok := z.Data().(float32)
		if !ok {
			t.Fail()
		}
		if v != float32(91) {
			t.Fail()
		}
	})

	t.Run("2,2x2,1", func(t *testing.T) {
		z, err := runMul(tensor.Shape{2, 2}, tensor.Shape{2, 1})
		if err != nil {
			t.Fatal(err)
		}
		v, ok := z.Data().(float32)
		if !ok {
			t.Fail()
		}
		if v != float32(6) {
			t.Fail()
		}
	})

	//panic
	t.Run("1,2x2,2", func(t *testing.T) {
		z, err := runMul(tensor.Shape{1, 2}, tensor.Shape{2, 2})
		if err != nil {
			t.Fatal(err)
		}
		v, ok := z.Data().(float32)
		if !ok {
			t.Fail()
		}
		if v != float32(6) {
			t.Fail()
		}
	})
}
