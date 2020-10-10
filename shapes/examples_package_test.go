package shapes_test

import (
	"fmt"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia/shapes"
)

type Dense struct {
	data  []float64
	shape shapes.Shape
}

func (d *Dense) Shape() shapes.Shape { return d.shape }

type elemwiseFn func(a, b *Dense) (*Dense, error)

func (f elemwiseFn) Shape() shapes.Expr {
	return shapes.Arrow{
		shapes.Var('a'),
		shapes.Arrow{shapes.Var('a'), shapes.Var('a')},
	}
}

type matmulFn func(a, b *Dense) (*Dense, error)

func (f matmulFn) Shape() shapes.Expr {
	return shapes.Arrow{
		shapes.Abstract{shapes.Var('a'), shapes.Var('b')},
		shapes.Arrow{
			shapes.Abstract{shapes.Var('b'), shapes.Var('c')},
			shapes.Abstract{shapes.Var('a'), shapes.Var('c')},
		},
	}
}

type applyFn func(*Dense, func(float64) float64) (*Dense, error)

func (f applyFn) Shape() shapes.Expr {
	return shapes.Arrow{
		shapes.Var('a'),
		shapes.Arrow{
			shapes.Arrow{
				shapes.Var('b'),
				shapes.Var('b'),
			},
			shapes.Var('a'),
		},
	}
}

func (d *Dense) MatMul(other *Dense) (*Dense, error) {
	expr := matmulFn((*Dense).MatMul).Shape()
	sh, err := infer(expr, d.Shape(), other.Shape())
	if err != nil {
		return nil, err
	}

	retVal := &Dense{
		data:  make([]float64, sh.TotalSize()),
		shape: sh,
	}
	return retVal, nil
}

func (d *Dense) Add(other *Dense) (*Dense, error) {
	expr := elemwiseFn((*Dense).Add).Shape()
	sh, err := infer(expr, d.Shape(), other.Shape())
	if err != nil {
		return nil, err
	}
	retVal := &Dense{
		data:  make([]float64, sh.TotalSize()),
		shape: sh,
	}
	return retVal, nil
}

func (d *Dense) Apply(fn func(float64) float64) (*Dense, error) {
	expr := applyFn((*Dense).Apply).Shape()
	fnShape := shapes.Arrow{shapes.Shape{}, shapes.Shape{}}
	sh, err := infer(expr, d.Shape(), fnShape)
	if err != nil {
		return nil, err
	}
	return &Dense{shape: sh}, nil
}

func infer(fn shapes.Expr, others ...shapes.Expr) (shapes.Shape, error) {
	retShape, err := shapes.InferApp(fn, others...)
	if err != nil {
		return nil, err
	}
	sh, err := shapes.ToShape(retShape)
	if err != nil {
		return nil, errors.Wrapf(err, "Expected a Shape in retShape. Got %v of %T instead", retShape, retShape)
	}
	return sh, nil
}

func Example_package() {
	a := &Dense{
		data:  []float64{1, 2, 3, 4, 5, 6},
		shape: shapes.Shape{2, 3},
	}
	b := &Dense{
		data:  []float64{10, 20, 30, 40, 50, 60},
		shape: shapes.Shape{3, 2},
	}
	c, err := a.MatMul(b)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("c: %v\n", c.Shape())

	d := &Dense{
		data:  []float64{0, 0, 0, 0},
		shape: shapes.Shape{2, 2},
	}
	e, err := c.Add(d)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("e: %v\n", e.Shape())

	f, err := e.Apply(func(a float64) float64 { return a * a })
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("f: %v\n", f.Shape())

	// trying to add to a bad shape will yield an error
	_, err = e.Add(a)
	fmt.Println(err)

	// Output:
	// c: (2, 2)
	// e: (2, 2)
	// f: (2, 2)
	// Failed to solve [{(2, 2) → (2, 2) = (2, 3) → a}] | a: Unification Fail. 2 ~ 3 cannot proceed

}
