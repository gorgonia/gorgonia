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
	fnShape := shapes.ShapeOf(fn)
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
	A := &Dense{
		data:  []float64{1, 2, 3, 4, 5, 6},
		shape: shapes.Shape{2, 3},
	}
	B := &Dense{
		data:  []float64{10, 20, 30, 40, 50, 60},
		shape: shapes.Shape{3, 2},
	}
	var fn shapes.Exprer = matmulFn((*Dense).MatMul)
	C, err := A.MatMul(B)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("×: %v\n", fn.Shape())
	fmt.Printf("\t   A   ×    B   =    C\n\t%v × %v = %v\n", A.Shape(), B.Shape(), C.Shape())
	fmt.Println("---")

	fn = elemwiseFn((*Dense).Add)
	D := &Dense{
		data:  []float64{0, 0, 0, 0},
		shape: shapes.Shape{2, 2},
	}
	E, err := C.Add(D)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("+: %v\n", fn.Shape())
	fmt.Printf("\t   C   +    D   =    E\n")
	fmt.Printf("\t%v + %v = %v\n", C.Shape(), D.Shape(), E.Shape())
	fmt.Println("---")

	square := func(a float64) float64 { return a * a }
	squareShape := shapes.ShapeOf(square)
	fn = applyFn((*Dense).Apply)
	F, err := E.Apply(square)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("@: %v\n", fn.Shape())
	fmt.Printf("square: %v\n", squareShape)
	fmt.Printf("\t   E   @  square =    F\n")
	fmt.Printf("\t%v @ (%v) = %v\n", E.Shape(), squareShape, F.Shape())
	fmt.Println("---")

	// trying to do a bad add (e.g. adding two matrices with different shapes) will yield an error
	fn = elemwiseFn((*Dense).Add)
	_, err = E.Add(A)
	fmt.Printf("+: %v\n", fn.Shape())
	fmt.Printf("\t   E   +   A    =\n")
	fmt.Printf("\t%v + %v = ", E.Shape(), A.Shape())
	fmt.Println(err)

	// Output:
	// ×: (a, b) → (b, c) → (a, c)
	//	   A   ×    B   =    C
	//	(2, 3) × (3, 2) = (2, 2)
	// ---
	// +: a → a → a
	//	   C   +    D   =    E
	//	(2, 2) + (2, 2) = (2, 2)
	// ---
	// @: a → (b → b) → a
	// square: () → ()
	//	   E   @  square =    F
	//	(2, 2) @ (() → ()) = (2, 2)
	// ---
	// +: a → a → a
	//	   E   +   A    =
	//	(2, 2) + (2, 3) = Failed to solve [{(2, 2) → (2, 2) = (2, 3) → a}] | a: Unification Fail. 2 ~ 3 cannot proceed
	//

}
