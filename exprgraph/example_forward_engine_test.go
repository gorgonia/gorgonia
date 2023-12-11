package exprgraph_test

import (
	"context"
	"fmt"
	"log"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia/exprgraph"
	"gorgonia.org/gorgonia/values/dual"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/dense"
)

var _ tensor.BLA[float64, *dual.Dual[float64, *dense.Dense[float64]]] = &FwdEngine[float64, *dense.Dense[float64]]{}
var _ tensor.Adder[float64, *dual.Dual[float64, *dense.Dense[float64]]] = &FwdEngine[float64, *dense.Dense[float64]]{}

// FwdEngine is a Engine that performs forwards mode differentiation
//
// Here the implementation is done by means of implementing MatMul and AddScalar
// Obviously in the real world situation, Add also needs to be implemented, but in this example
// we are not going to call Add, only AddScalar.
type FwdEngine[DT tensor.Num, T tensor.Tensor[DT, T]] struct {
	StandardEngine[DT, T]
	g *exprgraph.Graph
}

func (e *FwdEngine[DT, T]) Graph() *exprgraph.Graph { return e.g }

func (e *FwdEngine[DT, T]) SetGraph(g *exprgraph.Graph) { e.g = g }

func (e *FwdEngine[DT, T]) Lift(a exprgraph.Tensor) exprgraph.Tensor {
	switch t := a.(type) {
	case *dual.Dual[DT, T]:
		return a
	case T:
		return dual.New[DT, T](t)
	}
	panic("Unreachable")
}

func (e *FwdEngine[DT, T]) Inner(ctx context.Context, a, b *dual.Dual[DT, T]) (DT, error) {
	return 0, errors.New("NYI")
}

func (e *FwdEngine[DT, T]) FMA(ctx context.Context, a, x, retVal *dual.Dual[DT, T]) error {
	return errors.New("NYI")
}

func (e *FwdEngine[DT, T]) MatVecMul(ctx context.Context, a, b, retVal *dual.Dual[DT, T], incr []DT) error {
	return errors.New("NYI")
}

func (e *FwdEngine[DT, T]) Outer(ctx context.Context, a, b, retVal *dual.Dual[DT, T], incr []DT) error {
	return errors.New("NYI")
}

func (e *FwdEngine[DT, T]) MatMul(ctx context.Context, a, b, c *dual.Dual[DT, T], incr []DT) error {
	adv := a
	bdv := b
	cdv := c

	if err := e.StandardEngine.MatMul(ctx, adv.Value(), bdv.Value(), cdv.Value(), incr); err != nil {
		return err
	}

	advd := adv.Deriv()
	bdvd := bdv.Deriv()

	bdvT, err := bdv.Value().T()
	if err != nil {
		return err // cannot transpose
	}

	// dA = C×B'
	if err := e.StandardEngine.MatMul(ctx, cdv.Value(), bdvT, advd, nil); err != nil {
		return err
	}

	advT, err := adv.Value().T()
	if err != nil {
		return err // cannot transpose
	}

	// dB = A'×C
	if err := e.StandardEngine.MatMul(ctx, advT, cdv.Value(), bdvd, nil); err != nil {
		return err
	}

	return nil
}
func (e *FwdEngine[DT, T]) Add(ctx context.Context, a, b, retVal *dual.Dual[DT, T], toIncr bool) (err error) {
	return errors.New("NYI")
}

func (e *FwdEngine[DT, T]) AddScalar(ctx context.Context, a *dual.Dual[DT, T], b DT, retVal *dual.Dual[DT, T], leftTensor bool, toIncr bool) (err error) {
	adv := a
	//bdv := b
	if err = e.StandardEngine.AddScalar(ctx, adv.Value(), b, retVal.Value(), leftTensor, toIncr); err != nil {
		return err
	}

	advd := adv.Deriv()
	advd.Memset(1.0) // this is assuming we only work in float64. More needs to be done here
	return nil
}

// FwdEngine is a Engine that performs forwards mode differentiation
//
// Here the implementation is done by means of implementing MatMul and AddScalar
// Obviously in the real world situation, Add also needs to be implemented, but in this example
// we are not going to call Add, only AddScalar.
func Example_forward_differentiation_engine() {
	engine := &FwdEngine[float64, *dense.Dense[float64]]{StandardEngine: dense.StdFloat64Engine[*dense.Dense[float64]]{}}
	g := exprgraph.NewGraph(engine)
	engine.g = g

	x := exprgraph.New[float64](g, "x", tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))
	y := exprgraph.New[float64](g, "y", tensor.WithShape(3, 2), tensor.WithBacking([]float64{6, 5, 4, 3, 2, 1}))
	z := exprgraph.New[float64](g, "z", tensor.WithShape(), tensor.WithBacking([]float64{1}))
	xy, err := MatMul[float64, *dual.Dual[float64, *dense.Dense[float64]]](x, y)
	if err != nil {
		fmt.Println(err)
	}

	xypz, err := Add[float64, *dual.Dual[float64, *dense.Dense[float64]]](xy, z)
	if err != nil {
		fmt.Println(err)
		return
	}

	log.Printf("x %T, y %T", x, y)
	log.Printf("xy %T, xypz %T", xy, xypz)

	// note: getDeriv is defined in example_utils_test.go
	getD := getDeriv[float64, *dense.Dense[float64]]

	fmt.Printf("x:\n%v\ny:\n%v\nxy:\n%v\nxy+z:\n%v\n", x, y, xy, xypz)
	fmt.Printf("dx:\n%v\ndy:\n%v\ndxy:\n%v\ndxy+z:\n%v\n", getD(x), getD(y), getD(xy), getD(xypz))

	// Output:
	// x:
	// ⎡1  2  3⎤
	// ⎣4  5  6⎦
	//
	// y:
	// ⎡6  5⎤
	// ⎢4  3⎥
	// ⎣2  1⎦
	//
	// xy:
	// ⎡20  14⎤
	// ⎣56  41⎦
	//
	// xy+z:
	// ⎡21  15⎤
	// ⎣57  42⎦
	//
	// dx:
	// ⎡190  122   54⎤
	// ⎣541  347  153⎦
	//
	// dy:
	// ⎡244  178⎤
	// ⎢320  233⎥
	// ⎣396  288⎦
	//
	// dxy:
	// ⎡1  1⎤
	// ⎣1  1⎦
	//
	// dxy+z:
	// ⎡0  0⎤
	// ⎣0  0⎦

}
