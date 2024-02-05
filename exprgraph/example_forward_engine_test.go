package exprgraph_test

import (
	"context"
	"fmt"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia/exprgraph"
	"gorgonia.org/gorgonia/values/dual"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/dense"
)

var _ tensor.BLA[float64, tensor.Basic[float64]] = (&FwdEngine[float64, *dense.Dense[float64]]{}).BasicEng().(*FwdEngine[float64, *dense.Dense[float64]])
var _ tensor.Adder[float64, tensor.Basic[float64]] = &FwdEngine[float64, *dense.Dense[float64]]{}

// FwdEngine is a Engine that performs forwards mode differentiation
//
// Here the implementation is done by means of implementing MatMul and AddScalar
// Obviously in the real world situation, Add also needs to be implemented, but in this example
// we are not going to call Add, only AddScalar.
type FwdEngine[DT tensor.Num, T tensor.Basic[DT]] struct {
	StandardEngine[DT, T]
	g *exprgraph.Graph
}

func (e *FwdEngine[DT, T]) BasicEng() tensor.Engine {
	//return &FwdEngine[DT, tensor.Basic[DT]]{StandardEngine: e.StandardEngine.BasicEng().(StandardEngine[DT, tensor.Basic[DT]]), g: e.g}
	return e
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

func (e *FwdEngine[DT, T]) Inner(ctx context.Context, a, b tensor.Basic[DT]) (DT, error) {
	return 0, errors.New("NYI")
}

func (e *FwdEngine[DT, T]) FMA(ctx context.Context, a, x, retVal tensor.Basic[DT]) error {
	return errors.New("NYI")
}

func (e *FwdEngine[DT, T]) MatVecMul(ctx context.Context, a, b, retVal tensor.Basic[DT], incr []DT) error {
	return errors.New("NYI")
}

func (e *FwdEngine[DT, T]) Outer(ctx context.Context, a, b, retVal tensor.Basic[DT], incr []DT) error {
	return errors.New("NYI")
}

func (e *FwdEngine[DT, T]) MatMul(ctx context.Context, a, b, c tensor.Basic[DT], incr []DT) error {
	adv := a.(dual.V)
	bdv := b.(dual.V)
	cdv := c.(dual.V)

	advv := adv.V().(T)
	bdvv := bdv.V().(T)
	cdvv := cdv.V().(T)

	if err := e.StandardEngine.MatMul(ctx, advv, bdvv, cdvv, incr); err != nil {
		return err
	}

	advd := adv.DV().(T)
	bdvd := bdv.DV().(T)

	bdvT, err := bdv.V().(tensor.Operable[T]).T()
	if err != nil {
		return err // cannot transpose
	}

	// dA = C×B'
	if err := e.StandardEngine.MatMul(ctx, cdvv, bdvT, advd, advd.Data()); err != nil {
		return err
	}

	advT, err := adv.V().(tensor.Operable[T]).T()
	if err != nil {
		return err // cannot transpose
	}

	// dB = A'×C
	if err := e.StandardEngine.MatMul(ctx, advT, cdvv, bdvd, bdvd.Data()); err != nil {
		return err
	}

	return nil
}
func (e *FwdEngine[DT, T]) Add(ctx context.Context, a, b, retVal tensor.Basic[DT], toIncr bool) (err error) {
	return errors.New("NYI")
}

func (e *FwdEngine[DT, T]) AddScalar(ctx context.Context, a tensor.Basic[DT], b DT, retVal tensor.Basic[DT], leftTensor bool, toIncr bool) (err error) {
	adv := a.(dual.V)
	rdv := retVal.(dual.V)
	advv := adv.V().(T)
	rdvv := rdv.V().(T)

	//bdv := b
	if err = e.StandardEngine.AddScalar(ctx, advv, b, rdvv, leftTensor, toIncr); err != nil {
		return err
	}

	advd := adv.DV().(T)
	advd.Memset(1.0) // this is assuming we only work in float64. More needs to be done here
	return nil
}

func (e *FwdEngine[DT, T]) AddBroadcastable(ctx context.Context, a, b, retVal tensor.Basic[DT], expAPA, expAPB *tensor.AP, toIncr bool) (err error) {
	return errors.New("NYI")
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
	xy, err := MatMul[float64, tensor.Basic[float64]](x, y)
	if err != nil {
		fmt.Println(err)
	}

	xypz, err := Add[float64, tensor.Basic[float64]](xy, z)
	if err != nil {
		fmt.Println(err)
		return
	}

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
