package fwd

import (
	"context"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia/values/dual"
	"gorgonia.org/tensor"
)

func (e *Engine[DT, T]) Inner(ctx context.Context, a, b tensor.Basic[DT]) (DT, error) {
	var z DT
	return z, errors.New("NYI")
}

func (e *Engine[DT, T]) FMA(ctx context.Context, a, x, retVal tensor.Basic[DT]) error {
	return errors.New("NYI")
}

func (e *Engine[DT, T]) MatVecMul(ctx context.Context, a, b, retVal tensor.Basic[DT], incr []DT) error {
	return errors.New("NYI")
}

func (e *Engine[DT, T]) Outer(ctx context.Context, a, b, retVal tensor.Basic[DT], incr []DT) error {
	return errors.New("NYI")
}

func (e *Engine[DT, T]) MatMul(ctx context.Context, a, b, c tensor.Basic[DT], incr []DT) error {
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
