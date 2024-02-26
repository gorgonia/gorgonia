package fwd

import (
	"context"
	"errors"

	"gorgonia.org/gorgonia/internal"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/gorgonia/values/dual"
	"gorgonia.org/tensor"
)

func (e *Engine[DT, T]) Add(ctx context.Context, a, b, retVal tensor.Basic[DT], toIncr bool) (err error) {
	adv := a.(dual.V)
	bdv := b.(dual.V)
	rdv := retVal.(dual.V)
	advv := adv.V().(T)
	bdvv := bdv.V().(T)
	rdvv := rdv.V().(T)

	if err = e.StandardEngine.Add(ctx, advv, bdvv, rdvv, toIncr); err != nil {
		return err
	}

	one := values.One[DT]()
	advd := adv.DV().(T)
	bdvd := bdv.DV().(T)
	if err = advd.Memset(one); err != nil {
		return err
	}
	if err = bdvd.Memset(one); err != nil {
		return err
	}
	return nil
}

func (e *Engine[DT, T]) AddScalar(ctx context.Context, a tensor.Basic[DT], b DT, retVal tensor.Basic[DT], leftTensor bool, toIncr bool) (err error) {
	adv := a.(dual.V)
	rdv := retVal.(dual.V)
	advv := adv.V().(T)
	rdvv := rdv.V().(T)

	//bdv := b
	if err = e.StandardEngine.AddScalar(ctx, advv, b, rdvv, leftTensor, toIncr); err != nil {
		return err
	}

	advd := adv.DV().(T)
	one := values.One[DT]()
	advd.Memset(one)
	return nil
}

func (e *Engine[DT, T]) AddBroadcastable(ctx context.Context, a, b, retVal tensor.Basic[DT], expAPA, expAPB *tensor.AP, toIncr bool) (err error) {
	panic("NYI")
}
