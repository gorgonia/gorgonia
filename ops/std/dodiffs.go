package stdops

import (
	"context"

	"gorgonia.org/gorgonia/exprgraph"
	"gorgonia.org/gorgonia/values/dual"
	"gorgonia.org/tensor"
)

// DoDiff is the method that allows automatic differentiation of `add`.
func (op addOp) DoDiff(ctx context.Context, inputs []Tensor, output Tensor) (err error) {
	adv := exprgraph.T2B(inputs[0]).(*dual.Dual)
	bdv := exprgraph.T2B(inputs[1]).(*dual.Dual)

	advd := adv.Deriv()
	bdvd := bdv.Deriv()

	ones := one(adv.Dtype())
	if _, err = tensor.Add(advd, ones, tensor.UseUnsafe()); err != nil {
		return err
	}
	if _, err = tensor.Add(bdvd, ones, tensor.UseUnsafe()); err != nil {
		return err
	}
	return nil
}
