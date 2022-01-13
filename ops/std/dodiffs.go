package stdops

import (
	"context"

	"gorgonia.org/gorgonia/exprgraph"
	"gorgonia.org/gorgonia/internal/datatypes"
	"gorgonia.org/gorgonia/values/dual"
	"gorgonia.org/tensor"
)

// DoDiff is the method that allows automatic differentiation of `add`.
func (op addOp) DoDiff(ctx context.Context, inputs []datatypes.Tensor, output datatypes.Tensor) (err error) {
	adv := exprgraph.T2T(inputs[0]).(*dual.Dual)
	bdv := exprgraph.T2T(inputs[1]).(*dual.Dual)

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
