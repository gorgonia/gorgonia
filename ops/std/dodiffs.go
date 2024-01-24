package stdops

import (
	"context"

	"gorgonia.org/gorgonia/exprgraph"
	"gorgonia.org/gorgonia/internal/datatypes"
	"gorgonia.org/gorgonia/values/dual"
	"gorgonia.org/tensor"
)

// DoDiff is the method that allows automatic differentiation of `add`.
func (op addOp[DT, T]) DoDiff(ctx context.Context, inputs []datatypes.Tensor, output datatypes.Tensor) (err error) {
	adv := exprgraph.T2B[DT](inputs[0]).(*dual.Dual[DT, T])
	bdv := exprgraph.T2B[DT](inputs[1]).(*dual.Dual[DT, T])

	advd := adv.Deriv()
	bdvd := bdv.Deriv()

	ones := one(adv.Dtype())
	if _, err = tensor.Add(advd, ones, tensor.UseUnsafe); err != nil {
		return err
	}
	if _, err = tensor.Add(bdvd, ones, tensor.UseUnsafe); err != nil {
		return err
	}
	return nil
}
