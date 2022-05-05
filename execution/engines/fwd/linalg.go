package fwd

import (
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia/values/dual"
	"gorgonia.org/tensor"
)

func (e *Engine) MatMul(a, b, c tensor.Tensor) error {
	adv := a.(*dual.Dual)
	bdv := b.(*dual.Dual)
	cdv := c.(*dual.Dual)

	// perform forwards
	if err := e.StdEng.MatMul(adv.Value, bdv.Value, cdv.Value); err != nil {
		return errors.Wrap(err, "Unable to perform MatMul op in fwd.Engine")
	}

	// perform differentiation
	if err := e.diffMatMul(adv, bdv, cdv); err != nil {
		return errors.Wrap(err, "Unable to differentiate a MatMul op in fwd.Engine")
	}
	return nil
}

func (e *Engine) diffMatMul(adv, bdv, cdv *dual.Duaal) error {
	advd := adv.Deriv()
	bdvd := bdv.Deriv()

	if err := bdv.Value.T(); err != nil {
		return errors.Wrap(err, "Differentiaing a MatMul op: Unable to transpose B")
	}

	// dA = C×B'
	if err := e.StdEng.MatMul(cdv.Value, bdv.Value, advd); err != nil {
		return errors.Wrap(err, "Differentiating a MatMul op: Unable to perform C×B'")
	}

	if err := adv.Value.T(); err != nil {
		return errors.Wrap(err, "Differentiating a MatMul op: Unable to transpose A")
	}

	// dB = A'×C
	if err := e.StdEng.MatMul(adv.Value, cdv.Value, bdvd); err != nil {
		return errors.Wrap(err, "Differentiating a MatMul op: Unable to perform A'×C")
	}

	// now we undo our transposes
	adv.Value.UT()
	bdv.Value.UT()

	return nil
}
