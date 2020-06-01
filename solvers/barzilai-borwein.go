package solvers

import (
	"math"

	"github.com/pkg/errors"
	gerrors "gorgonia.org/gorgonia/internal/errors"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/gorgonia/values/dual"
	"gorgonia.org/tensor"
)

// BarzilaiBorweinSolver / Barzilai-Borwein performs Gradient Descent in steepest descend direction
// Solves 0 = F(x), by
//  xᵢ₊₁ = xᵢ - eta * Grad(F)(xᵢ)
// Where the learn rate eta is calculated by the Barzilai-Borwein method:
//  eta(xᵢ) = <(xᵢ - xᵢ₋₁), (Grad(F)(xᵢ) - Grad(F)(xᵢ₋₁))> /
//                  ∥(Grad(F)(xᵢ) - Grad(F)(xᵢ₋₁))∥²
// The input learn rate is used for the first iteration.
//
// TODO: Check out stochastic implementations, e.g. "Barzilai-Borwein Step Size for Stochastic Gradient Descent" https://arxiv.org/abs/1605.04131
type BarzilaiBorweinSolver struct {
	eta     float64 // initial learn rate
	clip    float64 // clip value
	useClip bool
	prevDV  []*dual.Dual // dual value for xᵢ₋₁ step
}

// NewBarzilaiBorweinSolver creates a new Barzilai-Borwein solver withs some default values:
// the learn rate is set to 0.001 and the solver does not use clipping.
func NewBarzilaiBorweinSolver(opts ...SolverOpt) *BarzilaiBorweinSolver {
	s := &BarzilaiBorweinSolver{
		eta:     0.001,
		useClip: false,
	}

	for _, opt := range opts {
		opt(s)
	}
	return s
}

// Step steps through each node in the model and applies the Barzilai-Borwein gradient descent algorithm on the value.
//
// This function will error out if the nodes do not have an associated Grad value.
func (s *BarzilaiBorweinSolver) Step(model []ValueGrad) (err error) {

	firstRun := false
	if s.prevDV == nil {
		firstRun = true
		s.prevDV = make([]*dual.Dual, len(model))
	}

	// Update the learning rate
	if false == firstRun {
		nominator := float64(0.0)
		denominator := float64(0.0)

		for nodeNr, node := range model {
			var weights, grad values.Value
			if weights, grad, err = extractWeightGrad(node); err != nil {
				return err
			}

			switch w := weights.(type) {
			case *tensor.Dense:
				g, ok := grad.(*tensor.Dense)
				if !ok {
					return errors.Errorf("Expected a *tensor.Dense in %v. Got %T instead", node, grad)
				}

				wOld, ok := s.prevDV[nodeNr].Value.(*tensor.Dense)
				if !ok {
					return errors.Errorf("Expected a *tensor.Dense in %v. Got %T instead", node, s.prevDV[nodeNr].Value)
				}

				gOld, ok := s.prevDV[nodeNr].Deriv().(*tensor.Dense)
				if !ok {
					return errors.Errorf("Expected a *tensor.Dense in %v. Got %T instead", node, s.prevDV[nodeNr].Deriv())
				}

				valueDiff, err := tensor.Sub(w, wOld)
				defer tensor.ReturnTensor(valueDiff)
				if err != nil {
					return errors.Wrap(err, subFail)
				}

				gradDiff, err := tensor.Sub(g, gOld)
				defer tensor.ReturnTensor(gradDiff)
				if err != nil {
					return errors.Wrap(err, subFail)
				}

				// <(xᵢ - xᵢ₋₁), (Grad(F)(xᵢ) - Grad(F)(xᵢ₋₁))>

				// Scalar Product == Total tensor contraction
				dims := valueDiff.Dims()
				contractionAxes := make([]int, dims, dims)
				for axis := 0; axis < len(contractionAxes); axis++ {
					contractionAxes[axis] = axis
				}

				valGradDiffscalarProd, err := tensor.Contract(valueDiff, gradDiff, contractionAxes, contractionAxes)
				if err != nil {
					return errors.New("operationError, Contracting value / gradient difference")
				}
				defer tensor.ReturnTensor(valGradDiffscalarProd)

				nominator += valGradDiffscalarProd.Data().(float64)

				// ∥(Grad(F)(xᵢ) - Grad(F)(xᵢ₋₁))∥²
				gradDiffscalarProd, err := tensor.Contract(gradDiff, gradDiff, contractionAxes, contractionAxes)
				if err != nil {
					return errors.New("operationError, Contracting value / gradient difference")
				}
				defer tensor.ReturnTensor(gradDiffscalarProd)

				denominator += gradDiffscalarProd.Data().(float64)

			default:
				return errors.Errorf(gerrors.NYIFail, "Barizai-Borwein step", w)
			}
		}

		s.eta = nominator / denominator

		if s.useClip && (math.Abs(s.eta) > s.clip) {
			if math.Signbit(s.eta) {
				s.eta = -s.clip
			} else {
				s.eta = s.clip
			}
		}
	}

	// Save this iteration's values for the next run
	for nodeNr, node := range model {
		var weights, grad values.Value
		if weights, grad, err = extractWeightGrad(node); err != nil {
			return err
		}

		if false == firstRun {
			// return memory for the old dual value used in this iteration
			dual.ReturnDV(s.prevDV[nodeNr])
		}
		var oldDV *dual.Dual
		if oldDV, err = newCachedDV(node, weights, grad, false); err != nil {
			return err
		}
		s.prevDV[nodeNr] = oldDV
	}

	// Update the weights
	for _, node := range model {
		var weights, grad values.Value
		if weights, grad, err = extractWeightGrad(node); err != nil {
			return err
		}

		switch w := weights.(type) {
		case *tensor.Dense:
			g, ok := grad.(*tensor.Dense)
			if !ok {
				return errors.Errorf("Expected a *tensor.Dense in %v. Got %T instead", node, grad)
			}

			upd, err := tensor.Mul(g, s.eta)
			defer tensor.ReturnTensor(upd)

			if err != nil {
				return errors.Wrap(err, pointwiseMulFail)
			}

			if _, err = tensor.Sub(w, upd, tensor.UseUnsafe()); err != nil {
				return errors.Wrap(err, subFail)
			}

			g.Zero()

		default:
			return errors.Errorf(gerrors.NYIFail, "Barizai-Borwein step", w)
		}
	}

	return nil
}
