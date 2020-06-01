package solvers

import (
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/gorgonia/values/dual"
)

// Solver is anything that does gradient updates.
// The name solvers is stolen from Caffe. A much shorter name than GradientUpdaters
type Solver interface {
	Step([]ValueGrad) error
}

// ValueGrad is any type that has a value and a grad. This is used for Solvers
type ValueGrad interface {
	values.Valuer
	Grad() (values.Value, error)
}

// Namer is anything that has a name
type Namer interface {
	Name() string
}

// newCachedDV creates a *dual.Dual to cache the grad and node values.
func newCachedDV(n ValueGrad, weights, grad values.Value, zero bool) (cached *dual.Dual, err error) {
	cached = new(dual.Dual)
	if cached.Value, err = values.Clone(weights); err != nil {
		if nm, ok := n.(Namer); ok {
			return nil, errors.Errorf("Failed to clone weights of %v", nm.Name())
		}
		return nil, errors.New("Failed to clone weights")
	}
	var cloned values.Value
	if cloned, err = values.Clone(grad); err != nil {
		if nm, ok := n.(Namer); ok {
			return nil, errors.Errorf("Failed to clone grad of %v", nm.Name())
		}
		return nil, errors.New("Failed to clone grad")
	}
	cached.SetDeriv(cloned)
	if zero {
		cached.Value = values.ZeroValue(cached.Value)
		cached.SetDeriv(values.ZeroValue(cached.Deriv()))
	}
	return
}

// extractWeightGrad takes a ValueGrad (usually a *Node), and returns its weights and grads.
//
// Bear in mind that the grad may not necessarily be attached to the *dual.Dual of the value.
func extractWeightGrad(n ValueGrad) (weights, grad values.Value, err error) {
	weights = n.Value()
	if grad, err = n.Grad(); err != nil {
		if dv, ok := weights.(*dual.Dual); ok {
			weights = dv.Value
			grad = dv.Deriv()
			return
		}
		if nm, ok := n.(Namer); ok {
			return weights, nil, errors.Wrapf(err, "No Grad found for %v", nm.Name())
		}
		return weights, nil, errors.Wrap(err, "No Grad found")
	}
	return
}
