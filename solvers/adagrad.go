package solvers

import (
	"math"

	"github.com/chewxy/math32"
	"gorgonia.org/gorgonia/internal/errors"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/gorgonia/values/dual"
	"gorgonia.org/tensor"
)

// AdaGradSolver is the solver that does adaptive gradient descent. Read the paper: http://jmlr.org/papers/v12/duchi11a.html
type AdaGradSolver struct {
	eta   float64 // learn rate
	eps   float64 // smoothing factor
	l1Reg float64 // l1reg param
	l2reg float64 // l2reg param
	clip  float64 // clip at

	useL2Reg, useClip bool

	cache []*dual.Dual
}

// NewAdaGradSolver creates a new AdaGradSolver with sane-ish default values
func NewAdaGradSolver(opts ...SolverOpt) *AdaGradSolver {
	s := &AdaGradSolver{
		eta: 0.001,
		eps: 1e-8,
	}

	for _, opt := range opts {
		opt(s)
	}
	return s
}

// Step steps through each node in the model and applies the Adaptive Gradient gradient descent algorithm on the value.
//
// This function will error out if the nodes do not have an associated Grad value.
func (s *AdaGradSolver) Step(model []ValueGrad) (err error) {
	if s.cache == nil {
		s.cache = make([]*dual.Dual, len(model))
	}

	for i, n := range model {
		var weights, grad values.Value
		if weights, grad, err = extractWeightGrad(n); err != nil {
			return err
		}

		var cached *dual.Dual
		if cached = s.cache[i]; cached == nil {
			if cached, err = newCachedDV(n, weights, grad, true); err != nil {
				return err
			}
			s.cache[i] = cached
		}

		cv := cached.Value

		switch cw := cv.(type) {
		case *tensor.Dense:
			var w, g, c, g2, regularized tensor.Tensor

			var l2reg, clip, negClip, eps, eta interface{}
			switch cw.Dtype() {
			case tensor.Float64:
				l2reg = s.l2reg
				clip = s.clip
				negClip = -s.clip
				eps = s.eps
				eta = -s.eta
			case tensor.Float32:
				l2reg = float32(s.l2reg)
				clip = float32(s.clip)
				negClip = float32(-s.clip)
				eps = float32(s.eps)
				eta = float32(-s.eta)
			}

			g = grad.(*tensor.Dense)
			if g2, err = tensor.Square(g); err != nil {
				return errors.Wrap(err, pointwiseSquareFail)
			}

			c = cw
			tensor.Add(c, g2, tensor.UseUnsafe())
			defer tensor.ReturnTensor(g2)

			if s.useClip {
				if _, err = tensor.Clamp(g, negClip, clip, tensor.UseUnsafe()); err != nil {
					return errors.Wrap(err, clampFail)
				}
			}

			// update
			var upd tensor.Tensor
			if upd, err = tensor.Add(c, eps); err != nil {
				return errors.Wrap(err, addFail)
			}

			if _, err = tensor.InvSqrt(upd, tensor.UseUnsafe()); err != nil {
				return errors.Wrap(err, invSqrtFail)
			}
			if _, err = tensor.Mul(g, eta, tensor.UseUnsafe()); err != nil {
				return errors.Wrap(err, pointwiseMulFail)
			}

			if _, err = tensor.Mul(upd, g, tensor.UseUnsafe()); err != nil {
				return errors.Wrap(err, pointwiseMulFail)
			}

			// regularize
			w = weights.(*tensor.Dense)

			if s.useL2Reg {
				if regularized, err = tensor.Mul(w, l2reg); err != nil {
					return errors.Wrap(err, pointwiseMulFail)
				}

				if _, err = tensor.Sub(upd, regularized, tensor.UseUnsafe()); err != nil {
					return errors.Wrap(err, subFail)
				}

				defer tensor.ReturnTensor(regularized)
			}

			if _, err = tensor.Add(w, upd, tensor.UseUnsafe()); err != nil {
				return errors.Wrap(err, addFail)
			}
			defer tensor.ReturnTensor(upd)

			// zero all
			g.Zero()

		case *values.F32:
			var w, g, c float32

			l2reg := float32(s.l2reg)
			clip := float32(s.clip)
			eps := float32(s.eps)
			eta := float32(s.eta)

			c = cw.Any()
			g = grad.(*values.F32).Any()

			c += g * g

			if s.useClip {
				if g > clip {
					g = clip
				} else if g < -clip {
					g = -clip
				}
			}

			w = weights.(*values.F32).Any()

			upd := -eta * g / math32.Sqrt(c+eps)

			if s.useL2Reg {
				upd -= w * l2reg
			}

			w += upd

			// because scalar values are copies, and not pointers, we have to actually re-update the dualValu in model[i]
			*(weights.(*values.F32)) = values.F32(w)
			*(grad.(*values.F32)) = values.F32(0.0)
		case *values.F64:
			var w, g, c float64

			l2reg := s.l2reg
			clip := s.clip
			eps := s.eps
			eta := s.eta

			c = cw.Any()
			g = grad.(*values.F64).Any()

			c += g * g

			if s.useClip {
				if g > clip {
					g = clip
				} else if g < -clip {
					g = -clip
				}
			}

			w = weights.(*values.F64).Any()
			upd := -eta * g / math.Sqrt(c+eps)
			if s.useL2Reg {
				upd -= w * l2reg
			}

			w += upd

			// because scalar values are copies, and not pointers, we have to actually re-update the dualValu in model[i]
			*(weights.(*values.F64)) = values.F64(w)
			*(grad.(*values.F64)) = values.F64(0.0)

		default:
			return errors.Errorf(errors.NYIFail, "Adagrad step", cv)
		}

	}

	return
}
