package solvers

import (
	"gorgonia.org/gorgonia/internal/errors"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/gorgonia/values/dual"
	"gorgonia.org/tensor"
)

// Momentum is the stochastic gradient descent optimizer with momentum item.
type Momentum struct {
	eta      float64 // learn rate
	momentum float64 // momentum
	clip     float64 // clip gradients
	l1reg    float64 // l1 regularization parameter
	l2reg    float64 // l2 regularization parameter
	batch    float64 // batch size

	useClip, useL1Reg, useL2Reg bool

	cache []*dual.Dual
}

// NewMomentum creates a new Momentum with sane-ish default values
func NewMomentum(opts ...SolverOpt) *Momentum {
	s := &Momentum{
		batch:    1,
		eta:      0.001,
		momentum: 0.9,
	}
	for _, opt := range opts {
		opt(s)
	}
	return s
}

// Step steps through each node in the model and applies the Momentum stochastic gradient descent algorithm on the value.
//
// This function will error out if the nodes do not have an associated Grad value.
func (s *Momentum) Step(model []ValueGrad) (err error) {
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
		// cw = cw * momentum - eta * grad
		// w = w + cw
		switch cw := cv.(type) {
		case *tensor.Dense:
			w := weights.(*tensor.Dense)
			g := grad.(*tensor.Dense)

			var l1reg, l2reg, clip, negClip, eta, momentum, onePerBatch interface{}
			switch cw.Dtype() {
			case tensor.Float64:
				l1reg = s.l1reg
				l2reg = s.l2reg
				clip = s.clip
				negClip = -s.clip
				eta = -s.eta
				momentum = s.momentum
				onePerBatch = float64(1) / s.batch
			case tensor.Float32:
				l1reg = float32(s.l1reg)
				l2reg = float32(s.l2reg)
				clip = float32(s.clip)
				negClip = float32(-s.clip)
				eta = float32(-s.eta)
				momentum = float32(s.momentum)
				onePerBatch = float32(1) / float32(s.batch)
			}

			// prep the regularization of gradients
			var l1regs, l2regs tensor.Tensor
			if s.useL1Reg {
				if l1regs, err = tensor.Sign(cw); err != nil {
					return errors.Wrap(err, signFail)
				}

				if l1regs, err = tensor.Mul(l1reg, l1regs, tensor.UseUnsafe()); err != nil {
					return errors.Wrap(err, pointwiseMulFail)
				}

				if _, err = tensor.Add(g, l1regs, tensor.UseUnsafe()); err != nil {
					return errors.Wrap(err, addFail)
				}

				defer tensor.ReturnTensor(l1regs)
			}

			if s.useL2Reg {
				if l2regs, err = tensor.Mul(l2reg, cw); err != nil {
					return errors.Wrap(err, pointwiseMulFail)
				}

				if _, err = tensor.Add(g, l2regs, tensor.UseUnsafe()); err != nil {
					return errors.Wrap(err, addFail)
				}

				defer tensor.ReturnTensor(l2regs)
			}

			if s.batch > 1 {
				if _, err = tensor.Mul(onePerBatch, g, tensor.UseUnsafe()); err != nil {
					return errors.Wrap(err, pointwiseMulFail)
				}
			}

			if s.useClip && s.clip > 0 {
				if _, err = tensor.Clamp(g, negClip, clip, tensor.UseUnsafe()); err != nil {
					return errors.Wrap(err, clampFail)
				}
			}

			// momentum
			if _, err = tensor.Mul(g, eta, tensor.UseUnsafe()); err != nil {
				return errors.Wrap(err, pointwiseMulFail)
			}

			// cw * momentum
			if _, err = tensor.Mul(cw, momentum, tensor.UseUnsafe()); err != nil {
				return errors.Wrap(err, pointwiseMulFail)
			}

			//  cw * momentum - eta * grad
			if _, err = tensor.Add(cw, g, tensor.UseUnsafe()); err != nil {
				return errors.Wrap(err, pointwiseMulFail)
			}

			if _, err = tensor.Add(w, cw, tensor.UseUnsafe()); err != nil {
				return errors.Wrap(err, addFail)
			}

			g.Zero()

		case *values.F32:
			l1reg := float32(s.l1reg)
			l2reg := float32(s.l2reg)
			batch := float32(s.batch)
			clip := float32(s.clip)
			eta := float32(s.eta)
			momentum := float32(s.momentum)

			g := grad.(*values.F32).Any()
			w := weights.(*values.F32).Any()
			c := cw.Any()

			if s.useL1Reg {
				if w < 0 {
					l1reg = -l1reg
				}
				g += l1reg
			}

			if s.useL2Reg {
				l2reg *= w
				g += l2reg
			}

			if batch > 1 {
				g *= (1 / batch)
			}

			if s.useClip {
				if g > clip {
					g = clip
				} else if g < -clip {
					g = -clip
				}
			}

			c = c*momentum - eta*g
			w += c

			*(weights.(*values.F32)) = values.F32(w)
			*(grad.(*values.F32)) = values.F32(0.0)
		case *values.F64:
			l1reg := s.l1reg
			l2reg := s.l2reg
			batch := s.batch
			clip := s.clip
			eta := s.eta
			momentum := s.momentum

			g := grad.(*values.F64).Any()
			w := weights.(*values.F64).Any()
			c := cw.Any()

			if s.useL1Reg {
				if w < 0 {
					l1reg = -l1reg
				}
				g += l1reg
			}

			if s.useL2Reg {
				l2reg *= w
				g += l2reg
			}

			if batch > 1 {
				g *= (1 / batch)
			}

			if s.useClip {
				if g > clip {
					g = clip
				} else if g < -clip {
					g = -clip
				}
			}

			c = c*momentum - eta*g
			w += c

			*(weights.(*values.F64)) = values.F64(w)
			*(grad.(*values.F64)) = values.F64(0.0)
		default:
			return errors.Errorf(errors.NYIFail, "Momentum.step", cv)
		}
	}
	return
}
