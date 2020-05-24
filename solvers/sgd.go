package solvers

import (
	"github.com/pkg/errors"
	gerrors "gorgonia.org/gorgonia/internal/errors"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/tensor"
)

// VanillaSolver is your bog standard stochastic gradient descent optimizer. There are no fancy features to this
type VanillaSolver struct {
	eta   float64 // learn rate
	clip  float64 // clip gradients
	l1reg float64 // l1 regularization parameter
	l2reg float64 // l2 regularization parameter
	batch float64 // batch size

	useClip, useL1Reg, useL2Reg bool
}

// NewVanillaSolver creates a new VanillaSolver with sane-ish default values
func NewVanillaSolver(opts ...SolverOpt) *VanillaSolver {
	s := &VanillaSolver{
		batch: 1,
		eta:   0.001,
	}
	for _, opt := range opts {
		opt(s)
	}
	return s
}

// Step steps through each node in the model and applies the most basic gradient descent algorithm on the value.
//
// This function will error out if the nodes do not have an associated Grad value.
func (s *VanillaSolver) Step(model []ValueGrad) (err error) {
	for _, n := range model {
		var weights, grad values.Value
		if weights, grad, err = extractWeightGrad(n); err != nil {
			return err
		}
		switch w := weights.(type) {
		case *tensor.Dense:
			g := grad.(*tensor.Dense)

			var l1reg, l2reg, clip, negClip, eta interface{}
			var onePerBatch interface{}
			switch w.Dtype() {
			case tensor.Float64:
				l1reg = s.l1reg
				l2reg = s.l2reg
				clip = s.clip
				negClip = -s.clip
				eta = -s.eta
				onePerBatch = float64(1) / s.batch
			case tensor.Float32:
				l1reg = float32(s.l1reg)
				l2reg = float32(s.l2reg)
				clip = float32(s.clip)
				negClip = float32(-s.clip)
				eta = float32(-s.eta)
				onePerBatch = float32(1) / float32(s.batch)
			}
			// prep the regularization of gradients
			var l1regs, l2regs tensor.Tensor
			if s.useL1Reg {
				if l1regs, err = tensor.Sign(w); err != nil {
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
				if l2regs, err = tensor.Mul(l2reg, w); err != nil {
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

			if _, err = tensor.Mul(eta, g, tensor.UseUnsafe()); err != nil {
				return errors.Wrap(err, pointwiseMulFail)
			}

			if _, err = tensor.Add(w, g, tensor.UseUnsafe()); err != nil {
				return errors.Wrap(err, addFail)
			}

			g.Zero()

		case *values.F32:
			g := grad.(*values.F32).Any()
			wv := w.Any()

			l1reg := float32(s.l1reg)
			l2reg := float32(s.l2reg)
			batch := float32(s.batch)
			clip := float32(s.clip)
			eta := float32(s.eta)

			if s.useL1Reg {
				if wv < 0 {
					l1reg = -l1reg
				}
				g += l1reg
			}

			if s.useL2Reg {
				l2reg *= wv
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

			upd := -eta * g
			wv += upd

			*(weights.(*values.F32)) = values.F32(wv)
			*(grad.(*values.F32)) = values.F32(0.0)
		case *values.F64:
			g := grad.(*values.F64).Any()
			wv := w.Any()

			l1reg := s.l1reg
			l2reg := s.l2reg
			batch := s.batch
			clip := s.clip
			eta := s.eta

			if s.useL1Reg {
				if wv < 0 {
					l1reg = -l1reg
				}
				g += l1reg
			}

			if s.useL2Reg {
				l2reg *= wv
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

			upd := -eta * g
			wv += upd

			*(weights.(*values.F64)) = values.F64(wv)
			*(grad.(*values.F64)) = values.F64(0.0)
		default:
			return errors.Errorf(gerrors.NYIFail, "VanillaSolver.step", w)
		}
	}
	return
}
