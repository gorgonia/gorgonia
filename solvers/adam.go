package solvers

import (
	"math"

	"github.com/pkg/errors"
	gerrors "gorgonia.org/gorgonia/internal/errors"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/gorgonia/values/dual"
	"gorgonia.org/tensor"
)

// AdamSolver is the Adaptive Moment Estimation solver (basically RMSProp on steroids).
// Paper: http://arxiv.org/abs/1412.6980
//
// We overload the purpose of existing data structure of a *values.Dual. However, instead of just holding a value and its derivative,
// the cache's *values.Duals hold the Means of gradients (in .Value) and the variances of the gradients (in .d)
type AdamSolver struct {
	eta   float64 // learn rate
	eps   float64 // smoothing
	beta1 float64 // modifier for means
	beta2 float64 // modifier for variances
	clip  float64 // clip gradients
	l1reg float64 // l1 regularization parameter
	l2reg float64 // l2 regularization parameter
	batch float64 // batch size

	useClip, useL1Reg, useL2Reg bool

	// unsettable
	iter  int
	cache []*dual.Dual
}

// NewAdamSolver creates an Adam solver with these default values:
//		eta (learn rate)	  	: 0.001
//		eps (smoothing factor)		: 1e-8
//		beta1				: 0.9
//		beta2 				: 0.999
//		batch				: 1
func NewAdamSolver(opts ...SolverOpt) *AdamSolver {
	s := &AdamSolver{
		eta:   0.001,
		eps:   1e-8,
		beta1: 0.9,
		beta2: 0.999,
		batch: 1,
	}

	for _, opt := range opts {
		opt(s)
	}
	return s
}

// Step steps through each node in the model and applies the Adaptive Moment Estimation gradient descent algorithm on the value.
//
// This function will error out if the nodes do not have an associated Grad value.
func (s *AdamSolver) Step(model []ValueGrad) (err error) {
	if s.cache == nil {
		s.cache = make([]*dual.Dual, len(model))
	}

	s.iter++
	correction1 := (1 - math.Pow(s.beta1, float64(s.iter)))
	correction2 := (1 - math.Pow(s.beta2, float64(s.iter)))

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

		cvm := cached.Value   // means of gradients
		cvv := cached.Deriv() // variances of gradients

		switch m := cvm.(type) {
		case *tensor.Dense:
			g := grad.(*tensor.Dense)
			w := weights.(*tensor.Dense)
			v := cvv.(*tensor.Dense)

			var l1reg, l2reg, clip, negClip, beta1, beta2, omβ1, omβ2, eps, eta, onePerBatch interface{}
			var correctionV1, correctionV2 interface{}
			switch m.Dtype() {
			case tensor.Float64:
				l1reg = s.l1reg
				l2reg = s.l2reg
				clip = s.clip
				negClip = -s.clip
				beta1 = s.beta1
				beta2 = s.beta2
				omβ1 = float64(1) - s.beta1
				omβ2 = float64(1) - s.beta2
				eps = s.eps
				eta = -s.eta
				onePerBatch = float64(1) / s.batch
				correctionV1 = float64(1) / float64(correction1)
				correctionV2 = float64(1) / float64(correction2)
			case tensor.Float32:
				l1reg = float32(s.l1reg)
				l2reg = float32(s.l2reg)
				clip = float32(s.clip)
				negClip = float32(s.clip)
				beta1 = float32(s.beta1)
				beta2 = float32(s.beta2)
				omβ1 = float32(1) - float32(s.beta1)
				omβ2 = float32(1) - float32(s.beta2)
				eps = float32(s.eps)
				eta = float32(-s.eta)
				onePerBatch = float32(1) / float32(s.batch)
				correctionV1 = float32(1) / float32(correction1)
				correctionV2 = float32(1) / float32(correction2)
			}

			// prep the regularization of gradients
			if s.useL1Reg {
				var l1regs tensor.Tensor
				if l1regs, err = tensor.Sign(w); err != nil {
					errors.Wrap(err, signFail)
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
				var l2regs tensor.Tensor
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

			// prep done. Now let's apply the formula:
			// the formula is
			//		(β_1 * m_t-1) + (1 - β_1)g_t ..................	1
			//		(β_2 * v_t-1) + (1 - β_2)*(g_t)² .............	2

			// equation(1)
			t1 := g.Clone().(*tensor.Dense)
			if _, err = tensor.Mul(omβ1, t1, tensor.UseUnsafe()); err != nil {
				return errors.Wrap(err, pointwiseMulFail)
			}

			// equation(2)
			if _, err = tensor.Mul(g, g, tensor.UseUnsafe()); err != nil {
				return errors.Wrap(err, pointwiseMulFail)
			}
			if _, err = tensor.Mul(omβ2, g, tensor.UseUnsafe()); err != nil {
				return errors.Wrap(err, pointwiseMulFail)
			}

			// equation (1)
			if _, err = tensor.Mul(beta1, m, tensor.WithIncr(t1)); err != nil {
				return errors.Wrap(err, pointwiseMulFail)
			}

			// equation (2)
			if _, err = tensor.Mul(beta2, v, tensor.WithIncr(g)); err != nil {
				return errors.Wrap(err, pointwiseMulFail)
			}

			defer tensor.ReturnTensor(m)
			defer tensor.ReturnTensor(v)
			cached.SetValue(t1)
			cached.SetDeriv(g.Clone().(*tensor.Dense))

			// now deal with the hats
			mHats := t1.Clone().(*tensor.Dense)
			vHats := g.Clone().(*tensor.Dense)

			if _, err = tensor.Mul(correctionV1, mHats, tensor.UseUnsafe()); err != nil {
				return errors.Wrap(err, pointwiseMulFail)
			}

			if _, err = tensor.Mul(correctionV2, vHats, tensor.UseUnsafe()); err != nil {
				return errors.Wrap(err, pointwiseMulFail)
			}

			// update := -eta * mHat / (sqrt(vHat) + epsilon)
			if _, err = tensor.Sqrt(vHats, tensor.UseUnsafe()); err != nil {
				return // TODO: rewrite this to use InvSqrt
			}

			if _, err = tensor.Add(eps, vHats, tensor.UseUnsafe()); err != nil {
				return
			}

			if _, err = tensor.Mul(eta, mHats, tensor.UseUnsafe()); err != nil {
				return errors.Wrap(err, pointwiseMulFail)
			}

			if _, err = tensor.Div(mHats, vHats, tensor.WithIncr(w)); err != nil {
				return
			}

			defer tensor.ReturnTensor(vHats)
			defer tensor.ReturnTensor(mHats)

			if _, err = tensor.Add(w, mHats, tensor.UseUnsafe()); err != nil {
				return errors.Wrap(err, addFail)
			}

			g.Zero()

		case *values.F32:
			g := grad.(*values.F32).Any()
			w := weights.(*values.F32).Any()
			v := cvv.(*values.F32).Any()
			mm := m.Any()

			l1reg := float32(s.l1reg)
			l2reg := float32(s.l2reg)
			batch := float32(s.batch)
			clip := float32(s.clip)
			beta1 := float32(s.beta1)
			beta2 := float32(s.beta2)
			eps := float32(s.eps)
			eta := float32(s.eta)

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

			newM := (beta1 * mm) + (1-beta1)*g
			newV := (beta2 * v) + (1-beta2)*g*g

			cached.Value, _ = values.AnyToScalar(newM)
			cachedD, _ := values.AnyToScalar(newV)
			cached.SetDeriv(cachedD)

			mHat := (1 / float32(correction1)) * newM
			vHat := (1 / float32(correction2)) * newV

			upd := -eta * mHat / (float32(math.Sqrt(float64(vHat))) + eps)
			w += upd

			*(weights.(*values.F32)) = values.F32(w)
			*(grad.(*values.F32)) = values.F32(0.0)
		case *values.F64:
			g := grad.(*values.F64).Any()
			w := weights.(*values.F64).Any()
			v := cvv.(*values.F64).Any()
			mm := m.Any()

			l1reg := s.l1reg
			l2reg := s.l2reg
			batch := s.batch
			clip := s.clip
			beta1 := s.beta1
			beta2 := s.beta2
			eps := s.eps
			eta := s.eta

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

			newM := (beta1 * mm) + (1-beta1)*g
			newV := (beta2 * v) + (1-beta2)*g*g

			cached.Value, _ = values.AnyToScalar(newM)
			cachedD, _ := values.AnyToScalar(newV)
			cached.SetDeriv(cachedD)

			mHat := (1 / correction1) * newM
			vHat := (1 / correction2) * newV

			upd := -eta * mHat / (math.Sqrt(vHat) + eps)
			w += upd

			*(weights.(*values.F64)) = values.F64(w)
			*(grad.(*values.F64)) = values.F64(0.0)

		default:
			err = errors.Errorf(gerrors.NYITypeFail, "AdamSolver", cvm)
			return
		}

	}
	return
}
