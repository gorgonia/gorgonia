package gorgonia

import (
	"fmt"
	"log"
	"math"

	"github.com/chewxy/math32"
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

// Solver is anything that does gradient updates.
// The name solvers is stolen from Caffe. A much shorter name than GradientUpdaters
type Solver interface {
	Step([]ValueGrad) error
}

// ValueGrad is any type that has a value and a grad. This is used for Solvers
type ValueGrad interface {
	Valuer
	Grad() (Value, error)
}

// Namer is anything that has a name
type Namer interface {
	Name() string
}

// SolverOpt is a function that provides construction options for a Solver
type SolverOpt func(s Solver)

// WithL2Reg adds a L2 regularization parameter to the solver. By default, the solvers do not use any regularization param
func WithL2Reg(l2reg float64) SolverOpt {
	f := func(s Solver) {
		switch st := s.(type) {
		case *RMSPropSolver:
			st.l2reg = l2reg
			st.useL2Reg = true
		case *AdamSolver:
			st.l2reg = l2reg
			st.useL2Reg = true
		case *VanillaSolver:
			st.l2reg = l2reg
			st.useL2Reg = true
		case *Momentum:
			st.l2reg = l2reg
			st.useL2Reg = true
		case *AdamW:
			st.l2reg = l2reg
			st.useL2Reg = true
		}
	}
	return f
}

// WithL1Reg adds a L1 regularization parameter to the solver. By default, the solvers do not use any regularization param
func WithL1Reg(l1reg float64) SolverOpt {
	f := func(s Solver) {
		switch st := s.(type) {
		case *AdamSolver:
			st.l1reg = l1reg
			st.useL1Reg = true
		case *VanillaSolver:
			st.l1reg = l1reg
			st.useL1Reg = true
		case *Momentum:
			st.l1reg = l1reg
			st.useL1Reg = true
		case *AdamW:
			st.l1reg = l1reg
			st.useL1Reg = true
		}
	}
	return f
}

// WithBatchSize sets the batch size for the solver. Currently only Adam and Vanilla (basic SGD) has batch size support
func WithBatchSize(batch float64) SolverOpt {
	f := func(s Solver) {
		switch st := s.(type) {
		case *AdamSolver:
			st.batch = batch
		case *VanillaSolver:
			st.batch = batch
		case *Momentum:
			st.batch = batch
		case *AdamW:
			st.batch = batch
		}
	}
	return f
}

// WithEps sets the smoothing factor for the solver.
func WithEps(eps float64) SolverOpt {
	f := func(s Solver) {
		switch st := s.(type) {
		case *RMSPropSolver:
			st.eps = eps
		case *AdamSolver:
			st.eps = eps
		case *AdamW:
			st.ɛ = eps
		}
	}
	return f
}

// WithClip clips the gradient if it gets too crazy. By default all solvers do not have any clips attached
func WithClip(clip float64) SolverOpt {
	f := func(s Solver) {
		switch st := s.(type) {
		case *RMSPropSolver:
			st.clip = clip
			st.useClip = true
		case *AdamSolver:
			st.clip = clip
			st.useClip = true
		case *VanillaSolver:
			st.clip = clip
			st.useClip = true
		case *BarzilaiBorweinSolver:
			st.clip = clip
			st.useClip = true
		case *Momentum:
			st.clip = clip
			st.useClip = true
		case *AdamW:
			st.clip = clip
			st.useClip = true
		}
	}
	return f
}

// WithLearnRate sets the learn rate or step size for the solver.
func WithLearnRate(eta float64) SolverOpt {
	f := func(s Solver) {
		switch st := s.(type) {
		case *RMSPropSolver:
			st.eta = eta
		case *AdamSolver:
			st.eta = eta
		case *VanillaSolver:
			st.eta = eta
		case *BarzilaiBorweinSolver:
			st.eta = eta
		case *Momentum:
			st.eta = eta
		case *AdamW:
			st.η = eta
		}
	}
	return f
}

// WithBeta1 sets the beta1 param of the solver. Only works with Adam
func WithBeta1(beta1 float64) SolverOpt {
	f := func(s Solver) {
		switch st := s.(type) {
		case *AdamSolver:
			st.beta1 = beta1
		case *AdamW:
			st.β1 = beta1
		}
	}
	return f
}

// WithBeta2 sets the beta1 param of the solver. Only works with Adam
func WithBeta2(beta2 float64) SolverOpt {
	f := func(s Solver) {
		switch st := s.(type) {
		case *AdamSolver:
			st.beta2 = beta2
		case *AdamW:
			st.β2 = beta2
		}
	}
	return f
}

// WithRho sets the decay parameter of the RMSProp solver
func WithRho(rho float64) SolverOpt {
	f := func(s Solver) {
		switch st := s.(type) {
		case *RMSPropSolver:
			st.decay = rho
		case *AdamW:
			st.λ = rho
		}
	}
	return f
}

// WithMomentum sets the momentum of the solver. It is a no-op is the solver's type is not Momentum
func WithMomentum(momentum float64) SolverOpt {
	f := func(s Solver) {
		switch st := s.(type) {
		case *Momentum:
			st.momentum = momentum
		}
	}
	return f
}

// RMSPropSolver is a solver that implements Geoffrey Hinton's RMSProp gradient descent optimization algorithm.
// http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
type RMSPropSolver struct {
	decay float64 // decay rate/rho
	eps   float64 // smoothing factor
	l2reg float64 // l2 regularization
	clip  float64 // clip value
	eta   float64 // learn rate

	useClip, useL2Reg bool

	// unsettable
	cache []*dualValue
}

// NewRMSPropSolver creates an RMSProp solver with these default values:
//		eta (learn rate)	  : 0.001
//		eps (smoothing factor): 1e-8
//		rho (decay factor)    : 0.999
func NewRMSPropSolver(opts ...SolverOpt) *RMSPropSolver {
	s := &RMSPropSolver{
		decay: 0.999,
		eps:   1e-8,
		eta:   0.001,
	}

	for _, opt := range opts {
		opt(s)
	}
	return s
}

// Step steps through each node in the model and applies the RMSProp gradient descent algorithm on the value.
//
// This function will error out if the nodes do not have an associated Grad value.
func (s *RMSPropSolver) Step(model []ValueGrad) (err error) {
	if s.cache == nil {
		s.cache = make([]*dualValue, len(model))
	}

	for i, n := range model {
		var weights, grad Value
		if weights, grad, err = extractWeightGrad(n); err != nil {
			return err
		}

		var cached *dualValue
		if cached = s.cache[i]; cached == nil {
			if cached, err = newCachedDV(n, weights, grad, true); err != nil {
				return err
			}
			s.cache[i] = cached
		}

		cv := cached.Value
		// cw = cw*decay + (1-decay) * grad²
		switch cw := cv.(type) {
		case *tensor.Dense:
			var gt, gt2, w, regularized tensor.Tensor
			var decay, omdecay, stepSize, eps, l2reg, clip, negClip interface{}
			switch cw.Dtype() {
			case tensor.Float64:
				decay = s.decay
				omdecay = 1.0 - s.decay
				stepSize = -s.eta
				eps = s.eps
				l2reg = s.l2reg
				clip = s.clip
				negClip = -s.clip
			case tensor.Float32:
				decay = float32(s.decay)
				omdecay = float32(1.0 - s.decay)
				stepSize = float32(-s.eta)
				eps = float32(s.eps)
				l2reg = float32(s.l2reg)
				clip = float32(s.clip)
				negClip = float32(-s.clip)
			}

			gt = grad.(tensor.Tensor)
			if gt2, err = tensor.Square(gt); err != nil {
				return errors.Wrap(err, pointWiseSquareFail)
			}
			tensor.Mul(cw, decay, tensor.UseUnsafe())
			tensor.Mul(gt2, omdecay, tensor.UseUnsafe())
			tensor.Add(cw, gt2, tensor.UseUnsafe())
			defer returnTensor(gt2)

			if s.useClip {
				if _, err = tensor.Clamp(gt, negClip, clip, tensor.UseUnsafe()); err != nil {
					return errors.Wrap(err, clampFail)
				}
			}

			// regularize
			var upd tensor.Tensor
			if upd, err = tensor.Add(cw, eps); err != nil {
				return errors.Wrap(err, "Failed to carry Add()")
			}

			if _, err = tensor.InvSqrt(upd, tensor.UseUnsafe()); err != nil {
				return errors.Wrap(err, invSqrtFail)
			}
			if _, err = tensor.Mul(gt, stepSize, tensor.UseUnsafe()); err != nil {
				return errors.Wrap(err, pointWiseMulFail)
			}
			if _, err = tensor.Mul(upd, gt, tensor.UseUnsafe()); err != nil {
				return errors.Wrap(err, pointWiseMulFail)
			}

			// update
			w = weights.(*tensor.Dense)
			if s.useL2Reg {
				if regularized, err = tensor.Mul(w, l2reg); err != nil {
					return errors.Wrap(err, pointWiseMulFail)
				}
				if _, err = tensor.Sub(upd, regularized, tensor.UseUnsafe()); err != nil {
					return errors.Wrap(err, subFail)
				}
				defer returnTensor(regularized)
			}

			if _, err = tensor.Add(w, upd, tensor.UseUnsafe()); err != nil {
				return errors.Wrap(err, addFail)
			}
			defer returnTensor(upd)

			// zero all
			gt.Zero()

		case *F32:
			decay := float32(s.decay)
			omdecay := float32(1.0 - s.decay)
			stepSize := float32(s.eta)
			eps := float32(s.eps)
			l2reg := float32(s.l2reg)

			gs := grad.(*F32).any()
			c := cw.any()
			c = c*decay + omdecay*gs*gs

			cached.Value, _ = anyToScalar(c)

			w := weights.(*F32).any()
			upd := -stepSize*gs/math32.Sqrt(c+eps) - l2reg*w
			w += upd

			// because scalar values are copies, and not pointers, we have to actually re-update the dualValu in model[i]
			*(weights.(*F32)) = F32(w)
			*(grad.(*F32)) = F32(0.0)
		case *F64:
			decay := s.decay
			omdecay := 1.0 - s.decay
			stepSize := s.eta
			eps := s.eps
			l2reg := s.l2reg

			gs := grad.(*F64).any()
			c := cw.any()
			c = c*decay + omdecay*gs*gs

			cached.Value, _ = anyToScalar(c)

			w := weights.(*F64).any()
			upd := -stepSize*gs/math.Sqrt(c+eps) - l2reg*w
			w += upd

			// because scalar values are copies, and not pointers, we have to actually re-update the dualValu in model[i]
			*(weights.(*F64)) = F64(w)
			*(grad.(*F64)) = F64(0.0)
		default:
		}
		solverLogf("AFTER %1.1s", n)
	}
	return nil
}

// AdamSolver is the Adaptive Moment Estimation solver (basically RMSProp on steroids).
// Paper: http://arxiv.org/abs/1412.6980
//
// We overload the purpose of existing data structure of a *dualValue. However, instead of just holding a value and its derivative,
// the cache's *dualValues hold the Means of gradients (in .Value) and the variances of the gradients (in .d)
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
	cache []*dualValue
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
		s.cache = make([]*dualValue, len(model))
	}

	for i, n := range model {
		s.iter++
		correction1 := (1 - math.Pow(s.beta1, float64(s.iter))) // 1 - β1^t
		correction2 := (1 - math.Pow(s.beta2, float64(s.iter))) // 1 - β2^t

		var weights, grad Value
		if weights, grad, err = extractWeightGrad(n); err != nil {
			return err
		}

		var cached *dualValue
		if cached = s.cache[i]; cached == nil {
			if cached, err = newCachedDV(n, weights, grad, true); err != nil {
				return err
			}
			s.cache[i] = cached
		}

		cvm := cached.Value // means of gradients
		cvv := cached.d     // variances of gradients

		switch m := cvm.(type) {
		case *tensor.Dense:

			g := grad.(*tensor.Dense)
			w := weights.(*tensor.Dense)
			v := cvv.(*tensor.Dense)

			var l1reg, l2reg, clip, negClip, beta1, beta2, omβ1, omβ2, eps, onePerBatch, stepSize interface{}
			var sqrtC2 interface{}
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
				sqrtCorrection2 := math.Sqrt(correction2)
				sqrtC2 = float64(sqrtCorrection2)
				eps = float64(s.eps * correction2)
				onePerBatch = float64(1) / float64(s.batch)
				stepSize = float64((-s.eta * sqrtCorrection2 / correction1))
			case tensor.Float32:
				l1reg = float32(s.l1reg)
				l2reg = float32(s.l2reg)
				clip = float32(s.clip)
				negClip = -float32(s.clip)
				beta1 = float32(s.beta1)
				beta2 = float32(s.beta2)
				omβ1 = float32(1) - float32(s.beta1)
				omβ2 = float32(1) - float32(s.beta2)
				sqrtCorrection2 := math.Sqrt(correction2)
				sqrtC2 = float32(sqrtCorrection2)
				eps = float32(s.eps * correction2)
				onePerBatch = float32(1) / float32(s.batch)
				stepSize = float32((-s.eta * sqrtCorrection2 / correction1))
			}

			// prep the regularization of gradients
			if s.useL1Reg {
				if err = doL1Reg(w, g, l1reg); err != nil {
					return errors.Wrap(err, "l1reg of gradients failed")
				}
			}

			if s.useL2Reg {
				if err = doL2Reg(w, g, l2reg); err != nil {
					return errors.Wrap(err, "L2Reg of gradients failed")
				}
			}

			if s.batch > 1 {
				if _, err = tensor.Mul(g, onePerBatch, tensor.UseUnsafe()); err != nil {
					return errors.Wrap(err, pointWiseMulFail)
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

			// equation (1): m *= β1. `m` has been modified
			if _, err = tensor.Mul(m, beta1, tensor.UseUnsafe()); err != nil {
				return errors.Wrap(err, pointWiseMulFail)
			}

			// equation (1): m += g * (1 - β1). `m` has been modified. `g` hasn't.
			if _, err = tensor.Mul(g, omβ1, tensor.WithIncr(m)); err != nil {
				return errors.Wrap(err, pointWiseMulFail)
			}

			// equation (2): v *= β2. `v` has been modified.
			if _, err = tensor.Mul(v, beta2, tensor.UseUnsafe()); err != nil {
				return errors.Wrap(err, pointWiseMulFail)
			}

			// equation (2): v += g² * (1 - β2). `v` has been modified. `g` now contains the values of g².
			if err = addcmul(v, g, g, omβ2); err != nil {
				return errors.Wrap(err, "v += (g*g) * (1-β2)")
			}

			if fmt.Sprintf("%v", n) == "w1 :: Matrix float64" || fmt.Sprintf("%v", n) == "w1 :: Matrix float32" {
				log.Printf("m %v\n%v\nv %v\n%v", n, m, n, v)
			}

			// compute the denom

			denom, err := tensor.Sqrt(v, tensor.WithReuse(g))
			if err != nil {
				return errors.Wrap(err, pointWiseSquareFail)
			}
			// if _, err = tensor.Mul(denom, sqrtC2, tensor.UseUnsafe()); err != nil {
			// 	return errors.Wrap(err, pointWiseMulFail)
			// }
			_ = sqrtC2
			if _, err = tensor.Add(denom, eps, tensor.UseUnsafe()); err != nil {
				return errors.Wrap(err, "add")
			}
			if fmt.Sprintf("%v", n) == "w1 :: Matrix float64" || fmt.Sprintf("%v", n) == "w1 :: Matrix float32" {
				log.Printf("%v \nm\n%v\ndenom\n%v", n, m, denom)
			}

			// note:
			// denom = m/denom
			// `denom` is the same data structure as `g`
			// DO NOT REUSE `m`
			if _, err = tensor.Div(m, denom, tensor.WithReuse(denom)); err != nil {
				return errors.Wrap(err, "Div")
			}
			if fmt.Sprintf("%v", n) == "w1 :: Matrix float64" || fmt.Sprintf("%v", n) == "w1 :: Matrix float32" {
				log.Printf("%v \ndenom\n%v", n, denom)
			}

			if _, err = tensor.Mul(denom, stepSize, tensor.WithIncr(w)); err != nil {
				return errors.Wrap(err, pointWiseMulFail)
			}

			if fmt.Sprintf("%v", n) == "w1 :: Matrix float64" {
				log.Printf("cached.v\n%v\ncached.d\n%v\n\n", cached.Value, cached.d)
			}
			g.Zero()

			/*
				g := grad.(*tensor.Dense)
				w := weights.(*tensor.Dense)
				v := cvv.(*tensor.Dense)

				var l1reg, l2reg, clip, negClip, beta1, beta2, omβ1, omβ2, eps, negEta, onePerBatch interface{}
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
					negEta = -s.eta
					onePerBatch = float64(1) / s.batch
					correctionV1 = float64(1) / float64(correction1)
					correctionV2 = float64(1) / float64(correction2)
				case tensor.Float32:
					l1reg = float32(s.l1reg)
					l2reg = float32(s.l2reg)
					clip = float32(s.clip)
					negClip = -float32(s.clip)
					beta1 = float32(s.beta1)
					beta2 = float32(s.beta2)
					omβ1 = float32(1) - float32(s.beta1)
					omβ2 = float32(1) - float32(s.beta2)
					eps = float32(s.eps)
					negEta = -float32(s.eta)
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
						return errors.Wrap(err, pointWiseMulFail)
					}
					if _, err = tensor.Add(g, l1regs, tensor.UseUnsafe()); err != nil {
						return errors.Wrap(err, addFail)
					}
					defer returnTensor(l1regs)
				}

				if s.useL2Reg {
					var l2regs tensor.Tensor
					if l2regs, err = tensor.Mul(w, l2reg); err != nil {
						return errors.Wrap(err, pointWiseMulFail)
					}

					if _, err = tensor.Add(g, l2regs, tensor.UseUnsafe()); err != nil {
						return errors.Wrap(err, addFail)
					}

					defer returnTensor(l2regs)
				}

				if s.batch > 1 {
					if _, err = tensor.Mul(g, onePerBatch, tensor.UseUnsafe()); err != nil {
						return errors.Wrap(err, pointWiseMulFail)
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

				// equation(1): t1 = grad * (1 - β_1)
				t1 := g.Clone().(*tensor.Dense)
				if _, err = tensor.Mul(t1, omβ1, tensor.UseUnsafe()); err != nil {
					return errors.Wrap(err, pointWiseMulFail)
				}

				// equation(2): g = grad**2 * (1 - β_2)
				if _, err = tensor.Mul(g, g, tensor.UseUnsafe()); err != nil {
					return errors.Wrap(err, pointWiseMulFail)
				}
				if _, err = tensor.Mul(g, omβ2, tensor.UseUnsafe()); err != nil {
					return errors.Wrap(err, pointWiseMulFail)
				}

				// equation (1): cached = cached * beta1 + t1
				if _, err = tensor.Mul(m, beta1, tensor.WithIncr(t1), tensor.UseUnsafe()); err != nil {
					return errors.Wrap(err, pointWiseMulFail)
				}

				// equation (2): v = v * beta2 + g
				if _, err = tensor.Mul(v, beta2, tensor.WithIncr(g), tensor.UseUnsafe()); err != nil {
					return errors.Wrap(err, pointWiseMulFail)
				}

				if fmt.Sprintf("%v", n) == "w1 :: Matrix float64" {
					log.Printf("m %v\n%v\nv %v\n%v", n, t1, n, g)

				}

				defer returnTensor(m)
				defer returnTensor(v)
				cached.SetValue(t1)
				cached.SetDeriv(g.Clone().(*tensor.Dense))

				// now deal with the hats
				mHats := t1.Clone().(*tensor.Dense)
				vHats := g.Clone().(*tensor.Dense)

				if _, err = tensor.Mul(mHats, correctionV1, tensor.UseUnsafe()); err != nil {
					return errors.Wrap(err, pointWiseMulFail)
				}

				if _, err = tensor.Mul(vHats, correctionV2, tensor.UseUnsafe()); err != nil {
					return errors.Wrap(err, pointWiseMulFail)
				}

				// update := -eta * mHat / (sqrt(vHat) + epsilon)
				if _, err = tensor.Sqrt(vHats, tensor.UseUnsafe()); err != nil {
					return // TODO: rewrite this to use InvSqrt
				}

				if _, err = tensor.Add(vHats, eps, tensor.UseUnsafe()); err != nil {
					return
				}

				if _, err = tensor.Mul(mHats, negEta, tensor.UseUnsafe()); err != nil {
					return errors.Wrap(err, pointWiseMulFail)
				}

				if _, err = tensor.Div(mHats, vHats, tensor.UseUnsafe()); err != nil {
					return
				}

				defer returnTensor(vHats)
				defer returnTensor(mHats)

				if _, err = tensor.Add(w, mHats, tensor.UseUnsafe()); err != nil {
					return errors.Wrap(err, addFail)
				}

				if fmt.Sprintf("%v", n) == "w1 :: Matrix float64" {
					log.Printf("cached.v\n%v\ncached.d\n%v\n\n", cached.Value, cached.d)
				}

				g.Zero()

			*/
		case *F32:
			g := grad.(*F32).any()
			w := weights.(*F32).any()
			v := cvv.(*F32).any()
			mm := m.any()

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

			cached.Value, _ = anyToScalar(newM)
			cached.d, _ = anyToScalar(newV)

			mHat := (1 / float32(correction1)) * newM
			vHat := (1 / float32(correction2)) * newV

			upd := -eta * mHat / (float32(math.Sqrt(float64(vHat))) + eps)
			w += upd

			*(weights.(*F32)) = F32(w)
			*(grad.(*F32)) = F32(0.0)
		case *F64:
			g := grad.(*F64).any()
			w := weights.(*F64).any()
			v := cvv.(*F64).any()
			mm := m.any()

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

			cached.Value, _ = anyToScalar(newM)
			cached.d, _ = anyToScalar(newV)

			mHat := (1 / correction1) * newM
			vHat := (1 / correction2) * newV

			upd := -eta * mHat / (math.Sqrt(vHat) + eps)
			w += upd

			*(weights.(*F64)) = F64(w)
			*(grad.(*F64)) = F64(0.0)

		default:
			err = errors.Errorf(nyiTypeFail, "AdamSolver", cvm)
			return
		}

	}
	return
}

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
		var weights, grad Value
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
					return errors.Wrap(err, pointWiseMulFail)
				}

				if _, err = tensor.Add(g, l1regs, tensor.UseUnsafe()); err != nil {
					return errors.Wrap(err, addFail)
				}

				defer returnTensor(l1regs)
			}

			if s.useL2Reg {
				if l2regs, err = tensor.Mul(w, l2reg); err != nil {
					return errors.Wrap(err, pointWiseMulFail)
				}

				if _, err = tensor.Add(g, l2regs, tensor.UseUnsafe()); err != nil {
					return errors.Wrap(err, addFail)
				}

				defer returnTensor(l2regs)
			}

			if s.batch > 1 {
				if _, err = tensor.Mul(g, onePerBatch, tensor.UseUnsafe()); err != nil {
					return errors.Wrap(err, pointWiseMulFail)
				}
			}

			if s.useClip && s.clip > 0 {
				if _, err = tensor.Clamp(g, negClip, clip, tensor.UseUnsafe()); err != nil {
					return errors.Wrap(err, clampFail)
				}
			}

			if _, err = tensor.Mul(g, eta, tensor.UseUnsafe()); err != nil {
				return errors.Wrap(err, pointWiseMulFail)
			}

			if _, err = tensor.Add(w, g, tensor.UseUnsafe()); err != nil {
				return errors.Wrap(err, addFail)
			}

			g.Zero()

		case *F32:
			g := grad.(*F32).any()
			wv := w.any()

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

			*(weights.(*F32)) = F32(wv)
			*(grad.(*F32)) = F32(0.0)
		case *F64:
			g := grad.(*F64).any()
			wv := w.any()

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

			*(weights.(*F64)) = F64(wv)
			*(grad.(*F64)) = F64(0.0)
		default:
			return errors.Errorf(nyiFail, "VanillaSolver.step", w)
		}
	}
	return
}

// Momentum is the stochastic gradient descent optimizer with momentum item.
type Momentum struct {
	eta      float64 // learn rate
	momentum float64 // momentum
	clip     float64 // clip gradients
	l1reg    float64 // l1 regularization parameter
	l2reg    float64 // l2 regularization parameter
	batch    float64 // batch size

	useClip, useL1Reg, useL2Reg bool

	cache []*dualValue
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
		s.cache = make([]*dualValue, len(model))
	}

	for i, n := range model {
		var weights, grad Value
		if weights, grad, err = extractWeightGrad(n); err != nil {
			return err
		}

		var cached *dualValue
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
					return errors.Wrap(err, pointWiseMulFail)
				}

				if _, err = tensor.Add(g, l1regs, tensor.UseUnsafe()); err != nil {
					return errors.Wrap(err, addFail)
				}

				defer returnTensor(l1regs)
			}

			if s.useL2Reg {
				if l2regs, err = tensor.Mul(cw, l2reg); err != nil {
					return errors.Wrap(err, pointWiseMulFail)
				}

				if _, err = tensor.Add(g, l2regs, tensor.UseUnsafe()); err != nil {
					return errors.Wrap(err, addFail)
				}

				defer returnTensor(l2regs)
			}

			if s.batch > 1 {
				if _, err = tensor.Mul(g, onePerBatch, tensor.UseUnsafe()); err != nil {
					return errors.Wrap(err, pointWiseMulFail)
				}
			}

			if s.useClip && s.clip > 0 {
				if _, err = tensor.Clamp(g, negClip, clip, tensor.UseUnsafe()); err != nil {
					return errors.Wrap(err, clampFail)
				}
			}

			// momentum
			if _, err = tensor.Mul(g, eta, tensor.UseUnsafe()); err != nil {
				return errors.Wrap(err, pointWiseMulFail)
			}

			// cw * momentum
			if _, err = tensor.Mul(cw, momentum, tensor.UseUnsafe()); err != nil {
				return errors.Wrap(err, pointWiseMulFail)
			}

			//  cw * momentum - eta * grad
			if _, err = tensor.Add(cw, g, tensor.UseUnsafe()); err != nil {
				return errors.Wrap(err, pointWiseMulFail)
			}

			if _, err = tensor.Add(w, cw, tensor.UseUnsafe()); err != nil {
				return errors.Wrap(err, addFail)
			}

			g.Zero()

		case *F32:
			l1reg := float32(s.l1reg)
			l2reg := float32(s.l2reg)
			batch := float32(s.batch)
			clip := float32(s.clip)
			eta := float32(s.eta)
			momentum := float32(s.momentum)

			g := grad.(*F32).any()
			w := weights.(*F32).any()
			c := cw.any()

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

			*(weights.(*F32)) = F32(w)
			*(grad.(*F32)) = F32(0.0)
		case *F64:
			l1reg := s.l1reg
			l2reg := s.l2reg
			batch := s.batch
			clip := s.clip
			eta := s.eta
			momentum := s.momentum

			g := grad.(*F64).any()
			w := weights.(*F64).any()
			c := cw.any()

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

			*(weights.(*F64)) = F64(w)
			*(grad.(*F64)) = F64(0.0)
		default:
			return errors.Errorf(nyiFail, "Momentum.step", cv)
		}
	}
	return
}

// AdaGradSolver is the solver that does adaptive gradient descent. Read the paper: http://jmlr.org/papers/v12/duchi11a.html
type AdaGradSolver struct {
	eta   float64 // learn rate
	eps   float64 // smoothing factor
	l1Reg float64 // l1reg param
	l2reg float64 // l2reg param
	clip  float64 // clip at

	useL2Reg, useClip bool

	cache []*dualValue
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
		s.cache = make([]*dualValue, len(model))
	}

	for i, n := range model {
		var weights, grad Value
		if weights, grad, err = extractWeightGrad(n); err != nil {
			return err
		}

		var cached *dualValue
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
				return errors.Wrap(err, pointWiseSquareFail)
			}

			c = cw
			tensor.Add(c, g2, tensor.UseUnsafe())
			defer returnTensor(g2)

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
				return errors.Wrap(err, pointWiseMulFail)
			}

			if _, err = tensor.Mul(upd, g, tensor.UseUnsafe()); err != nil {
				return errors.Wrap(err, pointWiseMulFail)
			}

			// regularize
			w = weights.(*tensor.Dense)

			if s.useL2Reg {
				if regularized, err = tensor.Mul(w, l2reg); err != nil {
					return errors.Wrap(err, pointWiseMulFail)
				}

				if _, err = tensor.Sub(upd, regularized, tensor.UseUnsafe()); err != nil {
					return errors.Wrap(err, subFail)
				}

				defer returnTensor(regularized)
			}

			if _, err = tensor.Add(w, upd, tensor.UseUnsafe()); err != nil {
				return errors.Wrap(err, addFail)
			}
			defer returnTensor(upd)

			// zero all
			g.Zero()

		case *F32:
			var w, g, c float32

			l2reg := float32(s.l2reg)
			clip := float32(s.clip)
			eps := float32(s.eps)
			eta := float32(s.eta)

			c = cw.any()
			g = grad.(*F32).any()

			c += g * g

			if s.useClip {
				if g > clip {
					g = clip
				} else if g < -clip {
					g = -clip
				}
			}

			w = weights.(*F32).any()

			upd := -eta * g / math32.Sqrt(c+eps)

			if s.useL2Reg {
				upd -= w * l2reg
			}

			w += upd

			// because scalar values are copies, and not pointers, we have to actually re-update the dualValu in model[i]
			*(weights.(*F32)) = F32(w)
			*(grad.(*F32)) = F32(0.0)
		case *F64:
			var w, g, c float64

			l2reg := s.l2reg
			clip := s.clip
			eps := s.eps
			eta := s.eta

			c = cw.any()
			g = grad.(*F64).any()

			c += g * g

			if s.useClip {
				if g > clip {
					g = clip
				} else if g < -clip {
					g = -clip
				}
			}

			w = weights.(*F64).any()
			upd := -eta * g / math.Sqrt(c+eps)
			if s.useL2Reg {
				upd -= w * l2reg
			}

			w += upd

			// because scalar values are copies, and not pointers, we have to actually re-update the dualValu in model[i]
			*(weights.(*F64)) = F64(w)
			*(grad.(*F64)) = F64(0.0)

		default:
			return errors.Errorf(nyiFail, "Adagrad step", cv)
		}

	}

	return
}

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
	prevDV  []*dualValue // dual value for xᵢ₋₁ step
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
		s.prevDV = make([]*dualValue, len(model))
	}

	// Update the learning rate
	if false == firstRun {
		nominator := float64(0.0)
		denominator := float64(0.0)

		for nodeNr, node := range model {
			var weights, grad Value
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

				gOld, ok := s.prevDV[nodeNr].d.(*tensor.Dense)
				if !ok {
					return errors.Errorf("Expected a *tensor.Dense in %v. Got %T instead", node, s.prevDV[nodeNr].d)
				}

				valueDiff, err := tensor.Sub(w, wOld)
				defer returnTensor(valueDiff)
				if err != nil {
					return errors.Wrap(err, subFail)
				}

				gradDiff, err := tensor.Sub(g, gOld)
				defer returnTensor(gradDiff)
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
				defer returnTensor(valGradDiffscalarProd)

				nominator += valGradDiffscalarProd.Data().([]float64)[0]

				// ∥(Grad(F)(xᵢ) - Grad(F)(xᵢ₋₁))∥²
				gradDiffscalarProd, err := tensor.Contract(gradDiff, gradDiff, contractionAxes, contractionAxes)
				if err != nil {
					return errors.New("operationError, Contracting value / gradient difference")
				}
				defer returnTensor(gradDiffscalarProd)

				denominator += gradDiffscalarProd.Data().([]float64)[0]

			default:
				return errors.Errorf(nyiFail, "Barizai-Borwein step", w)
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
		var weights, grad Value
		if weights, grad, err = extractWeightGrad(node); err != nil {
			return err
		}

		if false == firstRun {
			// return memory for the old dual value used in this iteration
			returnDV(s.prevDV[nodeNr])
		}
		var oldDV *dualValue
		if oldDV, err = newCachedDV(node, weights, grad, false); err != nil {
			return err
		}
		s.prevDV[nodeNr] = oldDV
	}

	// Update the weights
	for _, node := range model {
		var weights, grad Value
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
			defer returnTensor(upd)

			if err != nil {
				return errors.Wrap(err, pointWiseMulFail)
			}

			if _, err = tensor.Sub(w, upd, tensor.UseUnsafe()); err != nil {
				return errors.Wrap(err, subFail)
			}

			g.Zero()

		default:
			return errors.Errorf(nyiFail, "Barizai-Borwein step", w)
		}
	}

	return nil
}

type adamwState struct {
	expMA   tensor.Tensor // exponential moving average
	expMASq tensor.Tensor
	denom   tensor.Tensor // a temporary tensor used for computation
}

// AdamW is a Adam-like solver where the weight decay regularization is decoupled.
// See also: https://arxiv.org/abs/1711.05101
type AdamW struct {
	η     float64 // learn rate
	ε     float64 // smoothing
	λ     float64 // weight decay
	ɛ     float64 // epsilon, a fudge factor
	β1    float64
	β2    float64
	clip  float64 // clip gradients to between -clip and +clip
	l1reg float64 // l1 regularization parameter
	l2reg float64 // l2 regularization parameter
	batch float64 // batch size

	useL1Reg, useL2Reg, useClip bool

	// unsettable
	iter   float64
	states map[*Node]*adamwState
}

func NewAdamW(opts ...SolverOpt) *AdamW {
	s := &AdamW{
		η:      0.001,
		ɛ:      1e-8,
		λ:      0.01,
		β1:     0.9,
		β2:     0.999,
		states: make(map[*Node]*adamwState),
	}
	for _, opt := range opts {
		opt(s)
	}
	return s
}

func (a *AdamW) Step(model []ValueGrad) (err error) {
	/*
	 In the rest of the algorithm, we will use the following symbols for explainersx:
	 	θ  - the parameter to be updated.
	        _t - at time step t
	        m  - the moving average (first moment)
	        v  - the second momennt (MA squared)
	*/
	a.iter++
	for _, n := range model {
		n := n.(*Node)
		var weights, grad Value
		if weights, grad, err = extractWeightGrad(n); err != nil {
			return err
		}
		w := weights.(tensor.Tensor)
		g := grad.(tensor.Tensor)

		st, ok := a.states[n]
		if !ok {
			st = new(adamwState)
			st.expMA = tensor.New(tensor.WithShape(grad.Shape().Clone()...), tensor.Of(grad.Dtype()))
			st.expMASq = tensor.New(tensor.WithShape(grad.Shape().Clone()...), tensor.Of(grad.Dtype()))
			st.denom = tensor.New(tensor.WithShape(grad.Shape().Clone()...), tensor.Of(grad.Dtype()))
			a.states[n] = st
		}

		var decay, a1, a2, b1, b2, b2sqrt, ss, eps interface{}
		var l1reg, l2reg, clip, negClip interface{}
		switch weights.Dtype() {
		case tensor.Float64:
			lr := a.η
			wd := a.λ
			β1 := a.β1
			β2 := a.β2
			it := a.iter
			eps = a.ɛ
			decay = 1.0 - lr*wd
			b1f := 1.0 - math.Pow(β1, it) // correction for beta
			b2f := 1.0 - math.Pow(β2, it)
			a1 = 1.0 - b1f
			a2 = 1.0 - b2f // note here b2f is not sqrt'd
			b1 = b1f
			b2 = b2f
			b2sqrt = math.Sqrt(b2f)
			ss = -(lr / b1f)

			l1reg = a.l1reg
			l2reg = a.l2reg
			clip = a.clip
			negClip = -a.clip
		case tensor.Float32:
			lr := float32(a.η)
			wd := float32(a.λ)
			β1 := float32(a.β1)
			β2 := float32(a.β2)
			it := float32(a.iter)
			eps = float32(a.ɛ)

			decay = float32(1.0) - lr*wd
			b1f := float32(1.0) - math32.Pow(β1, it) // correction for beta
			b2f := float32(1.0) - math32.Pow(β2, it)
			a1 = float32(1.0) - b1f
			a2 = float32(1.0) - b2f // note here b2f is not sqrt'd
			b1 = b1f
			b2 = b2f
			b2sqrt = math32.Sqrt(b2f)
			ss = -(lr / b1f)

			l1reg = a.l1reg
			l2reg = a.l2reg
			clip = a.clip
			negClip = -a.clip

		}
		// regularization of gradients

		if a.useL1Reg {
			if err = doL1Reg(w, g, l1reg); err != nil {
				return errors.Wrapf(err, "Failed to perform L1 regularization on the gradients of %v", n)
			}
		}
		if a.useL2Reg {
			if err = doL2Reg(w, g, l2reg); err != nil {
				return errors.Wrapf(err, "Failed to perform L2 regularization on the gradients of %v", n)
			}
		}
		if a.batch > 1 {
			if err = divBatch(g, a.batch); err != nil {
				return errors.Wrapf(err, "Failed to divide gradients by batch count of %v", n)
			}
		}
		if a.useClip && a.clip > 0 {
			if err = clipGrad(g, clip, negClip); err != nil {
				return errors.Wrapf(err, "Failed to clip gradients of %v to between %v and %v", n, clip, negClip)
			}
		}

		var gSq tensor.Tensor

		// θ_t =  (1 - ηλ)θ_t-1
		if w, err = tensor.Mul(w, decay); err != nil {
			return err
		}

		// m_t = = β_1*m_t-1 + (1-β_1)g
		if st.expMA, err = tensor.Mul(st.expMA, b1, tensor.UseUnsafe()); err != nil {
			return err
		}
		if _, err = tensor.Mul(g, a1, tensor.WithIncr(st.expMA)); err != nil {
			return err
		}

		// v_t = β_2*v_t-1 + (1 - β_2)g²
		if st.expMASq, err = tensor.Mul(st.expMASq, b2, tensor.UseUnsafe()); err != nil {
			return err
		}
		if gSq, err = tensor.Mul(g, g, tensor.UseUnsafe()); err != nil {
			return err
		}
		if _, err = tensor.Mul(gSq, a2, tensor.WithIncr(st.expMASq)); err != nil {
			return err
		}

		if st.denom, err = tensor.Sqrt(st.expMASq, tensor.WithReuse(st.denom)); err != nil {
			return err
		}
		if st.denom, err = tensor.Div(st.denom, b2sqrt, tensor.UseUnsafe()); err != nil {
			return err
		}
		if st.denom, err = tensor.Add(st.denom, eps, tensor.UseUnsafe()); err != nil {
			return err
		}

		if st.denom, err = tensor.Div(st.expMA, st.denom, tensor.WithReuse(st.denom)); err != nil {
			return err
		}

		if w, err = tensor.Mul(st.denom, ss, tensor.WithIncr(w)); err != nil {
			return err
		}

		g.(tensor.Tensor).Zero()
	}
	return nil
}
