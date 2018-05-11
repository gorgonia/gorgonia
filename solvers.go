package gorgonia

import (
	"math"

	"github.com/chewxy/math32"
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

// Solver is anything that does gradient updates.
// The name solvers is stolen from Caffe. A much shorter name than GradientUpdaters
type Solver interface {
	Step(Nodes) error
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
func (s *RMSPropSolver) Step(model Nodes) (err error) {
	if s.cache == nil {
		s.cache = make([]*dualValue, len(model))
	}

	for i, n := range model {
		solverLogf("BEFORE (%v %p) : %+1.1s", n, n, n.boundTo)
		dv, ok := n.boundTo.(*dualValue)
		if !ok {
			return errors.Errorf("Expected a *dualValue in %v (%x). Got %T instead", n, n.Hashcode(), n.boundTo)
		}

		var cached *dualValue
		if cached = s.cache[i]; cached == nil {
			if cached, err = dv.clone0(); err != nil {
				return errors.Wrap(err, "Failed to carry out clone0()")
			}
			s.cache[i] = cached
		}

		grad := dv.d
		weights := dv.Value

		cv := cached.Value
		// cw = cw*decay + (1-decay) * grad^2
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
			dv.Value, _ = anyToScalar(w)
			dv.d = zero(Float32)
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
			dv.Value, _ = anyToScalar(w)
			dv.d = zero(Float64)
		default:
		}
		solverLogf("AFTER (%v): %+1.1s", n, n.boundTo)
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
func (s *AdamSolver) Step(model Nodes) (err error) {
	if s.cache == nil {
		s.cache = make([]*dualValue, len(model))
	}

	s.iter++
	correction1 := (1 - math.Pow(s.beta1, float64(s.iter)))
	correction2 := (1 - math.Pow(s.beta2, float64(s.iter)))

	for i, n := range model {
		dv, ok := n.boundTo.(*dualValue)
		if !ok {
			return errors.Errorf("Expected a *dualValue in %v (%x). Got %T instead", n, n.Hashcode(), n.boundTo)
		}

		var cached *dualValue
		if cached = s.cache[i]; cached == nil {
			if cached, err = dv.clone0(); err != nil {
				return errors.Wrap(err, "Failed to carry clone0()")
			}
			s.cache[i] = cached
		}

		grad := dv.d
		weights := dv.Value

		cvm := cached.Value // means of gradients
		cvv := cached.d     // variances of gradients

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
					return errors.Wrap(err, pointWiseMulFail)
				}
				if _, err = tensor.Add(g, l1regs, tensor.UseUnsafe()); err != nil {
					return errors.Wrap(err, addFail)
				}
				defer returnTensor(l1regs)
			}

			if s.useL2Reg {
				var l2regs tensor.Tensor
				if l2regs, err = tensor.Mul(l2reg, w); err != nil {
					return errors.Wrap(err, pointWiseMulFail)
				}

				if _, err = tensor.Add(g, l2regs, tensor.UseUnsafe()); err != nil {
					return errors.Wrap(err, addFail)
				}

				defer returnTensor(l2regs)
			}

			if s.batch > 1 {
				if _, err = tensor.Mul(onePerBatch, g, tensor.UseUnsafe()); err != nil {
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
			//		(β_2 * v_t-1) + (1 - β_2)*(g_t)^2 .............	2

			// equation(1)
			t1 := g.Clone().(*tensor.Dense)
			if _, err = tensor.Mul(omβ1, t1, tensor.UseUnsafe()); err != nil {
				return errors.Wrap(err, pointWiseMulFail)
			}

			// equation(2)
			if _, err = tensor.Mul(g, g, tensor.UseUnsafe()); err != nil {
				return errors.Wrap(err, pointWiseMulFail)
			}
			if _, err = tensor.Mul(omβ2, g, tensor.UseUnsafe()); err != nil {
				return errors.Wrap(err, pointWiseMulFail)
			}

			// equation (1)
			if _, err = tensor.Mul(beta1, m, tensor.WithIncr(t1)); err != nil {
				return errors.Wrap(err, pointWiseMulFail)
			}

			// equation (2)
			if _, err = tensor.Mul(beta2, v, tensor.WithIncr(g)); err != nil {
				return errors.Wrap(err, pointWiseMulFail)
			}

			defer returnTensor(m)
			defer returnTensor(v)
			cached.SetValue(t1)
			cached.SetDeriv(g.Clone().(*tensor.Dense))

			// now deal with the hats
			mHats := t1.Clone().(*tensor.Dense)
			vHats := g.Clone().(*tensor.Dense)

			if _, err = tensor.Mul(correctionV1, mHats, tensor.UseUnsafe()); err != nil {
				return errors.Wrap(err, pointWiseMulFail)
			}

			if _, err = tensor.Mul(correctionV2, vHats, tensor.UseUnsafe()); err != nil {
				return errors.Wrap(err, pointWiseMulFail)
			}

			// update := -eta * mHat / (sqrt(vHat) + epsilon)
			if _, err = tensor.Sqrt(vHats, tensor.UseUnsafe()); err != nil {
				return // TODO: rewrite this to use InvSqrt
			}

			if _, err = tensor.Add(eps, vHats, tensor.UseUnsafe()); err != nil {
				return
			}

			if _, err = tensor.Mul(eta, mHats, tensor.UseUnsafe()); err != nil {
				return errors.Wrap(err, pointWiseMulFail)
			}

			if _, err = tensor.Div(mHats, vHats, tensor.WithIncr(w)); err != nil {
				return
			}

			defer returnTensor(vHats)
			defer returnTensor(mHats)

			if _, err = tensor.Add(w, mHats, tensor.UseUnsafe()); err != nil {
				return errors.Wrap(err, addFail)
			}

			g.Zero()

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

			dv.Value, _ = anyToScalar(w)
			dv.d = zero(Float32)
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

			dv.Value, _ = anyToScalar(w)
			dv.d = zero(Float64)

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
func (s *VanillaSolver) Step(model Nodes) (err error) {
	for _, n := range model {
		dv, ok := n.boundTo.(*dualValue)
		if !ok {
			return errors.Errorf("Expected a *dualValue in %v (%x). Got %T instead", n, n.Hashcode(), n.boundTo)
		}

		grad := dv.d
		weights := dv.Value

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
				if l2regs, err = tensor.Mul(l2reg, w); err != nil {
					return errors.Wrap(err, pointWiseMulFail)
				}

				if _, err = tensor.Add(g, l2regs, tensor.UseUnsafe()); err != nil {
					return errors.Wrap(err, addFail)
				}

				defer returnTensor(l2regs)
			}

			if s.batch > 1 {
				if _, err = tensor.Mul(onePerBatch, g, tensor.UseUnsafe()); err != nil {
					return errors.Wrap(err, pointWiseMulFail)
				}
			}

			if s.useClip && s.clip > 0 {
				if _, err = tensor.Clamp(g, negClip, clip, tensor.UseUnsafe()); err != nil {
					return errors.Wrap(err, clampFail)
				}
			}

			if _, err = tensor.Mul(eta, g, tensor.UseUnsafe()); err != nil {
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

			dv.Value, _ = anyToScalar(wv)
			dv.d = zero(Float32)
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

			dv.Value, _ = anyToScalar(wv)
			dv.d = zero(Float64)
		default:
			return errors.Errorf(nyiFail, "VanillaSolver.step", w)
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
func (s *AdaGradSolver) Step(model Nodes) (err error) {
	if s.cache == nil {
		s.cache = make([]*dualValue, len(model))
	}

	for i, n := range model {
		dv, ok := n.boundTo.(*dualValue)
		if !ok {
			return errors.Errorf("Expected a *dualValue in %v (%x). Got %T instead", n, n.Hashcode(), n.boundTo)
		}

		var cached *dualValue
		if cached = s.cache[i]; cached == nil {
			if cached, err = dv.clone0(); err != nil {
				return errors.Wrap(err, clone0Fail)
			}
			s.cache[i] = cached
		}

		grad := dv.d
		weights := dv.Value

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
			dv.Value, _ = anyToScalar(w)
			dv.d = zero(Float32)
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
			dv.Value, _ = anyToScalar(w)
			dv.d = zero(Float64)

		default:
			return errors.Errorf(nyiFail, "Adagrad step", cv)
		}

	}

	return
}

// Barzilai-Borwein performs Gradient Descent in steepest descend direction
// Solves 0 = F(x), by
// x_{i+1} = x_i - eta * Grad(F)(x_i)
// Where the learn rate eta is calculated by the Barzilai-Borwein method:
// eta(x_i) = <(x_i - x_{i-1}), (Grad(F)(x_i) - Grad(F)(x_{i-1}))> /
//                            ||(Grad(F)(x_i) - Grad(F)(x_{i-1}))||^2
// The input learn rate is used for the first iteration.
// TODO: Check out stochastic implementations, e.g. "Barzilai-Borwein Step Size for Stochastic Gradient Descent" https://arxiv.org/abs/1605.04131
type BarzilaiBorweinSolver struct {
	eta     float64 // initial learn rate
	clip    float64 // clip value
	useClip bool
	prevDV  []*dualValue // dual value for x_{i-1} step
}

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
func (s *BarzilaiBorweinSolver) Step(model Nodes) error {

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
			dv, ok := node.boundTo.(*dualValue)
			if !ok {
				return errors.Errorf("Expected a *dualValue in %v (%x). Got %T instead", node, node.Hashcode(), node.boundTo)
			}

			weights := dv.Value
			grad := dv.d

			switch w := weights.(type) {
			case *tensor.Dense:
				g, ok := grad.(*tensor.Dense)
				if !ok {
					return errors.Errorf("Expected a *tensor.Dense in %v (%x). Got %T instead", node, node.Hashcode(), dv)
				}

				wOld, ok := s.prevDV[nodeNr].Value.(*tensor.Dense)
				if !ok {
					return errors.Errorf("Expected a *tensor.Dense in %v (%x). Got %T instead", node, node.Hashcode(), dv)
				}

				gOld, ok := s.prevDV[nodeNr].d.(*tensor.Dense)
				if !ok {
					return errors.Errorf("Expected a *tensor.Dense in %v (%x). Got %T instead", node, node.Hashcode(), dv)
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

				// <(x_i - x_{i-1}), (Grad(F)(x_i) - Grad(F)(x_{i-1}))>

				// Scalar Product == Total tensor contraction
				dims := valueDiff.Dims()
				contractionAxes := make([]int, dims, dims)
				for axis := 0; axis < len(contractionAxes); axis++ {
					contractionAxes[axis] = axis
				}

				valGradDiffscalarProd, err := tensor.Contract(valueDiff, gradDiff, contractionAxes, contractionAxes)
				if err != nil {
					return errors.New("operationError")
				}
				defer returnTensor(valGradDiffscalarProd)

				nominator += valGradDiffscalarProd.Data().(float64)

				// ||(Grad(F)(x_i) - Grad(F)(x_{i-1}))||^2
				gradDiffscalarProd, err := tensor.Contract(gradDiff, gradDiff, contractionAxes, contractionAxes)
				if err != nil {
					return errors.New("operationError")
				}
				defer returnTensor(gradDiffscalarProd)

				denominator += gradDiffscalarProd.Data().(float64)

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
		dv, ok := node.boundTo.(*dualValue)
		if !ok {
			return errors.Errorf("Expected a *dualValue in %v (%x). Got %T instead", node, node.Hashcode(), node.boundTo)
		}

		if false == firstRun {
			// return memory for the old dual value used in this iteration
			returnDV(s.prevDV[nodeNr])
		}

		oldDV, err := dv.Clone()
		if err != nil {
			return errors.Wrap(err, cloneFail)
		}
		s.prevDV[nodeNr] = oldDV.(*dualValue)
	}

	// Update the weights
	for _, node := range model {
		dv, ok := node.boundTo.(*dualValue)
		if !ok {
			return errors.Errorf("Expected a *dualValue in %v (%x). Got %T instead", node, node.Hashcode(), node.boundTo)
		}

		weights := dv.Value
		grad := dv.d

		switch w := weights.(type) {
		case *tensor.Dense:
			g, ok := grad.(*tensor.Dense)
			if !ok {
				return errors.Errorf("Expected a *tensor.Dense in %v (%x). Got %T instead", node, node.Hashcode(), dv)
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
