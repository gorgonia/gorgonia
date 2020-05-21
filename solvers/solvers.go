package solvers

import (
	"math"

	"github.com/chewxy/math32"
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia/values"
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
	Grad() (values.Value, error)
}

// Namer is anything that has a name
type Namer interface {
	Name() string
}

func newCachedDV(n ValueGrad, weights, grad values.Value, zero bool) (cached *dualValue, err error) {
	cached = new(dualValue)
	if cached.Value, err = CloneValue(weights); err != nil {
		if nm, ok := n.(Namer); ok {
			return nil, errors.Errorf("Failed to clone weights of %v", nm.Name())
		}
		return nil, errors.New("Failed to clone weights")
	}
	if cached.d, err = CloneValue(grad); err != nil {
		if nm, ok := n.(Namer); ok {
			return nil, errors.Errorf("Failed to clone grad of %v", nm.Name())
		}
		return nil, errors.New("Failed to clone grad")
	}
	if zero {
		cached.Value = ZeroValue(cached.Value)
		cached.d = ZeroValue(cached.d)
	}
	return
}

func extractWeightGrad(n ValueGrad) (weights, grad Value, err error) {
	weights = n.Value()
	if grad, err = n.Grad(); err != nil {
		if nm, ok := n.(Namer); ok {
			return weights, nil, errors.Wrapf(err, "No Grad found for %v", nm.Name())
		}
		return weights, nil, errors.Wrap(err, "No Grad found")
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

	cache []*values.Dual
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
		s.cache = make([]*values.Dual, len(model))
	}

	for i, n := range model {
		var weights, grad Value
		if weights, grad, err = extractWeightGrad(n); err != nil {
			return err
		}

		var cached *values.Dual
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
				if l2regs, err = tensor.Mul(l2reg, cw); err != nil {
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

	cache []*values.Dual
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
		s.cache = make([]*values.Dual, len(model))
	}

	for i, n := range model {
		var weights, grad Value
		if weights, grad, err = extractWeightGrad(n); err != nil {
			return err
		}

		var cached *values.Dual
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
	prevDV  []*values.Dual // dual value for xᵢ₋₁ step
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
		s.prevDV = make([]*values.Dual, len(model))
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

				nominator += valGradDiffscalarProd.Data().(float64)

				// ∥(Grad(F)(xᵢ) - Grad(F)(xᵢ₋₁))∥²
				gradDiffscalarProd, err := tensor.Contract(gradDiff, gradDiff, contractionAxes, contractionAxes)
				if err != nil {
					return errors.New("operationError, Contracting value / gradient difference")
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
		var weights, grad Value
		if weights, grad, err = extractWeightGrad(node); err != nil {
			return err
		}

		if false == firstRun {
			// return memory for the old dual value used in this iteration
			returnDV(s.prevDV[nodeNr])
		}
		var oldDV *values.Dual
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
