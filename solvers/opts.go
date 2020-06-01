package solvers

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
		case *Momentum:
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
