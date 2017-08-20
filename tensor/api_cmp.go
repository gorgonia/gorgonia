package tensor

import "github.com/pkg/errors"

// public API for comparison ops

// Lt performs a elementwise less than comparison (a < b). a and b can either be float64 or *Dense.
// It returns the same Tensor type as its input.
//
// If both operands are *Dense, shape is checked first.
// Even though the underlying data may have the same size (say (2,2) vs (4,1)), if they have different shapes, it will error out.
func Lt(a, b interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	var lter Lter
	var ok bool
	switch at := a.(type) {
	case Tensor:
		lter, ok = at.Engine().(Lter)
		switch bt := b.(type) {
		case Tensor:
			if !ok {
				if lter, ok = bt.Engine().(Lter); !ok {
					return nil, errors.Errorf("Neither operands have engines that support Lt")
				}
			}
			return lter.Lt(at, bt, opts...)
		default:
			if !ok {
				return nil, errors.Errorf("Engine does not support Lt")
			}
			return lter.LtScalar(at, bt, true, opts...)
		}
	default:
		switch bt := b.(type) {
		case Tensor:
			if lter, ok = bt.Engine().(Lter); !ok {
				return nil, errors.Errorf("Engine does not support Lt")
			}
			return lter.LtScalar(bt, at, false, opts...)
		default:
			return nil, errors.Errorf("Unable to perform Lt on %T and %T", a, b)
		}
	}
}

// Gt performs a elementwise greater than comparison (a > b). a and b can either be float64 or *Dense.
// It returns the same Tensor type as its input.
//
// If both operands are *Dense, shape is checked first.
// Even though the underlying data may have the same size (say (2,2) vs (4,1)), if they have different shapes, it will error out.
func Gt(a, b interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	var gter Gter
	var ok bool
	switch at := a.(type) {
	case Tensor:
		gter, ok = at.Engine().(Gter)
		switch bt := b.(type) {
		case Tensor:
			if !ok {
				if gter, ok = bt.Engine().(Gter); !ok {
					return nil, errors.Errorf("Neither operands have engines that support Gt")
				}
			}
			return gter.Gt(at, bt, opts...)
		default:
			if !ok {
				return nil, errors.Errorf("Engine does not support Gt")
			}
			return gter.GtScalar(at, bt, true, opts...)
		}
	default:
		switch bt := b.(type) {
		case Tensor:
			if gter, ok = bt.Engine().(Gter); !ok {
				return nil, errors.Errorf("Engine does not support Gt")
			}
			return gter.GtScalar(bt, at, false, opts...)
		default:
			return nil, errors.Errorf("Unable to perform Gt on %T and %T", a, b)
		}
	}
}

// Lte performs a elementwise less than eq comparison (a <= b). a and b can either be float64 or *Dense.
// It returns the same Tensor type as its input.
//
// If both operands are *Dense, shape is checked first.
// Even though the underlying data may have the same size (say (2,2) vs (4,1)), if they have different shapes, it will error out.
func Lte(a, b interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	var lteer Lteer
	var ok bool
	switch at := a.(type) {
	case Tensor:
		lteer, ok = at.Engine().(Lteer)
		switch bt := b.(type) {
		case Tensor:
			if !ok {
				if lteer, ok = bt.Engine().(Lteer); !ok {
					return nil, errors.Errorf("Neither operands have engines that support Lte")
				}
			}
			return lteer.Lte(at, bt, opts...)
		default:
			if !ok {
				return nil, errors.Errorf("Engine does not support Lte")
			}
			return lteer.LteScalar(at, bt, true, opts...)
		}
	default:
		switch bt := b.(type) {
		case Tensor:
			if lteer, ok = bt.Engine().(Lteer); !ok {
				return nil, errors.Errorf("Engine does not support Lte")
			}
			return lteer.LteScalar(bt, at, false, opts...)
		default:
			return nil, errors.Errorf("Unable to perform Lte on %T and %T", a, b)
		}
	}
}

// Gte performs a elementwise greater than eq comparison (a >= b). a and b can either be float64 or *Dense.
// It returns the same Tensor type as its input.
//
// If both operands are *Dense, shape is checked first.
// Even though the underlying data may have the same size (say (2,2) vs (4,1)), if they have different shapes, it will error out.
func Gte(a, b interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	var gteer Gteer
	var ok bool
	switch at := a.(type) {
	case Tensor:
		gteer, ok = at.Engine().(Gteer)
		switch bt := b.(type) {
		case Tensor:
			if !ok {
				if gteer, ok = bt.Engine().(Gteer); !ok {
					return nil, errors.Errorf("Neither operands have engines that support Gte")
				}
			}
			return gteer.Gte(at, bt, opts...)
		default:
			if !ok {
				return nil, errors.Errorf("Engine does not support Gte")
			}
			return gteer.GteScalar(at, bt, true, opts...)
		}
	default:
		switch bt := b.(type) {
		case Tensor:
			if gteer, ok = bt.Engine().(Gteer); !ok {
				return nil, errors.Errorf("Engine does not support Gte")
			}
			return gteer.GteScalar(bt, at, false, opts...)
		default:
			return nil, errors.Errorf("Unable to perform Gte on %T and %T", a, b)
		}
	}
}

// ElEq performs a elementwise equality comparison (a == b). a and b can either be float64 or *Dense.
// It returns the same Tensor type as its input.
//
// If both operands are *Dense, shape is checked first.
// Even though the underlying data may have the same size (say (2,2) vs (4,1)), if they have different shapes, it will error out.
func ElEq(a, b interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	var eleqer ElEqer
	var ok bool
	switch at := a.(type) {
	case Tensor:
		eleqer, ok = at.Engine().(ElEqer)
		switch bt := b.(type) {
		case Tensor:
			if !ok {
				if eleqer, ok = bt.Engine().(ElEqer); !ok {
					return nil, errors.Errorf("Neither operands have engines that support ElEq")
				}
			}
			return eleqer.ElEq(at, bt, opts...)
		default:
			if !ok {
				return nil, errors.Errorf("Engine does not support ElEq")
			}
			return eleqer.EqScalar(at, bt, true, opts...)
		}
	default:
		switch bt := b.(type) {
		case Tensor:
			if eleqer, ok = bt.Engine().(ElEqer); !ok {
				return nil, errors.Errorf("Engine does not support ElEq")
			}
			return eleqer.EqScalar(bt, at, false, opts...)
		default:
			return nil, errors.Errorf("Unable to perform ElEq on %T and %T", a, b)
		}
	}
}

// ElNe performs a elementwise equality comparison (a != b). a and b can either be float64 or *Dense.
// It returns the same Tensor type as its input.
//
// If both operands are *Dense, shape is checked first.
// Even though the underlying data may have the same size (say (2,2) vs (4,1)), if they have different shapes, it will error out.
func ElNe(a, b interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	var eleqer ElEqer
	var ok bool
	switch at := a.(type) {
	case Tensor:
		eleqer, ok = at.Engine().(ElEqer)
		switch bt := b.(type) {
		case Tensor:
			if !ok {
				if eleqer, ok = bt.Engine().(ElEqer); !ok {
					return nil, errors.Errorf("Neither operands have engines that support ElEq")
				}
			}
			return eleqer.ElNe(at, bt, opts...)
		default:
			if !ok {
				return nil, errors.Errorf("Engine does not support ElEq")
			}
			return eleqer.NeScalar(at, bt, true, opts...)
		}
	default:
		switch bt := b.(type) {
		case Tensor:
			if eleqer, ok = bt.Engine().(ElEqer); !ok {
				return nil, errors.Errorf("Engine does not support ElEq")
			}
			return eleqer.NeScalar(bt, at, false, opts...)
		default:
			return nil, errors.Errorf("Unable to perform ElEq on %T and %T", a, b)
		}
	}
}
