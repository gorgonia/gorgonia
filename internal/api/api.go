package gapi

import (
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia/execution/engines"
	"gorgonia.org/gorgonia/exprgraph"
	"gorgonia.org/gorgonia/internal/datatypes"
	gerrors "gorgonia.org/gorgonia/internal/errors"
	gtu "gorgonia.org/gorgonia/internal/tensorutils"
	stdops "gorgonia.org/gorgonia/ops/std"
	"gorgonia.org/tensor"
)

// Add adds two tensors.
func Add(a, b datatypes.Tensor) (retVal datatypes.Tensor, err error) {
	hybrid, ok := a.Engine().(engines.Hybrid)
	if !ok {
		hybrid, ok = b.Engine().(engines.Hybrid)
	}
	ctx := gtu.CtxFromEngines(a.Engine(), b.Engine())

	op := stdops.Add(a, b)
	if ok {
		// do symbolic stuff if there is an engine that supports symbolic things
		g := hybrid.Graph()
		if retVal, err = binopSymbolic(op, g, a, b); err != nil {
			return nil, errors.Wrapf(err, gerrors.SymbolicOpFail, "Add")
		}
	}

	// check if engine supports Add. If not, return
	if _, ok := a.Engine().(tensor.Adder); !ok {
		if _, ok := b.Engine().(tensor.Adder); !ok {
			return
		}
	}

	// do the values stuff
	at := exprgraph.T2T(a)
	bt := exprgraph.T2T(b)
	var ct tensor.Tensor
	switch {
	case at != nil && bt != nil && retVal != nil:
		// both a and b  are values, so we can "materialize" c
		ct = retVal.(*exprgraph.Node).Value() // Value will "lift" *header into a proper tensor.Dense
	case at != nil && bt != nil && retVal == nil:
		// we'd have to create one ourselves
		shp := a.Shape().Clone()
		dt := a.Dtype()
		ct = tensor.New(tensor.WithEngine(a.Engine()), tensor.WithShape(shp...), tensor.Of(dt))
	default:
		// one of a or b is not a value tensor
		return retVal, nil
	}

	if ct, err = op.PreallocDo(ctx, ct, at, bt); err != nil {
		return nil, err
	}
	if retVal == nil {
		retVal = ct // return not the Node, but the value.
	}

	// TODO:
	// queuer (backwards engine)

	return
}

func MatMul(a, b datatypes.Tensor) (retVal datatypes.Tensor, err error) {
	hybrid, ok := a.Engine().(engines.Hybrid)
	if !ok {
		hybrid, ok = b.Engine().(engines.Hybrid)
	}
	ctx := gtu.CtxFromEngines(a.Engine(), b.Engine())

	op := stdops.MatMul{}
	if ok {
		// do symbolic stuff
		g := hybrid.Graph()
		if retVal, err = binopSymbolic(op, g, a, b); err != nil {
			return nil, errors.Wrapf(err, gerrors.SymbolicOpFail, "MatMul")
		}
	}
	// do the values stuff
	at := exprgraph.T2T(a)
	bt := exprgraph.T2T(b)
	var ct tensor.Tensor

	switch {
	case at != nil && bt != nil && retVal != nil:
		// both a and b  are values, so we can "materialize" c
		ct = retVal.(*exprgraph.Node).Value() // Value will "lift" *header into a proper tensor.Dense
	case at != nil && bt != nil && retVal == nil:
		// we'd have to create one ourselves
		shp := tensor.Shape{a.Shape()[0], b.Shape()[1]}
		dt := a.Dtype()
		ct = tensor.New(tensor.WithEngine(a.Engine()), tensor.WithShape(shp...), tensor.Of(dt))
	default:
		// one of a or b is not a value tensor
		return retVal, nil
	}

	if ct, err = op.PreallocDo(ctx, ct, at, bt); err != nil {
		return nil, err
	}
	if retVal == nil {
		retVal = ct // return not the Node, but the value.
	}

	// TODO:
	// queuer (backwards engine)
	return
}

func ReduceAdd(xs []datatypes.Tensor, opts ...tensor.ConsOpt) (retVal datatypes.Tensor, err error) {
	switch len(xs) {
	case 0:
		return nil, nil
	case 1:
		return xs[0], nil
	default:
		retVal = xs[0]
		for i, x := range xs {
			if i == 0 {
				continue
			}
			if retVal, err = Add(retVal, x); err != nil {
				return nil, errors.Wrapf(err, "reduceAdd %dth term: %v", i, x)
			}
			for _, opt := range opts {
				opt(retVal.(tensor.Tensor))
			}
		}
	}
	return retVal, nil
}
