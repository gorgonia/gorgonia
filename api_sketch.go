package gorgonia

import (
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia/execution/engines"
	"gorgonia.org/gorgonia/exprgraph"
	gerrors "gorgonia.org/gorgonia/internal/errors"
	gtu "gorgonia.org/gorgonia/internal/tensorutils"
	stdops "gorgonia.org/gorgonia/ops/std"
	"gorgonia.org/tensor"
)

func Add(a, b Tensor) (retVal Tensor, err error) { return add(a, b) }

func MatMul(a, b Tensor) (retVal Tensor, err error) {
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
