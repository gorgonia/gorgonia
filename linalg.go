package gorgonia

import (
	"gorgonia.org/gorgonia/execution/engines"
	"gorgonia.org/gorgonia/exprgraph"
	gtu "gorgonia.org/gorgonia/internal/tensorutils"
	stdops "gorgonia.org/gorgonia/ops/std"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/tensor"
)

func MatMul[DT tensor.Num, T values.Value[DT]](a, b Tensor) (retVal Tensor, err error) {
	eng, ok := a.Engine().(engines.Hybrid)
	if !ok {
		eng, ok = b.Engine().(engines.Hybrid)
	}
	ctx := gtu.CtxFromEngines(a.Engine(), b.Engine())

	op := stdops.MatMul[DT, T]{}
	if ok {
		// do symbolic stuff
		if retVal, err = binopSymbolic[DT](op, eng, a, b); err != nil {
			return nil, err
		}
	}

	// check if engine supports MatMul. If not, return
	_, aok := a.Engine().Workhorse().(tensor.BLA[DT, T])
	_, bok := b.Engine().Workhorse().(tensor.BLA[DT, T])
	switch {
	case !aok && !bok:
		_, aok = a.Engine().Workhorse().BasicEng().(tensor.BLA[DT, T])
		_, bok = b.Engine().Workhorse().BasicEng().(tensor.BLA[DT, T])
		if !aok && !bok {
			return
		}
	default:

	}
	// do the values stuff
	at, aok := exprgraph.T2T[DT, T](a)
	bt, bok := exprgraph.T2T[DT, T](b)
	var ct T

	switch {
	case aok && bok && retVal != nil:
		// both a and b  are values, so we can "materialize" c
		rv := exprgraph.SymToVal[DT, T](retVal.(*exprgraph.Symbolic[DT])) // turn a Symbolic into a Value
		retVal = rv
		ct = rv.Value()
	case aok && bok && retVal == nil:
		// we'd have to create one ourselves
		shp := tensor.Shape{a.Shape()[0], b.Shape()[1]}
		ct = any(ct).(tensor.Aliker[T]).Alike(tensor.WithEngine(a.Engine()), tensor.WithShape(shp...))
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

	// check if engine is backwards (i.e. requires a queue)
	// if not, return.
	var q engines.Queueer[DT, T]
	q, ok = a.Engine().Workhorse().(engines.Queueer[DT, T])
	if !ok {
		q, ok = b.Engine().Workhorse().(engines.Queueer[DT, T])
	}
	if q != nil {
		// do queue stuff here
		err = q.Q(op, []Tensor{a, b}, retVal)
	}
	return
}
