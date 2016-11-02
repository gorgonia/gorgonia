package gorgonia

import (
	"math"

	"github.com/chewxy/gorgonia/tensor/types"
	"github.com/pkg/errors"
)

/* BINARY OPERATOR */

type ʘBinaryOperator interface {
	isArith() bool
	binOpType() ʘBinaryOperatorType
	Do(bool, ...Value) (Value, error)
	String() string
}

type scalarBinOp struct {
	ʘBinaryOperatorType
	t Dtype
}

func (o scalarBinOp) binOpType() ʘBinaryOperatorType { return o.ʘBinaryOperatorType }
func (o scalarBinOp) isArith() bool                  { return o.ʘBinaryOperatorType.isArith() }
func (o scalarBinOp) String() string                 { return o.ʘBinaryOperatorType.String() }

func (o scalarBinOp) Do(same bool, vals ...Value) (retVal Value, err error) {
	if len(vals) != 2 {
		err = NewError(GraphError, "Executing a binary operation expects 2 inputs. Got %d instead", len(vals))
		return
	}

	a, aok := vals[0].(Scalar)
	b, bok := vals[1].(Scalar)

	if !aok || !bok {
		err = NewError(RuntimeError, "Expected both inputs to binOp %v to be Scalar. Got %v (%T) and %#v(%T) instead", o, vals[0], vals[0], vals[1], vals[1])
		return
	}

	if a.t != o.t {
		err = NewError(RuntimeError, "Type mismatch for a. Expected %v. Got %v instead", o.t, a.t)
		return
	}

	if b.t != o.t {
		err = NewError(RuntimeError, "Type mismatch for b. Expected %v. Got %v instead | %v(%T) |%v(%T)", o.t, b.t, a, a, b, b)
		return
	}

	var r interface{} // float or bool only plz
	switch o.t {
	case Float64:
		af := a.v.(float64)
		bf := b.v.(float64)
		switch o.ʘBinaryOperatorType {
		case addOpType:
			r = af + bf
		case subOpType:
			r = af - bf
		case mulOpType:
			r = af * bf
		case divOpType:
			r = af / bf
		case powOpType:
			r = math.Pow(af, bf)
		case ltOpType:
			r = af < bf
		case gtOpType:
			r = af > bf
		case lteOpType:
			r = af <= bf
		case gteOpType:
			r = af >= bf
		case eqOpType:
			r = af == bf
		case neOpType:
			r = af != bf
		default:
			err = nyi("scalarBinOp.Do() - Float64", o.ʘBinaryOperatorType)
		}

		if same && !o.isArith() {
			if r.(bool) {
				r = float64(1)
			} else {
				r = float64(0)
			}
		}
	case Float32:
		af := a.v.(float32)
		bf := b.v.(float32)
		switch o.ʘBinaryOperatorType {
		case addOpType:
			r = af + bf
		case subOpType:
			r = af - bf
		case mulOpType:
			r = af * bf
		case divOpType:
			r = af / bf
		case powOpType:
			r = float32(math.Pow(float64(af), float64(bf)))
		case ltOpType:
			r = af < bf
		case gtOpType:
			r = af > bf
		case lteOpType:
			r = af <= bf
		case gteOpType:
			r = af >= bf
		case eqOpType:
			r = af == bf
		case neOpType:
			r = af != bf
		default:
			err = nyi("scalarBinOp.Do() - Float32", o.ʘBinaryOperatorType)
		}

		if same && !o.isArith() {
			if r.(bool) {
				r = float32(1)
			} else {
				r = float32(0)
			}
		}
	default:
		err = nyi("scalarBinOp.Do() - Unhandled Scalar Type", o.t)
	}
	if err != nil {
		return
	}

	return anyToValue(r)
}

type tBinOp struct {
	ʘBinaryOperatorType
	tensorLeft bool
}

func (o tBinOp) binOpType() ʘBinaryOperatorType { return o.ʘBinaryOperatorType }
func (o tBinOp) String() string                 { return o.ʘBinaryOperatorType.String() }
func (o tBinOp) isArith() bool                  { return o.ʘBinaryOperatorType.isArith() }

func (o tBinOp) Do(same bool, inputs ...Value) (Value, error) {
	if same {
		o.do(inputs, types.AsSameType())
	}
	return o.do(inputs)
}

func (o tBinOp) UnsafeDo(inputs ...Value) (Value, error) { return o.do(inputs, types.UseUnsafe()) }
func (o tBinOp) UsePreallocDo(v Value, inputs ...Value) (retVal Value, err error) {
	t, ok := v.(Tensor)
	if !ok {
		err = NewError(RuntimeError, "Expected Tensor as preallocated value. Got %v of %T instead", v, v)
		return
	}

	reuse := t.Tensor
	return o.do(inputs, types.WithReuse(reuse))
}

func (o tBinOp) IncrDo(incr Value, inputs ...Value) (err error) {
	var reuse types.Tensor

	t, ok := incr.(Tensor)
	if ok {
		reuse = t.Tensor
		_, err = o.do(inputs, types.WithIncr(reuse))
		return
	}

	var retVal Value
	if retVal, err = o.do(inputs); err != nil {
		err = errors.Wrapf(err, doFail, o)
		return
	}

	add := newEBOByType(addOpType, incr.Type(), retVal.Type())
	if retVal, err = add.UnsafeDo(incr, retVal); err != nil {
		err = errors.Wrapf(err, unsafeDoFail, add)
		return
	}

	err = noIncrErr{retVal}
	return
}

func (o tBinOp) do(vals []Value, opts ...types.FuncOpt) (retVal Value, err error) {
	if len(vals) != 2 {
		err = NewError(GraphError, "Executing a binary operation expects 2 inputs. Got %d instead", len(vals))
		return
	}
	// typecheck the operands
	d0 := vals[0].Dtype()
	d1 := vals[1].Dtype()

	if d0 != d1 {
		err = NewError(RuntimeError, "Dtype mismatch for bin op: %v and %v", d0, d1)
		return
	}

	// extract the goddamn values
	var a, b interface{}
	if o.tensorLeft {
		if t, ok := vals[0].(Tensor); !ok {
			err = NewError(RuntimeError, "Expected left value to be Tensor. Got %v of %T instead", vals[0], vals[0])
			return
		} else {
			a = t.Tensor.Materialize()
		}

		switch other := vals[1].(type) {
		case Scalar:
			b = other.v
		case Tensor:
			b = other.Tensor.Materialize()
		default:
			err = nyi("tBinOp.do() - Unknown Other (R)", vals[1])
			return
		}
	} else {
		if t, ok := vals[1].(Tensor); !ok {
			err = NewError(RuntimeError, "Expected right value to be Tensor. Got %v of %T instead", vals[1], vals[1])
			return
		} else {
			b = t.Tensor.Materialize()
		}

		switch other := vals[0].(type) {
		case Scalar:
			a = other.v
		case Tensor:
			a = other.Tensor.Materialize()
		default:
			err = nyi("tBinOp.do() - Unknown Other (L)", vals[0])
			return
		}
	}

	var r interface{}
	switch d0 {
	case Float64:
		// get function, call function
		if o.isArith() {
			fn := tf64BinOps[o.ʘBinaryOperatorType]
			if fn == nil {
				err = NewError(RuntimeError, "nil function returned for %v", o.ʘBinaryOperatorType)
				return
			}
			if r, err = (*fn)(a, b, opts...); err != nil {
				return
			}
		} else {
			fn := tf64CmpOps[o.ʘBinaryOperatorType]
			if fn == nil {
				err = NewError(RuntimeError, "nil function returned for %v", o.ʘBinaryOperatorType)
				return
			}
			if r, err = (*fn)(a, b, opts...); err != nil {
				return
			}
		}
	case Float32:
		// get function, call function
		if o.isArith() {
			fn := tf32BinOps[o.ʘBinaryOperatorType]
			if fn == nil {
				err = NewError(RuntimeError, "nil function returned for %v", o.ʘBinaryOperatorType)
				return
			}
			if r, err = (*fn)(a, b, opts...); err != nil {
				return
			}
		} else {
			fn := tf32CmpOps[o.ʘBinaryOperatorType]
			if fn == nil {
				err = NewError(RuntimeError, "nil function returned for %v", o.ʘBinaryOperatorType)
				return
			}
			if r, err = (*fn)(a, b, opts...); err != nil {
				return
			}
		}
	default:
		err = nyi("tBinOp.do() Unknown Dtype", d0)
		return
	}

	return anyToValue(r)
}

type binDiffFn func(x, y, z, gradZ *Node) (Nodes, err error)

func addDiffExpr(x, y, z, gradZ *Node) (retVal Nodes, err error) {
	return Nodes{gradZ, gradZ}, nil
}

func addDiff(x, y, z *Node) (err error) {
	xdv := x.boundTo.(*dualValue)
	ydv := y.boundTo.(*dualValue)
	zdv := z.boundTo.(*dualValue)

	add := newElemBinOp(addOpType, x, z)

	var d Value
	if x.IsScalar() {
		if d, err = add.Do(xdv.d, zdv.d); err != nil {
			err = errors.Wrapf(err, doFail, add)
			return
		}
	} else {
		if d, err = add.UnsafeDo(xdv.d, zdv.d); err != nil {
			err = errors.Wrapf(err, unsafeDoFail, add)
			return
		}
	}

	if !add.ReturnsPtr() || x.IsScalar() {
		xdv.SetDeriv(d) // ignore sanity check error on purpose
	}

	add = newElemBinOp(addOpType, y, z)

	if y.IsScalar() {
		if d, err = add.Do(ydv.d, zdv.d); err != nil {
			err = errors.Wrapf(err, doFail, add)
			return
		}
	} else {
		if d, err = add.UnsafeDo(ydv.d, zdv.d); err != nil {
			err = errors.Wrapf(err, unsafeDoFail, add)
			return
		}
	}
	if !add.ReturnsPtr() || y.IsScalar() {
		ydv.SetDeriv(d) // ignore errors on purpose
	}

	return nil
}

func subDiffExpr(x, y, z, gradZ *Node) (retVal Nodes, err error) {
	var dzdy *Node
	if dzdy, err = Neg(gradZ); err == nil {
		WithGroupName(gradClust)(dzdy)
		retVal = Nodes{gradZ, dzdy}
	}
	return
}

func subDiff(x, y, z *Node) (err error) {
	xdv := x.boundTo.(*dualValue)
	ydv := y.boundTo.(*dualValue)
	zdv := z.boundTo.(*dualValue)

	sub := newElemBinOp(subOpType, y, z)
	add := newElemBinOp(addOpType, x, z)

	var d Value
	// dz/dy

	if y.IsScalar() {
		if d, err = sub.Do(ydv.d, zdv.d); err != nil {
			err = errors.Wrapf(err, doFail, sub)
			return
		}
	} else {
		if d, err = sub.UnsafeDo(ydv.d, zdv.d); err != nil {
			err = errors.Wrapf(err, unsafeDoFail, sub)
			return
		}
	}

	if !sub.ReturnsPtr() || y.IsScalar() {
		ydv.SetDeriv(d) // ignore errors on purpose
	}

	// dz/dx
	if x.IsScalar() {
		if d, err = add.Do(xdv.d, zdv.d); err != nil {
			err = errors.Wrapf(err, doFail, add)
			return
		}
	} else {
		if d, err = add.UnsafeDo(xdv.d, zdv.d); err != nil {
			err = errors.Wrapf(err, unsafeDoFail, add)
			return
		}
	}

	if !add.ReturnsPtr() || x.IsScalar() {
		xdv.SetDeriv(d) // ignore errors on purpose
	}

	return nil
}

func hadamardProdDiffExpr(x, y, z, gradZ *Node) (retVal Nodes, err error) {
	var dzdx, dzdy *Node
	if dzdx, err = HadamardProd(y, gradZ); err == nil {
		dzdy, err = HadamardProd(x, gradZ)
		WithGroupName(gradClust)(dzdx)
		WithGroupName(gradClust)(dzdy)
		retVal = Nodes{dzdx, dzdy}
	}
	return
}

func hadamardProdDiff(x, y, z *Node) (err error) {
	xdv := x.boundTo.(*dualValue)
	ydv := y.boundTo.(*dualValue)
	zdv := z.boundTo.(*dualValue)

	// mul := newElemBinOp(mulOpType, x, z)
	zdvdType := zdv.d.Type()

	//dzdx
	mul := newEBOByType(mulOpType, ydv.Value.Type(), zdvdType)
	err = mul.IncrDo(xdv.d, ydv.Value, zdv.d)
	if err != nil {
		var ver Valuer
		var ok bool
		if ver, ok = err.(Valuer); !ok {
			return
		}

		xdv.SetDeriv(ver.Value()) // ignore errors on purpose
	}

	//dzdy
	mul = newEBOByType(mulOpType, xdv.Value.Type(), zdvdType)
	err = mul.IncrDo(ydv.d, xdv.Value, zdv.d)
	if err != nil {
		var ver Valuer
		var ok bool
		if ver, ok = err.(Valuer); !ok {
			return
		}

		ydv.SetDeriv(ver.Value()) // ignore errors on purpose
	}

	return nil
}

func hadamardDivDiffExpr(x, y, z, gradZ *Node) (retVal Nodes, err error) {
	var dzdx, dzdy *Node
	if dzdx, err = HadamardDiv(gradZ, y); err == nil {
		WithGroupName(gradClust)(dzdx)
		if dzdy, err = HadamardDiv(z, y); err == nil {
			WithGroupName(gradClust)(dzdy)
			if dzdy, err = Neg(dzdy); err == nil {
				WithGroupName(gradClust)(dzdy)
				if dzdy, err = HadamardProd(dzdy, gradZ); err == nil {
					WithGroupName(gradClust)(dzdy)
					retVal = Nodes{dzdx, dzdy}
				}
			}
		}
	}
	return
}

func hadamardDivDiff(x, y, z *Node) (err error) {
	xdv := x.boundTo.(*dualValue)
	ydv := y.boundTo.(*dualValue)
	zdv := z.boundTo.(*dualValue)

	div := newEBOByType(divOpType, zdv.d.Type(), ydv.Value.Type())

	// dzdx = 1/y * dz
	err = div.IncrDo(xdv.d, zdv.d, ydv.Value)
	if err != nil {
		var ver Valuer
		var ok bool
		if ver, ok = err.(Valuer); !ok {
			return
		}

		xdv.SetDeriv(ver.Value()) // ignore errors on purpose
	}

	//dzdy = -x/y^2
	// TODO: investigate if this can be done (if no other node uses z):
	//		unsafe do : neg zdv.d
	// 		unsafe do : mul zdv.d, zdv.Value
	//		incr do   : <incr: ydv.d> div zdv.d, ydv.Value
	var d Value
	if d, err = div.Do(zdv.Value, ydv.Value); err != nil {
		err = errors.Wrapf(err, doFail, div)
		return
	}

	neg := newElemUnaryOp(negOpType, y)
	if d, err = neg.Do(d); err != nil {
		err = errors.Wrapf(err, doFail, neg)
		return
	}

	mul := newElemBinOp(mulOpType, z, y)
	err = mul.IncrDo(ydv.d, zdv.d, d)
	if err != nil {
		var ver Valuer
		var ok bool
		if ver, ok = err.(Valuer); !ok {
			return
		}

		ydv.SetDeriv(ver.Value()) // ignore errors on purpose
	}

	return nil
}

// TODO: go back in time, pay more attention to calculus class in high school and learn how to differentiate x^y
func hadamardPowDiffExpr(x, y, z, grad *Node) (retVal Nodes, err error) {
	return nil, NewError(NotYetImplemented, "hadamardPowDiffExpr not yet implemented")
}

func hadamardPowDiff(x, y, z *Node) (err error) {
	return NewError(NotYetImplemented, "hadamardPowDiff not yet implemented")
}

func nondiffBinOpExpr(x, y, z, grad *Node) (retVal Nodes, err error) {
	return nil, NewError(SymbDiffError, "Nondifferentiable")
}

func nondiffBinOp(x, y, z *Node) (err error) {
	return NewError(AutoDiffError, "Non differentiable")
}
