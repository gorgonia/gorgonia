package gorgonia

import (
	"math"

	"github.com/chewxy/gorgonia/tensor"
	"github.com/chewxy/math32"
	"github.com/pkg/errors"
)

type incrDoerBinOp interface {
	IncrDo(v Value, retSame bool, inputs ...Value) error
}
type usePreallocDoerBinOp interface {
	UsePreallocDo(v Value, retSame bool, inputs ...Value) (retVal Value, err error)
}
type unsafeDoerBinOp interface {
	UnsafeDo(retSame bool, inputs ...Value) (Value, error)
}

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

func (o scalarBinOp) Arity() int                     { return 2 }
func (o scalarBinOp) binOpType() ʘBinaryOperatorType { return o.ʘBinaryOperatorType }
func (o scalarBinOp) isArith() bool                  { return o.ʘBinaryOperatorType.isArith() }
func (o scalarBinOp) String() string                 { return o.ʘBinaryOperatorType.String() }

func (o scalarBinOp) Do(same bool, vals ...Value) (retVal Value, err error) {
	if err = checkArity(o, len(vals)); err != nil {
		return
	}

	at := TypeOf(vals[0])
	bt := TypeOf(vals[1])
	if !at.Eq(bt) {
		err = errors.Errorf("Type Mismatch: %v != %v", at, bt)
		return
	}

	var r interface{} // float or bool only plz
	switch a := vals[0].(type) {
	case F64:
		b := vals[1].(F64)
		switch o.ʘBinaryOperatorType {
		case addOpType:
			r = a + b
		case subOpType:
			r = a - b
		case mulOpType:
			r = a * b
		case divOpType:
			r = a / b
		case powOpType:
			r = F64(math.Pow(float64(a), float64(b)))
		case ltOpType:
			r = a < b
		case gtOpType:
			r = a > b
		case lteOpType:
			r = a <= b
		case gteOpType:
			r = a >= b
		case eqOpType:
			r = a == b
		case neOpType:
			r = a != b
		default:
			err = errors.Errorf(nyiFail, "scalarBinOp.Do() - Float64", o.ʘBinaryOperatorType)
		}

		if same && !o.isArith() {
			if r.(bool) {
				r = F64(1)
			} else {
				r = F64(0)
			}
		}

	case F32:
		b := vals[1].(F32)
		switch o.ʘBinaryOperatorType {
		case addOpType:
			r = a + b
		case subOpType:
			r = a - b
		case mulOpType:
			r = a * b
		case divOpType:
			r = a / b
		case powOpType:
			r = F32(math32.Pow(float32(a), float32(b)))
		case ltOpType:
			r = a < b
		case gtOpType:
			r = a > b
		case lteOpType:
			r = a <= b
		case gteOpType:
			r = a >= b
		case eqOpType:
			r = a == b
		case neOpType:
			r = a != b
		default:
			err = errors.Errorf(nyiFail, "scalarBinOp.Do() - Float64", o.ʘBinaryOperatorType)
		}

		if same && !o.isArith() {
			if r.(bool) {
				r = F32(1)
			} else {
				r = F32(0)
			}
		}

	case I:
		b := vals[1].(I)
		switch o.ʘBinaryOperatorType {
		case addOpType:
			r = a + b
		case subOpType:
			r = a - b
		case mulOpType:
			r = a * b
		case divOpType:
			r = a / b
		// case powOpType:
		// 	r = math.Pow(a, b)
		case ltOpType:
			r = a < b
		case gtOpType:
			r = a > b
		case lteOpType:
			r = a <= b
		case gteOpType:
			r = a >= b
		case eqOpType:
			r = a == b
		case neOpType:
			r = a != b
		default:
			err = errors.Errorf(nyiFail, "scalarBinOp.Do() - Float64", o.ʘBinaryOperatorType)
		}

		if same && !o.isArith() {
			if r.(bool) {
				r = I(1)
			} else {
				r = I(0)
			}
		}
	case I32:
		b := vals[1].(I32)
		switch o.ʘBinaryOperatorType {
		case addOpType:
			r = a + b
		case subOpType:
			r = a - b
		case mulOpType:
			r = a * b
		case divOpType:
			r = a / b
		// case powOpType:
		// 	r = math.Pow(a, b)
		case ltOpType:
			r = a < b
		case gtOpType:
			r = a > b
		case lteOpType:
			r = a <= b
		case gteOpType:
			r = a >= b
		case eqOpType:
			r = a == b
		case neOpType:
			r = a != b
		default:
			err = errors.Errorf(nyiFail, "scalarBinOp.Do() - Float64", o.ʘBinaryOperatorType)
		}

		if same && !o.isArith() {
			if r.(bool) {
				r = I32(1)
			} else {
				r = I32(0)
			}
		}
	case I64:
		b := vals[1].(I64)
		switch o.ʘBinaryOperatorType {
		case addOpType:
			r = a + b
		case subOpType:
			r = a - b
		case mulOpType:
			r = a * b
		case divOpType:
			r = a / b
		// case powOpType:
		// 	r = math.Pow(a, b)
		case ltOpType:
			r = a < b
		case gtOpType:
			r = a > b
		case lteOpType:
			r = a <= b
		case gteOpType:
			r = a >= b
		case eqOpType:
			r = a == b
		case neOpType:
			r = a != b
		default:
			err = errors.Errorf(nyiFail, "scalarBinOp.Do() - Float64", o.ʘBinaryOperatorType)
		}

		if same && !o.isArith() {
			if r.(bool) {
				r = I64(1)
			} else {
				r = I64(0)
			}
		}
	case U8:
		b := vals[1].(U8)
		switch o.ʘBinaryOperatorType {
		case addOpType:
			r = a + b
		case subOpType:
			r = a - b
		case mulOpType:
			r = a * b
		case divOpType:
			r = a / b
		// case powOpType:
		// 	r = math.Pow(a, b)
		case ltOpType:
			r = a < b
		case gtOpType:
			r = a > b
		case lteOpType:
			r = a <= b
		case gteOpType:
			r = a >= b
		case eqOpType:
			r = a == b
		case neOpType:
			r = a != b
		default:
			err = errors.Errorf(nyiFail, "scalarBinOp.Do() - Float64", o.ʘBinaryOperatorType)
		}

		if same && !o.isArith() {
			if r.(bool) {
				r = U8(1)
			} else {
				r = U8(0)
			}
		}
	case B:
		b := vals[1].(B)
		switch o.ʘBinaryOperatorType {
		case eqOpType:
			r = a == b
		case neOpType:
			r = a != b
		default:
			err = errors.Errorf(nyiFail, "scalarBinOp.Do() - Float64", o.ʘBinaryOperatorType)
		}

	default:
		err = errors.Errorf(nyiFail, "scalarBinOp.Do() - Unhandled Scalar Type", o.t)
	}

	if err != nil {
		return
	}

	retVal, _ = anyToScalar(r)
	return
}

type tBinOp struct {
	ʘBinaryOperatorType
	tensorLeft bool
}

func (o tBinOp) Arity() int                     { return 2 }
func (o tBinOp) binOpType() ʘBinaryOperatorType { return o.ʘBinaryOperatorType }
func (o tBinOp) String() string                 { return o.ʘBinaryOperatorType.String() }
func (o tBinOp) isArith() bool                  { return o.ʘBinaryOperatorType.isArith() }

func (o tBinOp) Do(same bool, inputs ...Value) (Value, error) {
	if same {
		return o.do(inputs, tensor.AsSameType())
	}
	return o.do(inputs)
}

func (o tBinOp) UnsafeDo(retSame bool, inputs ...Value) (Value, error) {
	if retSame {
		return o.do(inputs, tensor.AsSameType(), tensor.UseUnsafe())
	}
	return o.do(inputs, tensor.UseUnsafe())
}
func (o tBinOp) UsePreallocDo(v Value, retSame bool, inputs ...Value) (retVal Value, err error) {
	t, ok := v.(tensor.Tensor)
	if !ok {
		return nil, errors.Errorf("Expected Tensor as preallocated value. Got %v of %T instead", v, v)
	}

	reuse := t
	if retSame {
		return o.do(inputs, tensor.WithReuse(reuse), tensor.AsSameType())
	}
	return o.do(inputs, tensor.WithReuse(reuse))
}

func (o tBinOp) IncrDo(incr Value, retSame bool, inputs ...Value) (err error) {
	reuse, ok := incr.(tensor.Tensor)
	if ok {
		_, err = o.do(inputs, tensor.WithIncr(reuse))
		return
	}

	var retVal Value
	if retSame {
		if retVal, err = o.do(inputs, tensor.AsSameType()); err != nil {
			return errors.Wrapf(err, doFail, o)
		}
	} else {
		if retVal, err = o.do(inputs); err != nil {
			return errors.Wrapf(err, doFail, o)
		}

	}

	add := newEBOByType(addOpType, TypeOf(incr), TypeOf(retVal))
	if retVal, err = add.UnsafeDo(incr, retVal); err != nil {
		return errors.Wrapf(err, unsafeDoFail, add)
	}

	err = noIncrErr{retVal}
	return
}

func (o tBinOp) do(vals []Value, opts ...tensor.FuncOpt) (retVal Value, err error) {
	if err = checkArity(o, len(vals)); err != nil {
		return
	}

	// typecheck the operands
	d0 := DtypeOf(vals[0])
	d1 := DtypeOf(vals[1])

	if d0 != d1 {
		return nil, errors.Errorf("Dtype mismatch for bin op: %v and %v", d0, d1)
	}

	// extract the goddamn values
	var a, b interface{}
	if o.tensorLeft {
		t, ok := vals[0].(tensor.Tensor)
		if !ok {
			return nil, errors.Errorf("Expected left value to be Tensor. Got %v of %T instead", vals[0], vals[0])
		}
		a = t.Materialize()

		switch other := vals[1].(type) {
		case F64:
			b = float64(other)
		case F32:
			b = float32(other)
		case tensor.Tensor:
			b = other.Materialize()
		default:
			return nil, errors.Errorf(nyiFail, "tBinOp.do()", vals[1])
		}
	} else {
		t, ok := vals[1].(tensor.Tensor)
		if !ok {
			return nil, errors.Errorf("Expected right value to be Tensor. Got %v of %T instead", vals[1], vals[1])
		}
		b = t.Materialize()

		switch other := vals[0].(type) {
		case F64:
			a = float64(other)
		case F32:
			a = float32(other)
		case tensor.Tensor:
			a = other.Materialize()
		default:
			return nil, errors.Errorf(nyiFail, "tBinOp.do()", vals[1])
		}
	}

	if o.isArith() {
		fn := binOps[o.ʘBinaryOperatorType]
		if fn == nil {
			return nil, errors.Errorf("nil function returned for %v", o.ʘBinaryOperatorType)
		}
		retVal, err = (*fn)(a, b, opts...)
	} else {
		fn := cmpOps[o.ʘBinaryOperatorType]
		if fn == nil {
			return nil, errors.Errorf("nil function returned for %v", o.ʘBinaryOperatorType)
		}
		retVal, err = (*fn)(a, b, opts...)

	}
	return
}

// type binDiffFn func(x, y, z, gradZ *Node) (Nodes, err error)

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
			return errors.Wrapf(err, doFail, add)
		}
	} else {
		if d, err = add.UnsafeDo(xdv.d, zdv.d); err != nil {
			return errors.Wrapf(err, unsafeDoFail, add)
		}
	}
	xdv.SetDeriv(d)

	add = newElemBinOp(addOpType, y, z)
	if y.IsScalar() {
		if d, err = add.Do(ydv.d, zdv.d); err != nil {
			return errors.Wrapf(err, doFail, add)
		}
	} else {
		if d, err = add.UnsafeDo(ydv.d, zdv.d); err != nil {
			return errors.Wrapf(err, unsafeDoFail, add)
		}
	}
	ydv.SetDeriv(d) // ignore errors on purpose

	return nil
}

func subDiffExpr(x, y, z, gradZ *Node) (retVal Nodes, err error) {
	var dzdy *Node
	if dzdy, err = Neg(gradZ); err == nil {
		WithGroupName(gradClust)(dzdy)
		WithGroupName(gradClust)(gradZ)
		retVal = Nodes{gradZ, dzdy}
	} else {
		return nil, errors.Wrap(err, "Failed to carry Neg()")
	}
	return
}

func subDiff(x, y, z *Node) (err error) {
	xdv := x.boundTo.(*dualValue)
	ydv := y.boundTo.(*dualValue)
	zdv := z.boundTo.(*dualValue)

	sub := newEBOByType(subOpType, y.t, z.t)
	add := newEBOByType(addOpType, x.t, z.t)

	var d Value

	// dz/dy
	if y.IsScalar() {
		if d, err = sub.Do(ydv.d, zdv.d); err != nil {
			return errors.Wrapf(err, doFail, sub)
		}
	} else {
		if d, err = sub.UnsafeDo(ydv.d, zdv.d); err != nil {
			return errors.Wrapf(err, unsafeDoFail, sub)
		}
	}
	ydv.SetDeriv(d) // ignore errors on purpose

	// dz/dx
	if x.IsScalar() {
		if d, err = add.Do(xdv.d, zdv.d); err != nil {
			return errors.Wrapf(err, doFail, add)
		}
	} else {
		if d, err = add.UnsafeDo(xdv.d, zdv.d); err != nil {
			return errors.Wrapf(err, unsafeDoFail, add)
		}
	}
	xdv.SetDeriv(d) // ignore errors on purpose

	return nil
}

func hadamardProdDiffExpr(x, y, z, gradZ *Node) (retVal Nodes, err error) {
	var dzdx, dzdy *Node
	dzdx, err = HadamardProd(y, gradZ)
	if dzdx, err = HadamardProd(y, gradZ); err == nil {
		dzdy, err = HadamardProd(x, gradZ)
		if err != nil {
			return nil, errors.Wrap(err, "Failed to carry HadamardProd()")
		}
		WithGroupName(gradClust)(dzdx)
		WithGroupName(gradClust)(dzdy)
		retVal = Nodes{dzdx, dzdy}
		return
	} else {
		return nil, errors.Wrap(err, "Failed to carry HadamardProd()")
	}
}

func hadamardProdDiff(x, y, z *Node) (err error) {
	xdv := x.boundTo.(*dualValue)
	ydv := y.boundTo.(*dualValue)
	zdv := z.boundTo.(*dualValue)

	var mul elemBinOp
	zdvdType := TypeOf(zdv.d)

	if x.isConstant() {
		goto dzdy
	}

	//dzdx
	mul = newEBOByType(mulOpType, TypeOf(ydv.Value), zdvdType)
	err = mul.IncrDo(xdv.d, ydv.Value, zdv.d)
	if err != nil {
		var ver Valuer
		var ok bool
		if ver, ok = err.(Valuer); !ok {
			return errors.Wrap(err, "IncrDo xdv.d failed")
		}

		xdv.SetDeriv(ver.Value()) // ignore errors on purpose
	}

dzdy:
	if y.isConstant() {
		goto end
	}

	mul = newEBOByType(mulOpType, TypeOf(xdv.Value), zdvdType)
	err = mul.IncrDo(ydv.d, xdv.Value, zdv.d)
	if err != nil {
		var ver Valuer
		var ok bool
		if ver, ok = err.(Valuer); !ok {
			return errors.Wrap(err, "IncrDo ydv.d failed")
		}

		ydv.SetDeriv(ver.Value()) // ignore errors on purpose
	}

end:
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
					return
				} else {
					return nil, errors.Wrap(err, "Failed to carry HadamardProd()")
				}
			} else {
				return nil, errors.Wrap(err, "Failed to carry Neg()")
			}

		} else {
			return nil, errors.Wrap(err, "Failed to carry HadamardProd()")
		}
	} else {
		return nil, errors.Wrap(err, "Failed to carry HadamardProd()")
	}
}

func hadamardDivDiff(x, y, z *Node) (err error) {
	xdv := x.boundTo.(*dualValue)
	ydv := y.boundTo.(*dualValue)
	zdv := z.boundTo.(*dualValue)

	div := newEBOByType(divOpType, TypeOf(zdv.d), TypeOf(ydv.Value))

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
		return errors.Wrapf(err, doFail, div)
	}

	neg := newElemUnaryOp(negOpType, y)
	if d, err = neg.Do(d); err != nil {
		return errors.Wrapf(err, doFail, neg)
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
	var one *Node
	var dt Dtype

	if dt, err = dtypeOf(y.t); err != nil {
		return nil, errors.Wrapf(err, dtypeExtractionFail, y.t)
	}

	switch dt {
	case Float32:
		one = onef32
	case Float64:
		one = onef64
	default:
		err = errors.Errorf(nyiTypeFail, "Hadamard Power Diff", y.t)
	}

	var ym1, pow *Node
	if ym1, err = Sub(y, one); err != nil {
		return
	}

	if pow, err = Pow(x, ym1); err != nil {
		return
	}

	var dzdx *Node
	if dzdx, err = HadamardProd(grad, y); err != nil {
		return
	}
	if dzdx, err = HadamardProd(dzdx, pow); err != nil {
		return
	}

	var logx *Node
	if logx, err = Log(x); err != nil {
		return
	}

	var dzdy *Node
	if dzdy, err = HadamardProd(grad, z); err != nil {
		return
	}
	if dzdy, err = HadamardProd(dzdy, logx); err != nil {
		return
	}

	retVal = Nodes{dzdx, dzdy}
	return
	// return nil, errors.New("hadamardPowDiffExpr not yet implemented")
}

func hadamardPowDiff(x, y, z *Node) (err error) {
	xdv := x.boundTo.(*dualValue)
	ydv := y.boundTo.(*dualValue)
	zdv := z.boundTo.(*dualValue)

	var ym1 Value
	switch ydvt := ydv.Value.(type) {
	case F64:
		ym1 = ydvt - F64(1)
	case F32:
		ym1 = ydvt - F32(1)
	case *tensor.Dense:
		var one interface{}
		switch x.t {
		case tensor.Float64:
			one = float64(1)
		case tensor.Float32:
			one = float32(1)

		}
		if ym1, err = tensor.Sub(ydvt, one); err != nil {
			return
		}
	default:
		err = errors.Errorf(nyiTypeFail, "hadamardPowDiff", ydv.Value)
		return
	}

	// dzdx
	var pow Value
	powOp := newEBOByType(powOpType, TypeOf(xdv.Value), TypeOf(ym1))
	if pow, err = powOp.Do(xdv.Value, ym1); err != nil {
		return
	}

	mul := newEBOByType(mulOpType, TypeOf(ydv.Value), TypeOf(xdv.Value))
	if pow, err = mul.UnsafeDo(pow, ydv.Value); err != nil {
		return
	}

	if err = mul.IncrDo(xdv.d, pow, zdv.d); err != nil {
		var ver Valuer
		var ok bool
		if ver, ok = err.(Valuer); !ok {
			return
		}

		xdv.SetDeriv(ver.Value())
	}

	// dzdy
	var logx Value
	logOp := newElemUnaryOp(lnOpType, x)
	if logx, err = logOp.Do(xdv.Value); err != nil {
		return
	}
	if logx, err = mul.Do(zdv.Value, logx); err != nil {
		return
	}
	if err = mul.IncrDo(ydv.d, logx, zdv.d); err != nil {
		var ver Valuer
		var ok bool
		if ver, ok = err.(Valuer); !ok {
			return
		}

		ydv.SetDeriv(ver.Value())
	}
	return nil
}

func nondiffBinOpExpr(x, y, z, grad *Node) (retVal Nodes, err error) {
	return nil, errors.New("Nondifferentiable")
}

func nondiffBinOp(x, y, z *Node) (err error) {
	return AutoDiffError{}
}
