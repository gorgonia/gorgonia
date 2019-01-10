package engine

import (
	"math"

	"github.com/chewxy/math32"
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia/internal/execution"
	"gorgonia.org/gorgonia/internal/value"
	"gorgonia.org/gorgonia/ops"
	"gorgonia.org/tensor"
)

type incrDoerBinOp interface {
	IncrDo(v value.Value, retSame bool, inputs ...value.Value) error
}
type usePreallocDoerBinOp interface {
	UsePreallocDo(v value.Value, retSame bool, inputs ...value.Value) (retVal value.Value, err error)
}
type unsafeDoerBinOp interface {
	UnsafeDo(retSame bool, inputs ...value.Value) (value.Value, error)
}

/* BINARY OPERATOR */

type ʘBinaryOperator interface {
	isArith() bool
	binOpType() ʘBinaryOperatorType
	Do(bool, ...value.Value) (value.Value, error)
	String() string
}

type scalarBinOp struct {
	ʘBinaryOperatorType
	t tensor.Dtype
}

func (o scalarBinOp) Arity() int                     { return 2 }
func (o scalarBinOp) binOpType() ʘBinaryOperatorType { return o.ʘBinaryOperatorType }
func (o scalarBinOp) isArith() bool                  { return o.ʘBinaryOperatorType.isArith() }
func (o scalarBinOp) String() string                 { return o.ʘBinaryOperatorType.String() }

func (o scalarBinOp) Do(same bool, vals ...value.Value) (retVal value.Value, err error) {
	if err = ops.CheckArity(o, len(vals)); err != nil {
		return
	}

	at := value.TypeOf(vals[0])
	bt := value.TypeOf(vals[1])
	if !at.Eq(bt) {
		err = errors.Errorf("Type Mismatch: %v != %v", at, bt)
		return
	}

	var r interface{} // float or bool only plz
	switch a := vals[0].(type) {
	case *value.F64:
		b := vals[1].(*value.F64)
		switch o.ʘBinaryOperatorType {
		case addOpType:
			r = value.NewF64(a.Any() + b.Any())
		case subOpType:
			r = value.NewF64(a.Any() - b.Any())
		case mulOpType:
			r = value.NewF64(a.Any() * b.Any())
		case divOpType:
			r = value.NewF64(a.Any() / b.Any())
		case powOpType:
			r = value.NewF64(math.Pow(a.Any(), b.Any()))
		case ltOpType:
			r = value.NewB(a.Any() < b.Any())
		case gtOpType:
			r = value.NewB(a.Any() > b.Any())
		case lteOpType:
			r = value.NewB(a.Any() <= b.Any())
		case gteOpType:
			r = value.NewB(a.Any() >= b.Any())
		case eqOpType:
			r = value.NewB(a.Any() == b.Any())
		case neOpType:
			r = value.NewB(a.Any() != b.Any())
		default:
			err = errors.Errorf(nyiFail, "scalarBinOp.Do() - Float64", o.ʘBinaryOperatorType)
		}

		if same && !o.isArith() {
			if *(r.(*value.B)) {
				r = value.NewF64(1.0)
			} else {
				r = value.NewF64(0.0)
			}
		}

	case *value.F32:
		b := vals[1].(*value.F32)
		switch o.ʘBinaryOperatorType {
		case addOpType:
			r = value.NewF32(a.Any() + b.Any())
		case subOpType:
			r = value.NewF32(a.Any() - b.Any())
		case mulOpType:
			r = value.NewF32(a.Any() * b.Any())
		case divOpType:
			r = value.NewF32(a.Any() / b.Any())
		case powOpType:
			r = value.NewF32(math32.Pow(float32(a.Any()), float32(b.Any())))
		case ltOpType:
			r = value.NewB(a.Any() < b.Any())
		case gtOpType:
			r = value.NewB(a.Any() > b.Any())
		case lteOpType:
			r = value.NewB(a.Any() <= b.Any())
		case gteOpType:
			r = value.NewB(a.Any() >= b.Any())
		case eqOpType:
			r = value.NewB(a.Any() == b.Any())
		case neOpType:
			r = value.NewB(a.Any() != b.Any())
		default:
			err = errors.Errorf(nyiFail, "scalarBinOp.Do() - Float32", o.ʘBinaryOperatorType)
		}

		if same && !o.isArith() {
			if *(r.(*value.B)) {
				r = value.F32(1)
			} else {
				r = value.F32(0)
			}
		}

	case *value.I:
		b := vals[1].(*value.I)
		switch o.ʘBinaryOperatorType {
		case addOpType:
			r = value.NewI(a.Any() + b.Any())
		case subOpType:
			r = value.NewI(a.Any() - b.Any())
		case mulOpType:
			r = value.NewI(a.Any() * b.Any())
		case divOpType:
			r = value.NewI(a.Any() / b.Any())
		// case powOpType:
		// 	r = math.Pow(a, b)
		case ltOpType:
			r = value.NewB(a.Any() < b.Any())
		case gtOpType:
			r = value.NewB(a.Any() > b.Any())
		case lteOpType:
			r = value.NewB(a.Any() <= b.Any())
		case gteOpType:
			r = value.NewB(a.Any() >= b.Any())
		case eqOpType:
			r = value.NewB(a.Any() == b.Any())
		case neOpType:
			r = value.NewB(a.Any() != b.Any())
		default:
			err = errors.Errorf(nyiFail, "scalarBinOp.Do() - Int", o.ʘBinaryOperatorType)
		}

		if same && !o.isArith() {
			if *(r.(*value.B)) {
				r = value.I(1)
			} else {
				r = value.I(0)
			}
		}
	case *value.I32:
		b := vals[1].(*value.I32)
		switch o.ʘBinaryOperatorType {
		case addOpType:
			r = value.NewI32(a.Any() + b.Any())
		case subOpType:
			r = value.NewI32(a.Any() - b.Any())
		case mulOpType:
			r = value.NewI32(a.Any() * b.Any())
		case divOpType:
			r = value.NewI32(a.Any() / b.Any())
		// case powOpType:
		// 	r = math.Pow(a, b)
		case ltOpType:
			r = value.NewB(a.Any() < b.Any())
		case gtOpType:
			r = value.NewB(a.Any() > b.Any())
		case lteOpType:
			r = value.NewB(a.Any() <= b.Any())
		case gteOpType:
			r = value.NewB(a.Any() >= b.Any())
		case eqOpType:
			r = value.NewB(a.Any() == b.Any())
		case neOpType:
			r = value.NewB(a.Any() != b.Any())
		default:
			err = errors.Errorf(nyiFail, "scalarBinOp.Do() - Int32", o.ʘBinaryOperatorType)
		}

		if same && !o.isArith() {
			if *(r.(*value.B)) {
				r = value.I32(1)
			} else {
				r = value.I32(0)
			}
		}
	case *value.I64:
		b := vals[1].(*value.I64)
		switch o.ʘBinaryOperatorType {
		case addOpType:
			r = value.NewI64(a.Any() + b.Any())
		case subOpType:
			r = value.NewI64(a.Any() - b.Any())
		case mulOpType:
			r = value.NewI64(a.Any() * b.Any())
		case divOpType:
			r = value.NewI64(a.Any() / b.Any())
		// case powOpType:
		// 	r = math.Pow(a, b)
		case ltOpType:
			r = value.NewB(a.Any() < b.Any())
		case gtOpType:
			r = value.NewB(a.Any() > b.Any())
		case lteOpType:
			r = value.NewB(a.Any() <= b.Any())
		case gteOpType:
			r = value.NewB(a.Any() >= b.Any())
		case eqOpType:
			r = value.NewB(a.Any() == b.Any())
		case neOpType:
			r = value.NewB(a.Any() != b.Any())
		default:
			err = errors.Errorf(nyiFail, "scalarBinOp.Do() - Int64", o.ʘBinaryOperatorType)
		}

		if same && !o.isArith() {
			if *(r.(*value.B)) {
				r = value.I64(1)
			} else {
				r = value.I64(0)
			}
		}
	case *value.U8:
		b := vals[1].(*value.U8)
		switch o.ʘBinaryOperatorType {
		case addOpType:
			r = value.NewU8(a.Any() + b.Any())
		case subOpType:
			r = value.NewU8(a.Any() - b.Any())
		case mulOpType:
			r = value.NewU8(a.Any() * b.Any())
		case divOpType:
			r = value.NewU8(a.Any() / b.Any())
		// case powOpType:
		// 	r = math.Pow(a, b)
		case ltOpType:
			r = value.NewB(a.Any() < b.Any())
		case gtOpType:
			r = value.NewB(a.Any() > b.Any())
		case lteOpType:
			r = value.NewB(a.Any() <= b.Any())
		case gteOpType:
			r = value.NewB(a.Any() >= b.Any())
		case eqOpType:
			r = value.NewB(a.Any() == b.Any())
		case neOpType:
			r = value.NewB(a.Any() != b.Any())
		default:
			err = errors.Errorf(nyiFail, "scalarBinOp.Do() - Byte", o.ʘBinaryOperatorType)
		}

		if same && !o.isArith() {
			if *(r.(*value.B)) {
				r = value.U8(1)
			} else {
				r = value.U8(0)
			}
		}
	case *value.B:
		b := vals[1].(*value.B)
		switch o.ʘBinaryOperatorType {
		case eqOpType:
			r = value.NewB(a.Any() == b.Any())
		case neOpType:
			r = value.NewB(a.Any() != b.Any())
		default:
			err = errors.Errorf(nyiFail, "scalarBinOp.Do() - Bool", o.ʘBinaryOperatorType)
		}

	default:
		err = errors.Errorf(nyiFail, "scalarBinOp.Do() - Unhandled value.Scalar Type", o.t)
	}

	if err != nil {
		return
	}

	retVal, _ = value.AnyToScalar(r)
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

func (o tBinOp) Do(same bool, inputs ...value.Value) (value.Value, error) {
	if same {
		return o.do(inputs, tensor.AsSameType())
	}
	return o.do(inputs)
}

func (o tBinOp) UnsafeDo(retSame bool, inputs ...value.Value) (value.Value, error) {
	if retSame {
		return o.do(inputs, tensor.AsSameType(), tensor.UseUnsafe())
	}
	return o.do(inputs, tensor.UseUnsafe())
}
func (o tBinOp) UsePreallocDo(v value.Value, retSame bool, inputs ...value.Value) (retVal value.Value, err error) {
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

func (o tBinOp) IncrDo(incr value.Value, retSame bool, inputs ...value.Value) (err error) {
	reuse, ok := incr.(tensor.Tensor)
	if ok {
		_, err = o.do(inputs, tensor.WithIncr(reuse))
		return
	}

	var retVal value.Value
	if retSame {
		if retVal, err = o.do(inputs, tensor.AsSameType()); err != nil {
			return errors.Wrapf(err, doFail, o)
		}
	} else {
		if retVal, err = o.do(inputs); err != nil {
			return errors.Wrapf(err, doFail, o)
		}

	}

	add := newEBOByType(addOpType, value.TypeOf(incr), value.TypeOf(retVal))
	if retVal, err = add.UnsafeDo(incr, retVal); err != nil {
		return errors.Wrapf(err, unsafeDoFail, add)
	}

	err = noIncrErr{retVal}
	return
}

func (o tBinOp) do(vals []value.Value, opts ...tensor.FuncOpt) (retVal value.Value, err error) {
	if err = ops.CheckArity(o, len(vals)); err != nil {
		return
	}

	// typecheck the operands
	d0 := vals[0].Dtype()
	d1 := vals[1].Dtype()

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
		a = tensor.Materialize(t)
		// a = t

		switch other := vals[1].(type) {
		case *value.F64:
			b = other.Any()
		case *value.F32:
			b = other.Any()
		case tensor.Tensor:
			b = tensor.Materialize(other)
		default:
			return nil, errors.Errorf(nyiFail, "tBinOp.do()", vals[1])
		}
	} else {
		t, ok := vals[1].(tensor.Tensor)
		if !ok {
			return nil, errors.Errorf("Expected right value to be Tensor. Got %v of %T instead", vals[1], vals[1])
		}
		b = tensor.Materialize(t)

		switch other := vals[0].(type) {
		case *value.F64:
			a = other.Any()
		case *value.F32:
			a = other.Any()
		case tensor.Tensor:
			a = tensor.Materialize(other)
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

func addDiff(ctx execution.Context, x, y, z *Node) (err error) {
	xdv, ydv := getDV(x, y)

	// set up the op to be executed
	op := NewAddOp(x, z, ctx)
	op.Device = x.Device()
	op.UseUnsafe = true

	// we'll use the same device as the device the data from the node resides in
	dev := op.Device

	var d, xd, yd, zd value.Value
	var extra bool

	// allocate if necessary
	if xd, extra, err = x.GradOnDevice(dev, ctx.External); err != nil {
		return errors.Wrapf(err, gradOnDeviceFail, x, dev)
	}
	if extra {
		defer ctx.PutValue(dev, xd)
	}

	if zd, extra, err = z.GradOnDevice(dev, ctx.External); err != nil {
		return errors.Wrapf(err, gradOnDeviceFail, z, dev)
	}
	if extra {
		defer ctx.PutValue(dev, xd)
	}

	// if x is scalar, an additional vector needs to be acquired
	if x.IsScalar() && dev != execution.CPU {
		var mem tensor.Memory
		var xd2 value.Value
		memsize := calcMemSize(zd.Dtype(), zd.Shape())
		if mem, err = ctx.Get(dev, memsize); err != nil {
			return
		}

		if xd2, err = makeValueFromMem(z.t, zd.Shape(), mem); err != nil {
			return
		}

		op.Prealloc = xd2
		defer ctx.Signal()
	}

	// xd += zd
	if d, err = op.Do(xd, zd); err != nil {
		return errors.Wrapf(err, doFail, op)
	}
	xdv.SetDeriv(d)

	// set up the op to be executed for y
	op = NewAddOp(y, z, ctx)
	op.Device = y.Device()
	op.UseUnsafe = true

	dev = op.Device

	if yd, extra, err = y.GradOnDevice(dev, ctx.External); err != nil {
		return errors.Wrapf(err, gradOnDeviceFail, y, dev)
	}
	if extra {
		defer ctx.PutValue(dev, yd)
	}

	if zd, extra, err = z.GradOnDevice(dev, ctx.External); err != nil {
		return errors.Wrapf(err, gradOnDeviceFail, z, dev)
	}
	if extra {
		defer ctx.PutValue(dev, zd)
	}

	// if y is scalar, an additional vector needs to be acquired
	if y.IsScalar() && dev != execution.CPU {
		var mem tensor.Memory
		var yd2 value.Value
		memsize := calcMemSize(zd.Dtype(), zd.Shape())
		if mem, err = ctx.Get(dev, memsize); err != nil {
			return
		}
		if yd2, err = makeValueFromMem(z.t, zd.Shape(), mem); err != nil {
			return
		}

		op.Prealloc = yd2
		defer ctx.Signal()
	}

	// yd += zd
	if d, err = op.Do(yd, zd); err != nil {
		return errors.Wrapf(err, doFail, op)
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

func subDiff(ctx execution.Context, x, y, z *Node) (err error) {
	xdv, ydv := getDV(x, y)

	add := NewAddOp(x, z, ctx)
	sub := NewSubOp(y, z, ctx)
	add.Device = x.Device()
	sub.Device = y.Device()
	sub.UseUnsafe = true
	add.UseUnsafe = true
	// sub := newEBOByType(subOpType, y.t, z.t)
	// add := newEBOByType(addOpType, x.t, z.t)

	dev := sub.Device
	var xd, yd, zd, d value.Value
	var extra bool

	if zd, extra, err = z.GradOnDevice(dev, ctx.External); err != nil {
		return errors.Wrapf(err, gradOnDeviceFail, z, dev)
	}
	if extra {
		defer ctx.PutValue(dev, zd)
	}

	if yd, extra, err = y.GradOnDevice(dev, ctx.External); err != nil {
		return errors.Wrapf(err, gradOnDeviceFail, y, dev)
	}
	if extra {
		defer ctx.PutValue(dev, yd)
	}

	// if y is scalar an additional vector needs to be allocated for the prelloc
	switch {
	case y.IsScalar() && dev != execution.CPU:
		var mem tensor.Memory
		var yd2 value.Value
		memsize := calcMemSize(zd.Dtype(), zd.Shape())
		if mem, err = ctx.Get(dev, memsize); err != nil {
			return errors.Wrapf(err, allocFail, memsize, dev)
		}
		if yd2, err = makeValueFromMem(z.t, zd.Shape(), mem); err != nil {
			return errors.Wrapf(err, makeValueFail, z.t, zd.Shape())
		}

		sub.Prealloc = yd2
		defer ctx.Signal()
	case y.IsScalar() && dev == execution.CPU:
		if sub.Prealloc, err = makeValue(z.t, zd.Shape()); err != nil {
			return
		}
	}

	// dz/dy
	if d, err = sub.Do(yd, zd); err != nil {
		return errors.Wrapf(err, doFail, sub)
	}
	ydv.SetDeriv(d) // errors are ignored on purpose

	//	handle x

	dev = add.Device
	if zd, extra, err = z.GradOnDevice(dev, ctx.External); err != nil {
		return errors.Wrapf(err, gradOnDeviceFail, z, dev)
	}
	if extra {
		defer ctx.PutValue(dev, zd)
	}

	if xd, extra, err = x.GradOnDevice(dev, ctx.External); err != nil {
		return errors.Wrapf(err, gradOnDeviceFail, x, dev)
	}
	if extra {
		defer ctx.PutValue(dev, xd)
	}

	switch {
	case x.IsScalar() && dev != execution.CPU:
		var mem tensor.Memory
		var xd2 value.Value
		memsize := calcMemSize(zd.Dtype(), zd.Shape())
		if mem, err = ctx.Get(dev, memsize); err != nil {
			return
		}

		if xd2, err = makeValueFromMem(z.t, zd.Shape(), mem); err != nil {
			return
		}
		add.Prealloc = xd2
		defer ctx.Signal()
	case x.IsScalar() && dev == execution.CPU:
		if sub.Prealloc, err = makeValue(z.t, zd.Shape()); err != nil {
			return
		}
	}

	// dz/dx
	if d, err = add.Do(xd, zd); err != nil {
		return errors.Wrapf(err, doFail, add)
	}
	xdv.SetDeriv(d) // ignore errors on purpose

	return nil
}

func hadamardProdDiffExpr(x, y, z, gradZ *Node) (retVal Nodes, err error) {
	var dzdx, dzdy *Node
	if dzdx, err = HadamardProd(y, gradZ); err == nil {
		dzdy, err = HadamardProd(x, gradZ)
		if err != nil {
			return nil, errors.Wrap(err, "Failed to carry HadamardProd()")
		}
		WithGroupName(gradClust)(dzdx)
		WithGroupName(gradClust)(dzdy)
		retVal = Nodes{dzdx, dzdy}
		return
	}
	return nil, errors.Wrap(err, "Failed to carry HadamardProd()")
}

func hadamardProdDiff(ctx execution.Context, x, y, z *Node) (err error) {
	xdv, ydv := getDV(x, y)

	var mul *ExternalOp
	var dev execution.Device
	var xd, yd, zd, d value.Value
	var extra bool

	if x.isConstant() {
		goto dzdy
	}

	//dzdx
	mul = NewHadamardProdOp(y, z, ctx)
	mul.Device = x.Device()
	dev = mul.Device

	if xd, extra, err = x.GradOnDevice(dev, ctx.External); err != nil {
		return errors.Wrapf(err, gradOnDeviceFail, x, dev)
	}
	if extra {
		defer ctx.PutValue(dev, xd)
	}

	if yd, extra, err = y.ValueOnDevice(dev, ctx.External); err != nil {
		return errors.Wrapf(err, gradOnDeviceFail, y, dev)
	}
	if extra {
		defer ctx.PutValue(dev, yd)
	}

	if zd, extra, err = z.GradOnDevice(dev, ctx.External); err != nil {
		return errors.Wrapf(err, gradOnDeviceFail, z, dev)
	}
	if extra {
		defer ctx.PutValue(dev, zd)
	}

	mul.Incr = xd

	// if y is value.Scalar, then it needs to be broadcasted across to the
	if x.IsScalar() && dev != execution.CPU && !zd.Shape().IsScalar() {
		var memIncr, mem2 tensor.Memory
		var xdIncr, xd2 value.Value
		memsize := calcMemSize(zd.Dtype(), zd.Shape())
		if mem2, err = ctx.Get(dev, memsize); err != nil {
			return errors.Wrapf(err, allocFail, memsize, dev)
		}

		if xd2, err = makeValueFromMem(z.t, zd.Shape(), mem2); err != nil {
			return errors.Wrapf(err, makeValueFail, z.t, zd.Shape())
		}

		// "broadcast" x (in a very sloppy way)
		if memIncr, err = ctx.Get(dev, memsize); err != nil {
			return errors.Wrapf(err, allocFail, memsize, dev)
		}

		if xdIncr, err = makeValueFromMem(z.t, zd.Shape(), memIncr); err != nil {
			return errors.Wrapf(err, makeValueFail, z.t, zd.Shape())
		}
		xdIncr.(tensor.Tensor).Memset(xdv.D.Data())

		mul.Prealloc = xd2
		mul.Incr = xdIncr

		defer ctx.PutValue(dev, xd2) // xd2 is temporary, we need to dealloc it
		defer ctx.Signal()           // work needs to be done
	}

	if d, err = mul.Do(yd, zd); err != nil {
		return errors.Wrapf(err, "IncrDo xd faile")
	}

	xdv.SetDeriv(d)

dzdy:
	if y.isConstant() {
		goto end
	}

	mul = NewHadamardProdOp(x, z, ctx)
	mul.Device = y.Device()
	dev = mul.Device

	if xd, extra, err = x.ValueOnDevice(dev, ctx.External); err != nil {
		return errors.Wrapf(err, gradOnDeviceFail, x, dev)
	}
	if extra {
		defer ctx.PutValue(dev, xd)
	}

	if yd, extra, err = y.GradOnDevice(dev, ctx.External); err != nil {
		return errors.Wrapf(err, gradOnDeviceFail, y, dev)
	}
	if extra {
		defer ctx.PutValue(dev, yd)
	}

	if zd, extra, err = z.GradOnDevice(dev, ctx.External); err != nil {
		return errors.Wrapf(err, gradOnDeviceFail, z, dev)
	}
	if extra {
		defer ctx.PutValue(dev, zd)
	}

	mul.Incr = yd

	// if y is value.Scalar, then it needs to be broadcasted across to the
	if y.IsScalar() && dev != execution.CPU && !zd.Shape().IsScalar() {
		var memIncr, mem2 tensor.Memory
		var ydIncr, yd2 value.Value
		memsize := calcMemSize(zd.Dtype(), zd.Shape())
		if mem2, err = ctx.Get(dev, memsize); err != nil {
			return errors.Wrapf(err, allocFail, memsize, dev)
		}

		if yd2, err = makeValueFromMem(z.t, zd.Shape(), mem2); err != nil {
			return errors.Wrapf(err, makeValueFail, z.t, zd.Shape())
		}

		// "broadcast" y (in a very sloppy way)
		if memIncr, err = ctx.Get(dev, memsize); err != nil {
			return errors.Wrapf(err, allocFail, memsize, dev)
		}

		if ydIncr, err = makeValueFromMem(z.t, zd.Shape(), memIncr); err != nil {
			return errors.Wrapf(err, makeValueFail, z.t, zd.Shape())
		}
		ydIncr.(tensor.Tensor).Memset(ydv.D.Data())

		mul.Prealloc = yd2
		mul.Incr = ydIncr

		defer ctx.PutValue(dev, yd2) // yd2 is temporary, we need to dealloc it
		defer ctx.Signal()           // work needs to be done
	}

	if d, err = mul.Do(xd, zd); err != nil {
		return errors.Wrapf(err, "IncrDo yd failed")
	}
	ydv.SetDeriv(d)

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
				}
				return nil, errors.Wrap(err, "Failed to carry HadamardProd()")
			}
			return nil, errors.Wrap(err, "Failed to carry Neg()")
		}
		return nil, errors.Wrap(err, "Failed to carry HadamardProd()")
	}
	return nil, errors.Wrap(err, "Failed to carry HadamardProd()")
}

func hadamardDivDiff(ctx execution.Context, x, y, z *Node) (err error) {
	xdv, ydv, zdv := getDV3(x, y, z)

	// dzdx = 1/y * dz
	div := newEBOByType(divOpType, value.TypeOf(zdv.D), value.TypeOf(ydv.Value))
	err = div.IncrDo(xdv.D, zdv.D, ydv.Value)
	if err != nil {
		var ver value.Valuer
		var ok bool
		if ver, ok = err.(value.Valuer); !ok {
			return
		}

		xdv.SetDeriv(ver.Value()) // ignore errors on purpose
	}

	//dzdy = -x/y^2
	// TODO: investigate if this can be done (if no other node uses z):
	//		unsafe do : neg zdv.D
	// 		unsafe do : mul zdv.D, zdv.Value
	//		incr do   : <incr: ydv.D> div zdv.D, ydv.Value
	var d value.Value
	if d, err = div.Do(zdv.Value, ydv.Value); err != nil {
		return errors.Wrapf(err, doFail, div)
	}

	neg := newElemUnaryOp(negOpType, y)
	if d, err = neg.Do(d); err != nil {
		return errors.Wrapf(err, doFail, neg)
	}

	mul := newElemBinOp(mulOpType, z, y)
	err = mul.IncrDo(ydv.D, zdv.D, d)
	if err != nil {
		var ver value.Valuer
		var ok bool
		if ver, ok = err.(value.Valuer); !ok {
			return
		}

		ydv.SetDeriv(ver.Value()) // ignore errors on purpose
	}

	return nil
}

// TODO: go back in time, pay more attention to calculus class in high school and learn how to differentiate x^y
func hadamardPowDiffExpr(x, y, z, grad *Node) (retVal Nodes, err error) {
	var one *Node
	var dt tensor.Dtype

	if dt, err = dtypeOf(y.t); err != nil {
		return nil, errors.Wrapf(err, dtypeExtractionFail, y.t)
	}

	switch dt {
	case Float32:
		one = onef32(x.g)
	case Float64:
		one = onef64(x.g)
	default:
		err = errors.Errorf(nyiTypeFail, "Hadamard Power Diff", y.t)
		return
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

func hadamardPowDiff(ctx execution.Context, x, y, z *Node) (err error) {
	xdv, ydv, zdv := getDV3(x, y, z)

	var ym1 value.Value
	switch ydvt := ydv.Value.(type) {
	case *value.F64:
		ym1 = value.NewF64(ydvt.Any() - float64(1))
	case *value.F32:
		ym1 = value.NewF32(ydvt.Any() - float32(1))
	case *tensor.Dense:
		var one interface{}
		switch ydvt.Dtype() {
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
	var pow value.Value
	powOp := newEBOByType(powOpType, value.TypeOf(xdv.Value), value.TypeOf(ym1))
	if pow, err = powOp.Do(xdv.Value, ym1); err != nil {
		return
	}

	mul := newEBOByType(mulOpType, value.TypeOf(ydv.Value), value.TypeOf(xdv.Value))
	if pow, err = mul.UnsafeDo(pow, ydv.Value); err != nil {
		return
	}

	if err = mul.IncrDo(xdv.D, pow, zdv.D); err != nil {
		var ver value.Valuer
		var ok bool
		if ver, ok = err.(value.Valuer); !ok {
			return
		}

		xdv.SetDeriv(ver.Value())
	}

	// dzdy
	var logx value.Value
	logOp := newElemUnaryOp(lnOpType, x)
	if logx, err = logOp.Do(xdv.Value); err != nil {
		return
	}
	if logx, err = mul.Do(zdv.Value, logx); err != nil {
		return
	}
	if err = mul.IncrDo(ydv.D, logx, zdv.D); err != nil {
		var ver value.Valuer
		var ok bool
		if ver, ok = err.(value.Valuer); !ok {
			return
		}

		ydv.SetDeriv(ver.Value())
	}
	return nil
}

func nondiffBinOpExpr(x, y, z, grad *Node) (retVal Nodes, err error) {
	return nil, errors.New("Nondifferentiable")
}

func nondiffBinOp(ctx execution.Context, x, y, z *Node) (err error) {
	return AutoDiffError{}
}
