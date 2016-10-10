package tensorf64

import "github.com/chewxy/gorgonia/tensor/types"

// public API for arithmetics and the stupidly crazy amount of overloaded semantics

// Add performs a pointwise a+b. a and b can either be float64 or *Tensor
//
// If both operands are *Tensor, shape is checked first.
// Even though the underlying data may have the same size (say (2,2) vs (4,1)), if they have different shapes, it will error out.
//
// If the Unsafe flag is passed in, the data of the first tensor will be overwritten
func Add(a, b interface{}, opts ...types.FuncOpt) (retVal *Tensor, err error) {
	safe, incr, reuse := parseSafeReuse(opts...)

	at, atok := a.(*Tensor)
	bt, btok := b.(*Tensor)
	af, afok := a.(float64)
	bf, bfok := b.(float64)

	toReuse := reuse != nil

	if atok && btok {
		// assert that they have the same shape'
		if !at.Shape().Eq(bt.Shape()) {
			err = types.NewError(types.ShapeMismatch, "Cannot add tensors with shapes %v and %v", at.Shape(), bt.Shape())
			return
		}
	}

	switch {
	case toReuse && atok:
		if err = reuseCheck(reuse, at); err != nil {
			return
		}
	case toReuse && btok:
		if err = reuseCheck(reuse, bt); err != nil {
			return
		}
	}

	switch {
	// incr
	case incr && atok && btok:
		if reuse == bt {
			vecAdd(reuse.data, bt.data)
			vecAdd(reuse.data, at.data)
		} else {
			vecAdd(reuse.data, at.data)
			vecAdd(reuse.data, bt.data)
		}
		retVal = reuse
	case incr && atok && bfok:
		vecAdd(reuse.data, at.data)
		vecTrans(bf, reuse.data)
		retVal = reuse
	case incr && afok && btok:
		vecAdd(reuse.data, bt.data)
		vecTrans(af, reuse.data)
		retVal = reuse

	//reuse
	case toReuse && atok && btok:
		safeVecAdd(at.data, bt.data, reuse.data)
		retVal = reuse
	case toReuse && atok && bfok:
		safeVecTrans(bf, at.data, reuse.data)
		retVal = reuse
	case toReuse && afok && btok:
		safeVecTrans(af, bt.data, reuse.data)
		retVal = reuse

	// safe
	case safe && atok && btok:
		retVal = newBorrowedTensor(len(at.data))
		if err = retVal.Reshape(at.Shape()...); err != nil {
			return
		}
		safeVecAdd(at.data, bt.data, retVal.data)
	case safe && atok && bfok:
		retVal = newBorrowedTensor(len(at.data))
		if err = retVal.Reshape(at.Shape()...); err != nil {
			return
		}
		safeVecTrans(bf, at.data, retVal.data)
	case safe && afok && btok:
		retVal = newBorrowedTensor(len(bt.data))
		if err = retVal.Reshape(bt.Shape()...); err != nil {
			return
		}
		safeVecTrans(af, bt.data, retVal.data)

	// unsafe
	case !safe && atok && btok:
		vecAdd(at.data, bt.data)
		retVal = at
	case !safe && atok && bfok:
		vecTrans(bf, at.data)
		retVal = at
	case !safe && afok && btok:
		vecTrans(af, bt.data)
		retVal = bt
	default:
		err = types.NewError(types.DtypeMismatch, "Addition cannot be done on %T and %T", a, b)
		return
	}
	return
}

// Sub performs a pointwise a-b . a and b can either be float64 or *Tensor
//
// If both operands are *Tensor, shape is checked first.
// Even though the underlying data may have the same size (say (2,2) vs (4,1)), if they have different shapes, it will error out.
//
// If the Unsafe flag is passed in, the data of the first tensor will be overwritten
func Sub(a, b interface{}, opts ...types.FuncOpt) (retVal *Tensor, err error) {
	safe, incr, reuse := parseSafeReuse(opts...)

	at, atok := a.(*Tensor)
	bt, btok := b.(*Tensor)
	af, afok := a.(float64)
	bf, bfok := b.(float64)

	toReuse := reuse != nil

	if atok && btok {
		// assert that they have the same shape
		if !at.Shape().Eq(bt.Shape()) {
			err = types.NewError(types.ShapeMismatch, "Cannot add tensors with shapes %v and %v", at.Shape(), bt.Shape())
			return
		}
	}

	switch {
	case toReuse && atok:
		if err = reuseCheck(reuse, at); err != nil {
			return
		}
	case toReuse && btok:
		if err = reuseCheck(reuse, bt); err != nil {
			return
		}
	}

	switch {
	// incr
	case incr && atok && btok:
		if reuse == bt {
			copy(reuse.data, at.data)
		} else {
			vecAdd(reuse.data, at.data)
			vecSub(reuse.data, bt.data)
		}
		retVal = reuse
	case incr && atok && bfok:
		vecAdd(reuse.data, at.data)
		vecTrans(-bf, reuse.data)
		retVal = reuse
	case incr && afok && btok:
		// fmt.Println("OK")
		vecTrans(af, reuse.data)
		vecSub(reuse.data, bt.data)
		retVal = reuse

	//reuse
	case toReuse && atok && btok:
		safeVecSub(at.data, bt.data, reuse.data)
		retVal = reuse
	case toReuse && atok && bfok:
		safeVecTrans(-bf, at.data, reuse.data)
		retVal = reuse
	case toReuse && afok && btok:
		safeVecTransFrom(af, bt.data, reuse.data)
		retVal = reuse

	// safe
	case safe && atok && btok:
		retVal = newBorrowedTensor(len(at.data))
		if err = retVal.Reshape(at.Shape()...); err != nil {
			return
		}
		safeVecSub(at.data, bt.data, retVal.data)
	case safe && atok && bfok:
		retVal = newBorrowedTensor(len(at.data))
		if err = retVal.Reshape(at.Shape()...); err != nil {
			return
		}
		safeVecTrans(-bf, at.data, retVal.data)
	case safe && afok && btok:
		retVal = newBorrowedTensor(len(bt.data))
		if err = retVal.Reshape(bt.Shape()...); err != nil {
			return
		}
		safeVecTransFrom(af, bt.data, retVal.data)

	// unsafe
	case !safe && atok && btok:
		vecSub(at.data, bt.data)
		retVal = at
	case !safe && atok && bfok:
		vecTrans(-bf, at.data)
		retVal = at
	case !safe && afok && btok:
		vecTransFrom(af, bt.data)
		retVal = bt
	default:
		err = types.NewError(types.DtypeMismatch, "Subtraction cannot be done on %T and %T", a, b)
		return
	}
	return
}

// PointwiseMul performs a pointwise a * b. a and b can either be float64 or *Tensor
//
// If both operands are *Tensor, shape is checked first.
// Even though the underlying data may have the same size (say (2,2) vs (4,1)), if they have different shapes, it will error out.
//
// If the Unsafe flag is passed in, the data of the first tensor will be overwritten
func PointwiseMul(a, b interface{}, opts ...types.FuncOpt) (retVal *Tensor, err error) {
	safe, incr, reuse := parseSafeReuse(opts...)

	at, atok := a.(*Tensor)
	bt, btok := b.(*Tensor)
	af, afok := a.(float64)
	bf, bfok := b.(float64)

	toReuse := reuse != nil

	if atok && btok {
		// assert that they have the same shape
		if !at.Shape().Eq(bt.Shape()) {
			err = types.NewError(types.ShapeMismatch, "Cannot add tensors with shapes %v and %v", at.Shape(), bt.Shape())
			return
		}
	}

	switch {
	case toReuse && atok:
		if err = reuseCheck(reuse, at); err != nil {
			return
		}
	case toReuse && btok:
		if err = reuseCheck(reuse, bt); err != nil {
			return
		}
	}

	switch {
	// incr
	case incr && atok && btok:
		incrVecMul(reuse.data, at.data, bt.data)
		retVal = reuse
	case incr && atok && bfok:
		incrVecScale(reuse.data, at.data, bf)
		retVal = reuse
	case incr && afok && btok:
		incrVecScale(reuse.data, bt.data, af)
		retVal = reuse

	//reuse
	case toReuse && atok && btok:
		safeVecMul(at.data, bt.data, reuse.data)
		retVal = reuse
	case toReuse && atok && bfok:
		safeVecScale(bf, at.data, reuse.data)
		retVal = reuse
	case toReuse && afok && btok:
		safeVecScale(af, bt.data, reuse.data)
		retVal = reuse

	// safe
	case safe && atok && btok:
		retVal = newBorrowedTensor(len(at.data))
		if err = retVal.Reshape(at.Shape()...); err != nil {
			return
		}
		safeVecMul(at.data, bt.data, retVal.data)
	case safe && atok && bfok:
		retVal = newBorrowedTensor(len(at.data))
		if err = retVal.Reshape(at.Shape()...); err != nil {
			return
		}
		safeVecScale(bf, at.data, retVal.data)
	case safe && afok && btok:
		retVal = newBorrowedTensor(len(bt.data))
		if err = retVal.Reshape(bt.Shape()...); err != nil {
			return
		}
		safeVecScale(af, bt.data, retVal.data)

	// unsafe
	case !safe && atok && btok:
		vecMul(at.data, bt.data)
		retVal = at
	case !safe && atok && bfok:
		vecScale(bf, at.data)
		retVal = at
	case !safe && afok && btok:
		vecScale(af, bt.data)
		retVal = bt
	default:
		err = types.NewError(types.DtypeMismatch, "Multiplication cannot be done on %T and %T", a, b)
		return
	}
	return
}

// PointwiseDiv performs a pointwise a / b. Valid values are either float64 or *Tensor.
//
// If both operands are *Tensor, shape is checked first.
// Even though the underlying data may have the same size (say (2,2) vs (4,1)), if they have different shapes, it will error out.
//
// If the Unsafe flag is passed in, the data of the first tensor will be overwritten
func PointwiseDiv(a, b interface{}, opts ...types.FuncOpt) (retVal *Tensor, err error) {
	safe, incr, reuse := parseSafeReuse(opts...)

	at, atok := a.(*Tensor)
	bt, btok := b.(*Tensor)
	af, afok := a.(float64)
	bf, bfok := b.(float64)

	toReuse := reuse != nil

	if atok && btok {
		// assert that they have the same shape
		if !at.Shape().Eq(bt.Shape()) {
			err = types.NewError(types.ShapeMismatch, "Cannot add tensors with shapes %v and %v", at.Shape(), bt.Shape())
			return
		}
	}

	switch {
	case toReuse && atok:
		if err = reuseCheck(reuse, at); err != nil {
			return
		}
	case toReuse && btok:
		if err = reuseCheck(reuse, bt); err != nil {
			return
		}
	}

	switch {
	// incr
	case incr && atok && btok:
		incrVecDiv(reuse.data, at.data, bt.data)
		retVal = reuse
	case incr && atok && bfok:
		incrVecScale(reuse.data, at.data, float64(1)/bf)
		retVal = reuse
	case incr && afok && btok:
		incrVecDivBy(reuse.data, bt.data, af)
		retVal = reuse

	//reuse
	case toReuse && atok && btok:
		safeVecDiv(at.data, bt.data, reuse.data)
		retVal = reuse
	case toReuse && atok && bfok:
		safeVecScale(float64(1)/bf, at.data, reuse.data)
		retVal = reuse
	case toReuse && afok && btok:
		safeVecDivBy(af, bt.data, reuse.data)
		retVal = reuse

	// safe
	case safe && atok && btok:
		retVal = newBorrowedTensor(len(at.data))
		if err = retVal.Reshape(at.Shape()...); err != nil {
			return
		}
		safeVecDiv(at.data, bt.data, retVal.data)
	case safe && atok && bfok:
		retVal = newBorrowedTensor(len(at.data))
		if err = retVal.Reshape(at.Shape()...); err != nil {
			return
		}
		safeVecScale(float64(1)/bf, at.data, retVal.data)
	case safe && afok && btok:
		retVal = newBorrowedTensor(len(bt.data))
		if err = retVal.Reshape(bt.Shape()...); err != nil {
			return
		}
		safeVecDivBy(af, bt.data, retVal.data)

	// unsafe
	case !safe && atok && btok:
		vecDiv(at.data, bt.data)
		retVal = at
	case !safe && atok && bfok:
		vecScale(float64(1)/bf, at.data)
		retVal = at
	case !safe && afok && btok:
		vecDivBy(af, bt.data)
		retVal = bt
	default:
		err = types.NewError(types.DtypeMismatch, "Division cannot be done on %T and %T", a, b)
		return
	}
	return
}

// PointwisePow performs a pointwise a ^ b. Valid values are either float64 or *Tensor.
//
// If both operands are *Tensor, shape is checked first.
// Even though the underlying data may have the same size (say (2,2) vs (4,1)), if they have different shapes, it will error out.
//
// If the Unsafe flag is passed in, the data of the first tensor will be overwritten
func PointwisePow(a, b interface{}, opts ...types.FuncOpt) (retVal *Tensor, err error) {
	safe, incr, reuse := parseSafeReuse(opts...)

	at, atok := a.(*Tensor)
	bt, btok := b.(*Tensor)
	af, afok := a.(float64)
	bf, bfok := b.(float64)

	toReuse := reuse != nil

	if atok && btok {
		// assert that they have the same shape
		if !at.Shape().Eq(bt.Shape()) {
			err = types.NewError(types.ShapeMismatch, "Cannot add tensors with shapes %v and %v", at.Shape(), bt.Shape())
			return
		}
	}

	switch {
	case toReuse && atok:
		if err = reuseCheck(reuse, at); err != nil {
			return
		}
	case toReuse && btok:
		if err = reuseCheck(reuse, bt); err != nil {
			return
		}
	}

	switch {
	// incr
	case incr && atok && btok:
		incrVecPow(reuse.data, at.data, bt.data)
		retVal = reuse
	case incr && atok && bfok:
		incrVecPower(reuse.data, at.data, bf)
		retVal = reuse
	case incr && afok && btok:
		incrVecPowerFrom(reuse.data, bt.data, af)
		retVal = reuse

	//reuse
	case toReuse && atok && btok:
		safeVecPow(at.data, bt.data, reuse.data)
		retVal = reuse
	case toReuse && atok && bfok:
		safeVecPower(bf, at.data, reuse.data)
		retVal = reuse
	case toReuse && afok && btok:
		safeVecPowerFrom(af, bt.data, reuse.data)
		retVal = reuse

	// safe
	case safe && atok && btok:
		retVal = newBorrowedTensor(len(at.data))
		if err = retVal.Reshape(at.Shape()...); err != nil {
			return
		}
		safeVecPow(at.data, bt.data, retVal.data)
	case safe && atok && bfok:
		retVal = newBorrowedTensor(len(at.data))
		if err = retVal.Reshape(at.Shape()...); err != nil {
			return
		}
		safeVecPower(bf, at.data, retVal.data)
	case safe && afok && btok:
		retVal = newBorrowedTensor(len(bt.data))
		if err = retVal.Reshape(bt.Shape()...); err != nil {
			return
		}
		safeVecPowerFrom(af, bt.data, retVal.data)

	// unsafe
	case !safe && atok && btok:
		vecPow(at.data, bt.data)
		retVal = at
	case !safe && atok && bfok:
		vecPower(bf, at.data)
		retVal = at
	case !safe && afok && btok:
		vecPowerFrom(af, bt.data)
		retVal = bt
	default:
		err = types.NewError(types.DtypeMismatch, "Exponentiation cannot be done on %T and %T", a, b)
		return
	}
	return
}
