package tensor

// exported API for arithmetics and the stupidly crazy amount of overloaded semantics
// Add performs a pointwise a+b. a and b can either be float64 or Tensor
//
// If both operands are Tensor, shape is checked first.
// Even though the underlying data may have the same size (say (2,2) vs (4,1)), if they have different shapes, it will error out.
//
// If the Unsafe flag is passed in, the data of the first tensor will be overwritten

/*
func Add(a, b interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	safe, incr, reuse := parseSafeReuse(opts...)

	ad, adok := a.(*Dense)
	bd, bdok := b.(*Dense)
	af, afok := a.(float64)
	bf, bfok := b.(float64)

	toReuse := reuse != nil

	if adok && bdok {
		// assert that they have the same shape'
		if !ad.Shape().Eq(bd.Shape()) {
			err = NewError(ShapeMismatch, "Cannot add tensors with shapes %v and %v", ad.Shape(), bd.Shape())
			return
		}
	}

	switch {
	case toReuse && adok:
		if err = reuseCheck(reuse, ad); err != nil {
			return
		}
	case toReuse && bdok:
		if err = reuseCheck(reuse, bd); err != nil {
			return
		}
	}

	switch {
	// incr
	case incr && adok && bdok:
		if reuse == bd {
			vecAdd(reuse.data, bd.data)
			vecAdd(reuse.data, ad.data)
		} else {
			vecAdd(reuse.data, ad.data)
			vecAdd(reuse.data, bd.data)
		}
		retVal = reuse
	case incr && adok && bfok:
		vecAdd(reuse.data, ad.data)
		vecTrans(bf, reuse.data)
		retVal = reuse
	case incr && afok && bdok:
		vecAdd(reuse.data, bd.data)
		vecTrans(af, reuse.data)
		retVal = reuse

	//reuse
	case toReuse && adok && bdok:
		safeVecAdd(ad.data, bd.data, reuse.data)
		retVal = reuse
	case toReuse && adok && bfok:
		safeVecTrans(bf, ad.data, reuse.data)
		retVal = reuse
	case toReuse && afok && bdok:
		safeVecTrans(af, bd.data, reuse.data)
		retVal = reuse

	// safe
	case safe && adok && bdok:
		retVal = newBorrowedTensor(len(ad.data), WithShape(ad.Shape()...))
		safeVecAdd(ad.data, bd.data, retVal.data)
	case safe && adok && bfok:
		retVal = newBorrowedTensor(len(ad.data), WithShape(ad.Shape()...))
		safeVecTrans(bf, ad.data, retVal.data)
	case safe && afok && bdok:
		retVal = newBorrowedTensor(len(bd.data), WithShape(bd.Shape()...))
		safeVecTrans(af, bd.data, retVal.data)

	// unsafe
	case !safe && adok && bdok:
		vecAdd(ad.data, bd.data)
		retVal = ad
	case !safe && adok && bfok:
		vecTrans(bf, ad.data)
		retVal = ad
	case !safe && afok && bdok:
		vecTrans(af, bd.data)
		retVal = bd
	default:
		err = NewError(DtypeMismatch, "Addition cannot be done on %T and %T", a, b)
		return
	}
	return
}
*/
