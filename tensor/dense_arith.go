package tensor

import (
	"math"
	"math/cmplx"
	"reflect"

	"github.com/chewxy/math32"
	"github.com/pkg/errors"
)

/*
GENERATED FILE. DO NOT EDIT
*/

func prepBinaryDense(a, b *Dense, opts ...FuncOpt) (reuse *Dense, safe, toReuse, incr bool, err error) {
	if !isNumber(a.t) && !isNumber(b.t) {
		err = noopError{}
		return
	}
	if a.t.Kind() != b.t.Kind() {
		err = errors.Errorf(typeMismatch, a.t, b.t)
		return
	}
	if !a.Shape().Eq(b.Shape()) {
		err = errors.Errorf(shapeMismatch, b.Shape(), a.Shape())
		return
	}

	fo := ParseFuncOpts(opts...)
	reuseT, incr := fo.IncrReuse()
	safe = fo.Safe()
	toReuse = reuseT != nil

	if toReuse {
		if reuse, err = getDense(reuseT); err != nil {
			err = errors.Wrapf(err, "Cannot reuse a different type of Tensor in a *Dense-Scalar operation")
			return
		}

		if reuse.t.Kind() != a.t.Kind() {
			err = errors.Errorf(typeMismatch, a.t, reuse.t)
			err = errors.Wrapf(err, "Cannot use reuse")
			return
		}

		if reuse.len() != a.Shape().TotalSize() {
			err = errors.Errorf(shapeMismatch, reuse.Shape(), a.Shape())
			err = errors.Wrapf(err, "Cannot use reuse: shape mismatch")
			return
		}
	}
	return
}

func prepUnaryDense(a *Dense, opts ...FuncOpt) (reuse *Dense, safe, toReuse, incr bool, err error) {
	if !isNumber(a.t) {
		err = noopError{}
		return
	}

	fo := ParseFuncOpts(opts...)
	reuseT, incr := fo.IncrReuse()
	safe = fo.Safe()
	toReuse = reuseT != nil

	if toReuse {
		if reuse, err = getDense(reuseT); err != nil {
			err = errors.Wrapf(err, "Cannot reuse a different type of Tensor in a *Dense-Scalar operation")
			return
		}

		if reuse.t.Kind() != a.t.Kind() {
			err = errors.Errorf(typeMismatch, a.t, reuse.t)
			err = errors.Wrapf(err, "Cannot use reuse")
			return
		}

		if reuse.len() != a.Shape().TotalSize() {
			err = errors.Errorf(shapeMismatch, reuse.Shape(), a.Shape())
			err = errors.Wrapf(err, "Cannot use reuse: shape mismatch")
			return
		}
	}
	return
}

/* Add */

// Add performs the operation on another *Dense. It takes a list of FuncOpts.
func (t *Dense) Add(other *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	if t.e != nil {
		if add, ok := t.e.(Adder); ok {
			// if safe, then make a copy
			var ret Tensor
			if ret, err = add.Add(t, other, opts...); err != nil {
				goto attemptGo
			}
			retVal = ret.(*Dense)
			return
		}
	}

attemptGo:
	reuse, safe, toReuse, incr, err := prepBinaryDense(t, other, opts...)
	if err != nil {
		return nil, err
	}

	// check if the tensors are accessible
	if !t.isNativeAccessible() {
		err = errors.Errorf(inaccessibleData, t)
		return
	}

	if !other.isNativeAccessible() {
		err = errors.Errorf(inaccessibleData, reuse)
		return
	}

	if reuse != nil && !reuse.isNativeAccessible() {
		err = errors.Errorf(inaccessibleData, reuse)
		return
	}

	var it, ot *FlatMaskedIterator
	if t.IsMaterializable() {
		it = NewFlatMaskedIterator(t.AP, t.mask)
	}
	if other.IsMaterializable() {
		ot = NewFlatMaskedIterator(other.AP, other.mask)
	}
	switch {
	case incr:
		// when incr returned, the reuse is the *Dense to be incremented
		retVal = reuse
		retVal.MaskFromDense(t, other)
		if it != nil {
			it.mask = retVal.mask
		}
		if ot != nil {
			ot.mask = retVal.mask
		}
		isMasked := retVal.IsMasked()
		switch reuse.t.Kind() {
		case reflect.Int:
			data := reuse.Ints()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += t.GetI(i) + other.GetI(j)
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetI(i) + other.GetI(j)
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetI(i) + other.GetI(j)
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += t.GetI(i) + other.GetI(j)
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecAddIMasked(t.Ints(), other.Ints(), reuse.Ints(), reuse.mask)
					} else {
						err = incrVecAddI(t.Ints(), other.Ints(), reuse.Ints())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += t.GetI(i) + other.GetI(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += t.GetI(i) + other.GetI(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetI(i) + other.GetI(j)
						incrI += iterStep
					}
				}
			}
		case reflect.Int8:
			data := reuse.Int8s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += t.GetI8(i) + other.GetI8(j)
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetI8(i) + other.GetI8(j)
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetI8(i) + other.GetI8(j)
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += t.GetI8(i) + other.GetI8(j)
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecAddI8Masked(t.Int8s(), other.Int8s(), reuse.Int8s(), reuse.mask)
					} else {
						err = incrVecAddI8(t.Int8s(), other.Int8s(), reuse.Int8s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += t.GetI8(i) + other.GetI8(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += t.GetI8(i) + other.GetI8(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetI8(i) + other.GetI8(j)
						incrI += iterStep
					}
				}
			}
		case reflect.Int16:
			data := reuse.Int16s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += t.GetI16(i) + other.GetI16(j)
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetI16(i) + other.GetI16(j)
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetI16(i) + other.GetI16(j)
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += t.GetI16(i) + other.GetI16(j)
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecAddI16Masked(t.Int16s(), other.Int16s(), reuse.Int16s(), reuse.mask)
					} else {
						err = incrVecAddI16(t.Int16s(), other.Int16s(), reuse.Int16s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += t.GetI16(i) + other.GetI16(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += t.GetI16(i) + other.GetI16(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetI16(i) + other.GetI16(j)
						incrI += iterStep
					}
				}
			}
		case reflect.Int32:
			data := reuse.Int32s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += t.GetI32(i) + other.GetI32(j)
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetI32(i) + other.GetI32(j)
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetI32(i) + other.GetI32(j)
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += t.GetI32(i) + other.GetI32(j)
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecAddI32Masked(t.Int32s(), other.Int32s(), reuse.Int32s(), reuse.mask)
					} else {
						err = incrVecAddI32(t.Int32s(), other.Int32s(), reuse.Int32s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += t.GetI32(i) + other.GetI32(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += t.GetI32(i) + other.GetI32(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetI32(i) + other.GetI32(j)
						incrI += iterStep
					}
				}
			}
		case reflect.Int64:
			data := reuse.Int64s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += t.GetI64(i) + other.GetI64(j)
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetI64(i) + other.GetI64(j)
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetI64(i) + other.GetI64(j)
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += t.GetI64(i) + other.GetI64(j)
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecAddI64Masked(t.Int64s(), other.Int64s(), reuse.Int64s(), reuse.mask)
					} else {
						err = incrVecAddI64(t.Int64s(), other.Int64s(), reuse.Int64s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += t.GetI64(i) + other.GetI64(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += t.GetI64(i) + other.GetI64(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetI64(i) + other.GetI64(j)
						incrI += iterStep
					}
				}
			}
		case reflect.Uint:
			data := reuse.Uints()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += t.GetU(i) + other.GetU(j)
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetU(i) + other.GetU(j)
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetU(i) + other.GetU(j)
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += t.GetU(i) + other.GetU(j)
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecAddUMasked(t.Uints(), other.Uints(), reuse.Uints(), reuse.mask)
					} else {
						err = incrVecAddU(t.Uints(), other.Uints(), reuse.Uints())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += t.GetU(i) + other.GetU(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += t.GetU(i) + other.GetU(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetU(i) + other.GetU(j)
						incrI += iterStep
					}
				}
			}
		case reflect.Uint8:
			data := reuse.Uint8s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += t.GetU8(i) + other.GetU8(j)
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetU8(i) + other.GetU8(j)
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetU8(i) + other.GetU8(j)
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += t.GetU8(i) + other.GetU8(j)
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecAddU8Masked(t.Uint8s(), other.Uint8s(), reuse.Uint8s(), reuse.mask)
					} else {
						err = incrVecAddU8(t.Uint8s(), other.Uint8s(), reuse.Uint8s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += t.GetU8(i) + other.GetU8(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += t.GetU8(i) + other.GetU8(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetU8(i) + other.GetU8(j)
						incrI += iterStep
					}
				}
			}
		case reflect.Uint16:
			data := reuse.Uint16s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += t.GetU16(i) + other.GetU16(j)
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetU16(i) + other.GetU16(j)
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetU16(i) + other.GetU16(j)
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += t.GetU16(i) + other.GetU16(j)
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecAddU16Masked(t.Uint16s(), other.Uint16s(), reuse.Uint16s(), reuse.mask)
					} else {
						err = incrVecAddU16(t.Uint16s(), other.Uint16s(), reuse.Uint16s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += t.GetU16(i) + other.GetU16(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += t.GetU16(i) + other.GetU16(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetU16(i) + other.GetU16(j)
						incrI += iterStep
					}
				}
			}
		case reflect.Uint32:
			data := reuse.Uint32s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += t.GetU32(i) + other.GetU32(j)
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetU32(i) + other.GetU32(j)
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetU32(i) + other.GetU32(j)
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += t.GetU32(i) + other.GetU32(j)
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecAddU32Masked(t.Uint32s(), other.Uint32s(), reuse.Uint32s(), reuse.mask)
					} else {
						err = incrVecAddU32(t.Uint32s(), other.Uint32s(), reuse.Uint32s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += t.GetU32(i) + other.GetU32(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += t.GetU32(i) + other.GetU32(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetU32(i) + other.GetU32(j)
						incrI += iterStep
					}
				}
			}
		case reflect.Uint64:
			data := reuse.Uint64s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += t.GetU64(i) + other.GetU64(j)
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetU64(i) + other.GetU64(j)
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetU64(i) + other.GetU64(j)
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += t.GetU64(i) + other.GetU64(j)
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecAddU64Masked(t.Uint64s(), other.Uint64s(), reuse.Uint64s(), reuse.mask)
					} else {
						err = incrVecAddU64(t.Uint64s(), other.Uint64s(), reuse.Uint64s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += t.GetU64(i) + other.GetU64(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += t.GetU64(i) + other.GetU64(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetU64(i) + other.GetU64(j)
						incrI += iterStep
					}
				}
			}
		case reflect.Float32:
			data := reuse.Float32s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += t.GetF32(i) + other.GetF32(j)
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetF32(i) + other.GetF32(j)
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetF32(i) + other.GetF32(j)
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += t.GetF32(i) + other.GetF32(j)
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecAddF32Masked(t.Float32s(), other.Float32s(), reuse.Float32s(), reuse.mask)
					} else {
						err = incrVecAddF32(t.Float32s(), other.Float32s(), reuse.Float32s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += t.GetF32(i) + other.GetF32(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += t.GetF32(i) + other.GetF32(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetF32(i) + other.GetF32(j)
						incrI += iterStep
					}
				}
			}
		case reflect.Float64:
			data := reuse.Float64s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += t.GetF64(i) + other.GetF64(j)
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetF64(i) + other.GetF64(j)
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetF64(i) + other.GetF64(j)
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += t.GetF64(i) + other.GetF64(j)
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecAddF64Masked(t.Float64s(), other.Float64s(), reuse.Float64s(), reuse.mask)
					} else {
						err = incrVecAddF64(t.Float64s(), other.Float64s(), reuse.Float64s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += t.GetF64(i) + other.GetF64(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += t.GetF64(i) + other.GetF64(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetF64(i) + other.GetF64(j)
						incrI += iterStep
					}
				}
			}
		case reflect.Complex64:
			data := reuse.Complex64s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += t.GetC64(i) + other.GetC64(j)
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetC64(i) + other.GetC64(j)
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetC64(i) + other.GetC64(j)
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += t.GetC64(i) + other.GetC64(j)
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecAddC64Masked(t.Complex64s(), other.Complex64s(), reuse.Complex64s(), reuse.mask)
					} else {
						err = incrVecAddC64(t.Complex64s(), other.Complex64s(), reuse.Complex64s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += t.GetC64(i) + other.GetC64(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += t.GetC64(i) + other.GetC64(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetC64(i) + other.GetC64(j)
						incrI += iterStep
					}
				}
			}
		case reflect.Complex128:
			data := reuse.Complex128s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += t.GetC128(i) + other.GetC128(j)
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetC128(i) + other.GetC128(j)
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetC128(i) + other.GetC128(j)
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += t.GetC128(i) + other.GetC128(j)
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecAddC128Masked(t.Complex128s(), other.Complex128s(), reuse.Complex128s(), reuse.mask)
					} else {
						err = incrVecAddC128(t.Complex128s(), other.Complex128s(), reuse.Complex128s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += t.GetC128(i) + other.GetC128(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += t.GetC128(i) + other.GetC128(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetC128(i) + other.GetC128(j)
						incrI += iterStep
					}
				}
			}
		}
	case toReuse:
		if t.IsMaterializable() {
			copyDenseIter(reuse, t, nil, it.FlatIterator)
		} else {
			copyDense(reuse, t) // technically copyDenseIter would have done the same but it's much slower
		}
		err = reuse.add(other, nil, ot)
		retVal = reuse
	case safe:
		if t.IsMaterializable() {
			retVal = t.Materialize().(*Dense)
		} else {
			retVal = t.Clone().(*Dense)
		}
		err = retVal.add(other, nil, ot)
	case !safe:
		err = t.add(other, it, ot)
		retVal = t
	}
	if it != nil {
		it.mask = nil
	}
	if ot != nil {
		ot.mask = nil
	}
	return
}
func (t *Dense) add(other *Dense, itt, ott Iterator) (err error) {
	var it, ot *FlatMaskedIterator
	if itt != nil {
		it = new(FlatMaskedIterator)
		switch iter := itt.(type) {
		case *FlatIterator:
			it.FlatIterator = iter
		case *FlatMaskedIterator:
			it = iter
		case *MultIterator:
			it.FlatIterator = iter.fit0
			it.mask = iter.mask
		}
	}
	if ott != nil {
		ot = new(FlatMaskedIterator)
		switch iter := ott.(type) {
		case *FlatIterator:
			ot.FlatIterator = iter
		case *FlatMaskedIterator:
			ot = iter
		case *MultIterator:
			ot.FlatIterator = iter.fit0
			ot.mask = iter.mask
		}
	}

	t.MaskFromDense(t, other)

	if it != nil {
		it.mask = t.mask
	}
	if ot != nil {
		ot.mask = t.mask
	}

	switch t.t.Kind() {
	case reflect.Int:
		tdata := t.Ints()
		odata := other.Ints()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = tdata[i] + odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = tdata[i] + odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = tdata[i] + odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecAddIMasked(tdata, odata, t.mask)
			} else {
				vecAddI(tdata, odata)
			}
		}
	case reflect.Int8:
		tdata := t.Int8s()
		odata := other.Int8s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = tdata[i] + odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = tdata[i] + odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = tdata[i] + odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecAddI8Masked(tdata, odata, t.mask)
			} else {
				vecAddI8(tdata, odata)
			}
		}
	case reflect.Int16:
		tdata := t.Int16s()
		odata := other.Int16s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = tdata[i] + odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = tdata[i] + odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = tdata[i] + odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecAddI16Masked(tdata, odata, t.mask)
			} else {
				vecAddI16(tdata, odata)
			}
		}
	case reflect.Int32:
		tdata := t.Int32s()
		odata := other.Int32s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = tdata[i] + odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = tdata[i] + odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = tdata[i] + odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecAddI32Masked(tdata, odata, t.mask)
			} else {
				vecAddI32(tdata, odata)
			}
		}
	case reflect.Int64:
		tdata := t.Int64s()
		odata := other.Int64s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = tdata[i] + odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = tdata[i] + odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = tdata[i] + odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecAddI64Masked(tdata, odata, t.mask)
			} else {
				vecAddI64(tdata, odata)
			}
		}
	case reflect.Uint:
		tdata := t.Uints()
		odata := other.Uints()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = tdata[i] + odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = tdata[i] + odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = tdata[i] + odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecAddUMasked(tdata, odata, t.mask)
			} else {
				vecAddU(tdata, odata)
			}
		}
	case reflect.Uint8:
		tdata := t.Uint8s()
		odata := other.Uint8s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = tdata[i] + odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = tdata[i] + odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = tdata[i] + odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecAddU8Masked(tdata, odata, t.mask)
			} else {
				vecAddU8(tdata, odata)
			}
		}
	case reflect.Uint16:
		tdata := t.Uint16s()
		odata := other.Uint16s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = tdata[i] + odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = tdata[i] + odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = tdata[i] + odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecAddU16Masked(tdata, odata, t.mask)
			} else {
				vecAddU16(tdata, odata)
			}
		}
	case reflect.Uint32:
		tdata := t.Uint32s()
		odata := other.Uint32s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = tdata[i] + odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = tdata[i] + odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = tdata[i] + odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecAddU32Masked(tdata, odata, t.mask)
			} else {
				vecAddU32(tdata, odata)
			}
		}
	case reflect.Uint64:
		tdata := t.Uint64s()
		odata := other.Uint64s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = tdata[i] + odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = tdata[i] + odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = tdata[i] + odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecAddU64Masked(tdata, odata, t.mask)
			} else {
				vecAddU64(tdata, odata)
			}
		}
	case reflect.Float32:
		tdata := t.Float32s()
		odata := other.Float32s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = tdata[i] + odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = tdata[i] + odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = tdata[i] + odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecAddF32Masked(tdata, odata, t.mask)
			} else {
				vecAddF32(tdata, odata)
			}
		}
	case reflect.Float64:
		tdata := t.Float64s()
		odata := other.Float64s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = tdata[i] + odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = tdata[i] + odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = tdata[i] + odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecAddF64Masked(tdata, odata, t.mask)
			} else {
				vecAddF64(tdata, odata)
			}
		}
	case reflect.Complex64:
		tdata := t.Complex64s()
		odata := other.Complex64s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = tdata[i] + odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = tdata[i] + odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = tdata[i] + odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecAddC64Masked(tdata, odata, t.mask)
			} else {
				vecAddC64(tdata, odata)
			}
		}
	case reflect.Complex128:
		tdata := t.Complex128s()
		odata := other.Complex128s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = tdata[i] + odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = tdata[i] + odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = tdata[i] + odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecAddC128Masked(tdata, odata, t.mask)
			} else {
				vecAddC128(tdata, odata)
			}
		}
	default:
		// TODO: Handle Number interface
	}

	return
}

/* Sub */

// Sub performs the operation on another *Dense. It takes a list of FuncOpts.
func (t *Dense) Sub(other *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	if t.e != nil {
		if sub, ok := t.e.(Suber); ok {
			// if safe, then make a copy
			var ret Tensor
			if ret, err = sub.Sub(t, other, opts...); err != nil {
				goto attemptGo
			}
			retVal = ret.(*Dense)
			return
		}
	}

attemptGo:
	reuse, safe, toReuse, incr, err := prepBinaryDense(t, other, opts...)
	if err != nil {
		return nil, err
	}

	// check if the tensors are accessible
	if !t.isNativeAccessible() {
		err = errors.Errorf(inaccessibleData, t)
		return
	}

	if !other.isNativeAccessible() {
		err = errors.Errorf(inaccessibleData, reuse)
		return
	}

	if reuse != nil && !reuse.isNativeAccessible() {
		err = errors.Errorf(inaccessibleData, reuse)
		return
	}

	var it, ot *FlatMaskedIterator
	if t.IsMaterializable() {
		it = NewFlatMaskedIterator(t.AP, t.mask)
	}
	if other.IsMaterializable() {
		ot = NewFlatMaskedIterator(other.AP, other.mask)
	}
	switch {
	case incr:
		// when incr returned, the reuse is the *Dense to be incremented
		retVal = reuse
		retVal.MaskFromDense(t, other)
		if it != nil {
			it.mask = retVal.mask
		}
		if ot != nil {
			ot.mask = retVal.mask
		}
		isMasked := retVal.IsMasked()
		switch reuse.t.Kind() {
		case reflect.Int:
			data := reuse.Ints()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += t.GetI(i) - other.GetI(j)
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetI(i) - other.GetI(j)
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetI(i) - other.GetI(j)
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += t.GetI(i) - other.GetI(j)
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecSubIMasked(t.Ints(), other.Ints(), reuse.Ints(), reuse.mask)
					} else {
						err = incrVecSubI(t.Ints(), other.Ints(), reuse.Ints())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += t.GetI(i) - other.GetI(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += t.GetI(i) - other.GetI(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetI(i) - other.GetI(j)
						incrI += iterStep
					}
				}
			}
		case reflect.Int8:
			data := reuse.Int8s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += t.GetI8(i) - other.GetI8(j)
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetI8(i) - other.GetI8(j)
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetI8(i) - other.GetI8(j)
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += t.GetI8(i) - other.GetI8(j)
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecSubI8Masked(t.Int8s(), other.Int8s(), reuse.Int8s(), reuse.mask)
					} else {
						err = incrVecSubI8(t.Int8s(), other.Int8s(), reuse.Int8s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += t.GetI8(i) - other.GetI8(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += t.GetI8(i) - other.GetI8(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetI8(i) - other.GetI8(j)
						incrI += iterStep
					}
				}
			}
		case reflect.Int16:
			data := reuse.Int16s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += t.GetI16(i) - other.GetI16(j)
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetI16(i) - other.GetI16(j)
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetI16(i) - other.GetI16(j)
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += t.GetI16(i) - other.GetI16(j)
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecSubI16Masked(t.Int16s(), other.Int16s(), reuse.Int16s(), reuse.mask)
					} else {
						err = incrVecSubI16(t.Int16s(), other.Int16s(), reuse.Int16s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += t.GetI16(i) - other.GetI16(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += t.GetI16(i) - other.GetI16(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetI16(i) - other.GetI16(j)
						incrI += iterStep
					}
				}
			}
		case reflect.Int32:
			data := reuse.Int32s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += t.GetI32(i) - other.GetI32(j)
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetI32(i) - other.GetI32(j)
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetI32(i) - other.GetI32(j)
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += t.GetI32(i) - other.GetI32(j)
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecSubI32Masked(t.Int32s(), other.Int32s(), reuse.Int32s(), reuse.mask)
					} else {
						err = incrVecSubI32(t.Int32s(), other.Int32s(), reuse.Int32s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += t.GetI32(i) - other.GetI32(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += t.GetI32(i) - other.GetI32(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetI32(i) - other.GetI32(j)
						incrI += iterStep
					}
				}
			}
		case reflect.Int64:
			data := reuse.Int64s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += t.GetI64(i) - other.GetI64(j)
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetI64(i) - other.GetI64(j)
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetI64(i) - other.GetI64(j)
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += t.GetI64(i) - other.GetI64(j)
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecSubI64Masked(t.Int64s(), other.Int64s(), reuse.Int64s(), reuse.mask)
					} else {
						err = incrVecSubI64(t.Int64s(), other.Int64s(), reuse.Int64s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += t.GetI64(i) - other.GetI64(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += t.GetI64(i) - other.GetI64(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetI64(i) - other.GetI64(j)
						incrI += iterStep
					}
				}
			}
		case reflect.Uint:
			data := reuse.Uints()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += t.GetU(i) - other.GetU(j)
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetU(i) - other.GetU(j)
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetU(i) - other.GetU(j)
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += t.GetU(i) - other.GetU(j)
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecSubUMasked(t.Uints(), other.Uints(), reuse.Uints(), reuse.mask)
					} else {
						err = incrVecSubU(t.Uints(), other.Uints(), reuse.Uints())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += t.GetU(i) - other.GetU(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += t.GetU(i) - other.GetU(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetU(i) - other.GetU(j)
						incrI += iterStep
					}
				}
			}
		case reflect.Uint8:
			data := reuse.Uint8s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += t.GetU8(i) - other.GetU8(j)
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetU8(i) - other.GetU8(j)
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetU8(i) - other.GetU8(j)
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += t.GetU8(i) - other.GetU8(j)
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecSubU8Masked(t.Uint8s(), other.Uint8s(), reuse.Uint8s(), reuse.mask)
					} else {
						err = incrVecSubU8(t.Uint8s(), other.Uint8s(), reuse.Uint8s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += t.GetU8(i) - other.GetU8(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += t.GetU8(i) - other.GetU8(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetU8(i) - other.GetU8(j)
						incrI += iterStep
					}
				}
			}
		case reflect.Uint16:
			data := reuse.Uint16s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += t.GetU16(i) - other.GetU16(j)
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetU16(i) - other.GetU16(j)
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetU16(i) - other.GetU16(j)
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += t.GetU16(i) - other.GetU16(j)
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecSubU16Masked(t.Uint16s(), other.Uint16s(), reuse.Uint16s(), reuse.mask)
					} else {
						err = incrVecSubU16(t.Uint16s(), other.Uint16s(), reuse.Uint16s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += t.GetU16(i) - other.GetU16(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += t.GetU16(i) - other.GetU16(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetU16(i) - other.GetU16(j)
						incrI += iterStep
					}
				}
			}
		case reflect.Uint32:
			data := reuse.Uint32s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += t.GetU32(i) - other.GetU32(j)
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetU32(i) - other.GetU32(j)
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetU32(i) - other.GetU32(j)
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += t.GetU32(i) - other.GetU32(j)
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecSubU32Masked(t.Uint32s(), other.Uint32s(), reuse.Uint32s(), reuse.mask)
					} else {
						err = incrVecSubU32(t.Uint32s(), other.Uint32s(), reuse.Uint32s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += t.GetU32(i) - other.GetU32(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += t.GetU32(i) - other.GetU32(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetU32(i) - other.GetU32(j)
						incrI += iterStep
					}
				}
			}
		case reflect.Uint64:
			data := reuse.Uint64s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += t.GetU64(i) - other.GetU64(j)
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetU64(i) - other.GetU64(j)
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetU64(i) - other.GetU64(j)
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += t.GetU64(i) - other.GetU64(j)
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecSubU64Masked(t.Uint64s(), other.Uint64s(), reuse.Uint64s(), reuse.mask)
					} else {
						err = incrVecSubU64(t.Uint64s(), other.Uint64s(), reuse.Uint64s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += t.GetU64(i) - other.GetU64(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += t.GetU64(i) - other.GetU64(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetU64(i) - other.GetU64(j)
						incrI += iterStep
					}
				}
			}
		case reflect.Float32:
			data := reuse.Float32s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += t.GetF32(i) - other.GetF32(j)
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetF32(i) - other.GetF32(j)
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetF32(i) - other.GetF32(j)
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += t.GetF32(i) - other.GetF32(j)
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecSubF32Masked(t.Float32s(), other.Float32s(), reuse.Float32s(), reuse.mask)
					} else {
						err = incrVecSubF32(t.Float32s(), other.Float32s(), reuse.Float32s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += t.GetF32(i) - other.GetF32(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += t.GetF32(i) - other.GetF32(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetF32(i) - other.GetF32(j)
						incrI += iterStep
					}
				}
			}
		case reflect.Float64:
			data := reuse.Float64s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += t.GetF64(i) - other.GetF64(j)
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetF64(i) - other.GetF64(j)
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetF64(i) - other.GetF64(j)
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += t.GetF64(i) - other.GetF64(j)
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecSubF64Masked(t.Float64s(), other.Float64s(), reuse.Float64s(), reuse.mask)
					} else {
						err = incrVecSubF64(t.Float64s(), other.Float64s(), reuse.Float64s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += t.GetF64(i) - other.GetF64(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += t.GetF64(i) - other.GetF64(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetF64(i) - other.GetF64(j)
						incrI += iterStep
					}
				}
			}
		case reflect.Complex64:
			data := reuse.Complex64s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += t.GetC64(i) - other.GetC64(j)
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetC64(i) - other.GetC64(j)
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetC64(i) - other.GetC64(j)
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += t.GetC64(i) - other.GetC64(j)
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecSubC64Masked(t.Complex64s(), other.Complex64s(), reuse.Complex64s(), reuse.mask)
					} else {
						err = incrVecSubC64(t.Complex64s(), other.Complex64s(), reuse.Complex64s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += t.GetC64(i) - other.GetC64(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += t.GetC64(i) - other.GetC64(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetC64(i) - other.GetC64(j)
						incrI += iterStep
					}
				}
			}
		case reflect.Complex128:
			data := reuse.Complex128s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += t.GetC128(i) - other.GetC128(j)
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetC128(i) - other.GetC128(j)
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetC128(i) - other.GetC128(j)
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += t.GetC128(i) - other.GetC128(j)
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecSubC128Masked(t.Complex128s(), other.Complex128s(), reuse.Complex128s(), reuse.mask)
					} else {
						err = incrVecSubC128(t.Complex128s(), other.Complex128s(), reuse.Complex128s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += t.GetC128(i) - other.GetC128(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += t.GetC128(i) - other.GetC128(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetC128(i) - other.GetC128(j)
						incrI += iterStep
					}
				}
			}
		}
	case toReuse:
		if t.IsMaterializable() {
			copyDenseIter(reuse, t, nil, it.FlatIterator)
		} else {
			copyDense(reuse, t) // technically copyDenseIter would have done the same but it's much slower
		}
		err = reuse.sub(other, nil, ot)
		retVal = reuse
	case safe:
		if t.IsMaterializable() {
			retVal = t.Materialize().(*Dense)
		} else {
			retVal = t.Clone().(*Dense)
		}
		err = retVal.sub(other, nil, ot)
	case !safe:
		err = t.sub(other, it, ot)
		retVal = t
	}
	if it != nil {
		it.mask = nil
	}
	if ot != nil {
		ot.mask = nil
	}
	return
}
func (t *Dense) sub(other *Dense, itt, ott Iterator) (err error) {
	var it, ot *FlatMaskedIterator
	if itt != nil {
		it = new(FlatMaskedIterator)
		switch iter := itt.(type) {
		case *FlatIterator:
			it.FlatIterator = iter
		case *FlatMaskedIterator:
			it = iter
		case *MultIterator:
			it.FlatIterator = iter.fit0
			it.mask = iter.mask
		}
	}
	if ott != nil {
		ot = new(FlatMaskedIterator)
		switch iter := ott.(type) {
		case *FlatIterator:
			ot.FlatIterator = iter
		case *FlatMaskedIterator:
			ot = iter
		case *MultIterator:
			ot.FlatIterator = iter.fit0
			ot.mask = iter.mask
		}
	}

	t.MaskFromDense(t, other)

	if it != nil {
		it.mask = t.mask
	}
	if ot != nil {
		ot.mask = t.mask
	}

	switch t.t.Kind() {
	case reflect.Int:
		tdata := t.Ints()
		odata := other.Ints()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = tdata[i] - odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = tdata[i] - odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = tdata[i] - odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecSubIMasked(tdata, odata, t.mask)
			} else {
				vecSubI(tdata, odata)
			}
		}
	case reflect.Int8:
		tdata := t.Int8s()
		odata := other.Int8s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = tdata[i] - odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = tdata[i] - odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = tdata[i] - odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecSubI8Masked(tdata, odata, t.mask)
			} else {
				vecSubI8(tdata, odata)
			}
		}
	case reflect.Int16:
		tdata := t.Int16s()
		odata := other.Int16s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = tdata[i] - odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = tdata[i] - odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = tdata[i] - odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecSubI16Masked(tdata, odata, t.mask)
			} else {
				vecSubI16(tdata, odata)
			}
		}
	case reflect.Int32:
		tdata := t.Int32s()
		odata := other.Int32s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = tdata[i] - odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = tdata[i] - odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = tdata[i] - odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecSubI32Masked(tdata, odata, t.mask)
			} else {
				vecSubI32(tdata, odata)
			}
		}
	case reflect.Int64:
		tdata := t.Int64s()
		odata := other.Int64s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = tdata[i] - odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = tdata[i] - odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = tdata[i] - odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecSubI64Masked(tdata, odata, t.mask)
			} else {
				vecSubI64(tdata, odata)
			}
		}
	case reflect.Uint:
		tdata := t.Uints()
		odata := other.Uints()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = tdata[i] - odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = tdata[i] - odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = tdata[i] - odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecSubUMasked(tdata, odata, t.mask)
			} else {
				vecSubU(tdata, odata)
			}
		}
	case reflect.Uint8:
		tdata := t.Uint8s()
		odata := other.Uint8s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = tdata[i] - odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = tdata[i] - odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = tdata[i] - odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecSubU8Masked(tdata, odata, t.mask)
			} else {
				vecSubU8(tdata, odata)
			}
		}
	case reflect.Uint16:
		tdata := t.Uint16s()
		odata := other.Uint16s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = tdata[i] - odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = tdata[i] - odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = tdata[i] - odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecSubU16Masked(tdata, odata, t.mask)
			} else {
				vecSubU16(tdata, odata)
			}
		}
	case reflect.Uint32:
		tdata := t.Uint32s()
		odata := other.Uint32s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = tdata[i] - odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = tdata[i] - odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = tdata[i] - odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecSubU32Masked(tdata, odata, t.mask)
			} else {
				vecSubU32(tdata, odata)
			}
		}
	case reflect.Uint64:
		tdata := t.Uint64s()
		odata := other.Uint64s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = tdata[i] - odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = tdata[i] - odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = tdata[i] - odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecSubU64Masked(tdata, odata, t.mask)
			} else {
				vecSubU64(tdata, odata)
			}
		}
	case reflect.Float32:
		tdata := t.Float32s()
		odata := other.Float32s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = tdata[i] - odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = tdata[i] - odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = tdata[i] - odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecSubF32Masked(tdata, odata, t.mask)
			} else {
				vecSubF32(tdata, odata)
			}
		}
	case reflect.Float64:
		tdata := t.Float64s()
		odata := other.Float64s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = tdata[i] - odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = tdata[i] - odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = tdata[i] - odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecSubF64Masked(tdata, odata, t.mask)
			} else {
				vecSubF64(tdata, odata)
			}
		}
	case reflect.Complex64:
		tdata := t.Complex64s()
		odata := other.Complex64s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = tdata[i] - odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = tdata[i] - odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = tdata[i] - odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecSubC64Masked(tdata, odata, t.mask)
			} else {
				vecSubC64(tdata, odata)
			}
		}
	case reflect.Complex128:
		tdata := t.Complex128s()
		odata := other.Complex128s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = tdata[i] - odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = tdata[i] - odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = tdata[i] - odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecSubC128Masked(tdata, odata, t.mask)
			} else {
				vecSubC128(tdata, odata)
			}
		}
	default:
		// TODO: Handle Number interface
	}

	return
}

/* Mul */

// Mul performs the operation on another *Dense. It takes a list of FuncOpts.
func (t *Dense) Mul(other *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	if t.e != nil {
		if mul, ok := t.e.(Muler); ok {
			// if safe, then make a copy
			var ret Tensor
			if ret, err = mul.Mul(t, other, opts...); err != nil {
				goto attemptGo
			}
			retVal = ret.(*Dense)
			return
		}
	}

attemptGo:
	reuse, safe, toReuse, incr, err := prepBinaryDense(t, other, opts...)
	if err != nil {
		return nil, err
	}

	// check if the tensors are accessible
	if !t.isNativeAccessible() {
		err = errors.Errorf(inaccessibleData, t)
		return
	}

	if !other.isNativeAccessible() {
		err = errors.Errorf(inaccessibleData, reuse)
		return
	}

	if reuse != nil && !reuse.isNativeAccessible() {
		err = errors.Errorf(inaccessibleData, reuse)
		return
	}

	var it, ot *FlatMaskedIterator
	if t.IsMaterializable() {
		it = NewFlatMaskedIterator(t.AP, t.mask)
	}
	if other.IsMaterializable() {
		ot = NewFlatMaskedIterator(other.AP, other.mask)
	}
	switch {
	case incr:
		// when incr returned, the reuse is the *Dense to be incremented
		retVal = reuse
		retVal.MaskFromDense(t, other)
		if it != nil {
			it.mask = retVal.mask
		}
		if ot != nil {
			ot.mask = retVal.mask
		}
		isMasked := retVal.IsMasked()
		switch reuse.t.Kind() {
		case reflect.Int:
			data := reuse.Ints()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += t.GetI(i) * other.GetI(j)
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetI(i) * other.GetI(j)
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetI(i) * other.GetI(j)
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += t.GetI(i) * other.GetI(j)
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecMulIMasked(t.Ints(), other.Ints(), reuse.Ints(), reuse.mask)
					} else {
						err = incrVecMulI(t.Ints(), other.Ints(), reuse.Ints())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += t.GetI(i) * other.GetI(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += t.GetI(i) * other.GetI(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetI(i) * other.GetI(j)
						incrI += iterStep
					}
				}
			}
		case reflect.Int8:
			data := reuse.Int8s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += t.GetI8(i) * other.GetI8(j)
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetI8(i) * other.GetI8(j)
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetI8(i) * other.GetI8(j)
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += t.GetI8(i) * other.GetI8(j)
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecMulI8Masked(t.Int8s(), other.Int8s(), reuse.Int8s(), reuse.mask)
					} else {
						err = incrVecMulI8(t.Int8s(), other.Int8s(), reuse.Int8s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += t.GetI8(i) * other.GetI8(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += t.GetI8(i) * other.GetI8(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetI8(i) * other.GetI8(j)
						incrI += iterStep
					}
				}
			}
		case reflect.Int16:
			data := reuse.Int16s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += t.GetI16(i) * other.GetI16(j)
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetI16(i) * other.GetI16(j)
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetI16(i) * other.GetI16(j)
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += t.GetI16(i) * other.GetI16(j)
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecMulI16Masked(t.Int16s(), other.Int16s(), reuse.Int16s(), reuse.mask)
					} else {
						err = incrVecMulI16(t.Int16s(), other.Int16s(), reuse.Int16s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += t.GetI16(i) * other.GetI16(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += t.GetI16(i) * other.GetI16(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetI16(i) * other.GetI16(j)
						incrI += iterStep
					}
				}
			}
		case reflect.Int32:
			data := reuse.Int32s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += t.GetI32(i) * other.GetI32(j)
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetI32(i) * other.GetI32(j)
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetI32(i) * other.GetI32(j)
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += t.GetI32(i) * other.GetI32(j)
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecMulI32Masked(t.Int32s(), other.Int32s(), reuse.Int32s(), reuse.mask)
					} else {
						err = incrVecMulI32(t.Int32s(), other.Int32s(), reuse.Int32s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += t.GetI32(i) * other.GetI32(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += t.GetI32(i) * other.GetI32(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetI32(i) * other.GetI32(j)
						incrI += iterStep
					}
				}
			}
		case reflect.Int64:
			data := reuse.Int64s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += t.GetI64(i) * other.GetI64(j)
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetI64(i) * other.GetI64(j)
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetI64(i) * other.GetI64(j)
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += t.GetI64(i) * other.GetI64(j)
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecMulI64Masked(t.Int64s(), other.Int64s(), reuse.Int64s(), reuse.mask)
					} else {
						err = incrVecMulI64(t.Int64s(), other.Int64s(), reuse.Int64s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += t.GetI64(i) * other.GetI64(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += t.GetI64(i) * other.GetI64(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetI64(i) * other.GetI64(j)
						incrI += iterStep
					}
				}
			}
		case reflect.Uint:
			data := reuse.Uints()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += t.GetU(i) * other.GetU(j)
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetU(i) * other.GetU(j)
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetU(i) * other.GetU(j)
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += t.GetU(i) * other.GetU(j)
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecMulUMasked(t.Uints(), other.Uints(), reuse.Uints(), reuse.mask)
					} else {
						err = incrVecMulU(t.Uints(), other.Uints(), reuse.Uints())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += t.GetU(i) * other.GetU(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += t.GetU(i) * other.GetU(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetU(i) * other.GetU(j)
						incrI += iterStep
					}
				}
			}
		case reflect.Uint8:
			data := reuse.Uint8s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += t.GetU8(i) * other.GetU8(j)
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetU8(i) * other.GetU8(j)
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetU8(i) * other.GetU8(j)
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += t.GetU8(i) * other.GetU8(j)
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecMulU8Masked(t.Uint8s(), other.Uint8s(), reuse.Uint8s(), reuse.mask)
					} else {
						err = incrVecMulU8(t.Uint8s(), other.Uint8s(), reuse.Uint8s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += t.GetU8(i) * other.GetU8(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += t.GetU8(i) * other.GetU8(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetU8(i) * other.GetU8(j)
						incrI += iterStep
					}
				}
			}
		case reflect.Uint16:
			data := reuse.Uint16s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += t.GetU16(i) * other.GetU16(j)
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetU16(i) * other.GetU16(j)
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetU16(i) * other.GetU16(j)
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += t.GetU16(i) * other.GetU16(j)
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecMulU16Masked(t.Uint16s(), other.Uint16s(), reuse.Uint16s(), reuse.mask)
					} else {
						err = incrVecMulU16(t.Uint16s(), other.Uint16s(), reuse.Uint16s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += t.GetU16(i) * other.GetU16(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += t.GetU16(i) * other.GetU16(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetU16(i) * other.GetU16(j)
						incrI += iterStep
					}
				}
			}
		case reflect.Uint32:
			data := reuse.Uint32s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += t.GetU32(i) * other.GetU32(j)
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetU32(i) * other.GetU32(j)
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetU32(i) * other.GetU32(j)
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += t.GetU32(i) * other.GetU32(j)
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecMulU32Masked(t.Uint32s(), other.Uint32s(), reuse.Uint32s(), reuse.mask)
					} else {
						err = incrVecMulU32(t.Uint32s(), other.Uint32s(), reuse.Uint32s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += t.GetU32(i) * other.GetU32(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += t.GetU32(i) * other.GetU32(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetU32(i) * other.GetU32(j)
						incrI += iterStep
					}
				}
			}
		case reflect.Uint64:
			data := reuse.Uint64s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += t.GetU64(i) * other.GetU64(j)
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetU64(i) * other.GetU64(j)
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetU64(i) * other.GetU64(j)
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += t.GetU64(i) * other.GetU64(j)
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecMulU64Masked(t.Uint64s(), other.Uint64s(), reuse.Uint64s(), reuse.mask)
					} else {
						err = incrVecMulU64(t.Uint64s(), other.Uint64s(), reuse.Uint64s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += t.GetU64(i) * other.GetU64(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += t.GetU64(i) * other.GetU64(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetU64(i) * other.GetU64(j)
						incrI += iterStep
					}
				}
			}
		case reflect.Float32:
			data := reuse.Float32s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += t.GetF32(i) * other.GetF32(j)
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetF32(i) * other.GetF32(j)
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetF32(i) * other.GetF32(j)
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += t.GetF32(i) * other.GetF32(j)
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecMulF32Masked(t.Float32s(), other.Float32s(), reuse.Float32s(), reuse.mask)
					} else {
						err = incrVecMulF32(t.Float32s(), other.Float32s(), reuse.Float32s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += t.GetF32(i) * other.GetF32(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += t.GetF32(i) * other.GetF32(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetF32(i) * other.GetF32(j)
						incrI += iterStep
					}
				}
			}
		case reflect.Float64:
			data := reuse.Float64s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += t.GetF64(i) * other.GetF64(j)
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetF64(i) * other.GetF64(j)
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetF64(i) * other.GetF64(j)
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += t.GetF64(i) * other.GetF64(j)
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecMulF64Masked(t.Float64s(), other.Float64s(), reuse.Float64s(), reuse.mask)
					} else {
						err = incrVecMulF64(t.Float64s(), other.Float64s(), reuse.Float64s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += t.GetF64(i) * other.GetF64(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += t.GetF64(i) * other.GetF64(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetF64(i) * other.GetF64(j)
						incrI += iterStep
					}
				}
			}
		case reflect.Complex64:
			data := reuse.Complex64s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += t.GetC64(i) * other.GetC64(j)
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetC64(i) * other.GetC64(j)
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetC64(i) * other.GetC64(j)
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += t.GetC64(i) * other.GetC64(j)
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecMulC64Masked(t.Complex64s(), other.Complex64s(), reuse.Complex64s(), reuse.mask)
					} else {
						err = incrVecMulC64(t.Complex64s(), other.Complex64s(), reuse.Complex64s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += t.GetC64(i) * other.GetC64(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += t.GetC64(i) * other.GetC64(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetC64(i) * other.GetC64(j)
						incrI += iterStep
					}
				}
			}
		case reflect.Complex128:
			data := reuse.Complex128s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += t.GetC128(i) * other.GetC128(j)
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetC128(i) * other.GetC128(j)
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetC128(i) * other.GetC128(j)
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += t.GetC128(i) * other.GetC128(j)
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecMulC128Masked(t.Complex128s(), other.Complex128s(), reuse.Complex128s(), reuse.mask)
					} else {
						err = incrVecMulC128(t.Complex128s(), other.Complex128s(), reuse.Complex128s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += t.GetC128(i) * other.GetC128(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += t.GetC128(i) * other.GetC128(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetC128(i) * other.GetC128(j)
						incrI += iterStep
					}
				}
			}
		}
	case toReuse:
		if t.IsMaterializable() {
			copyDenseIter(reuse, t, nil, it.FlatIterator)
		} else {
			copyDense(reuse, t) // technically copyDenseIter would have done the same but it's much slower
		}
		err = reuse.mul(other, nil, ot)
		retVal = reuse
	case safe:
		if t.IsMaterializable() {
			retVal = t.Materialize().(*Dense)
		} else {
			retVal = t.Clone().(*Dense)
		}
		err = retVal.mul(other, nil, ot)
	case !safe:
		err = t.mul(other, it, ot)
		retVal = t
	}
	if it != nil {
		it.mask = nil
	}
	if ot != nil {
		ot.mask = nil
	}
	return
}
func (t *Dense) mul(other *Dense, itt, ott Iterator) (err error) {
	var it, ot *FlatMaskedIterator
	if itt != nil {
		it = new(FlatMaskedIterator)
		switch iter := itt.(type) {
		case *FlatIterator:
			it.FlatIterator = iter
		case *FlatMaskedIterator:
			it = iter
		case *MultIterator:
			it.FlatIterator = iter.fit0
			it.mask = iter.mask
		}
	}
	if ott != nil {
		ot = new(FlatMaskedIterator)
		switch iter := ott.(type) {
		case *FlatIterator:
			ot.FlatIterator = iter
		case *FlatMaskedIterator:
			ot = iter
		case *MultIterator:
			ot.FlatIterator = iter.fit0
			ot.mask = iter.mask
		}
	}

	t.MaskFromDense(t, other)

	if it != nil {
		it.mask = t.mask
	}
	if ot != nil {
		ot.mask = t.mask
	}

	switch t.t.Kind() {
	case reflect.Int:
		tdata := t.Ints()
		odata := other.Ints()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = tdata[i] * odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = tdata[i] * odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = tdata[i] * odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecMulIMasked(tdata, odata, t.mask)
			} else {
				vecMulI(tdata, odata)
			}
		}
	case reflect.Int8:
		tdata := t.Int8s()
		odata := other.Int8s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = tdata[i] * odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = tdata[i] * odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = tdata[i] * odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecMulI8Masked(tdata, odata, t.mask)
			} else {
				vecMulI8(tdata, odata)
			}
		}
	case reflect.Int16:
		tdata := t.Int16s()
		odata := other.Int16s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = tdata[i] * odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = tdata[i] * odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = tdata[i] * odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecMulI16Masked(tdata, odata, t.mask)
			} else {
				vecMulI16(tdata, odata)
			}
		}
	case reflect.Int32:
		tdata := t.Int32s()
		odata := other.Int32s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = tdata[i] * odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = tdata[i] * odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = tdata[i] * odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecMulI32Masked(tdata, odata, t.mask)
			} else {
				vecMulI32(tdata, odata)
			}
		}
	case reflect.Int64:
		tdata := t.Int64s()
		odata := other.Int64s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = tdata[i] * odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = tdata[i] * odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = tdata[i] * odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecMulI64Masked(tdata, odata, t.mask)
			} else {
				vecMulI64(tdata, odata)
			}
		}
	case reflect.Uint:
		tdata := t.Uints()
		odata := other.Uints()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = tdata[i] * odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = tdata[i] * odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = tdata[i] * odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecMulUMasked(tdata, odata, t.mask)
			} else {
				vecMulU(tdata, odata)
			}
		}
	case reflect.Uint8:
		tdata := t.Uint8s()
		odata := other.Uint8s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = tdata[i] * odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = tdata[i] * odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = tdata[i] * odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecMulU8Masked(tdata, odata, t.mask)
			} else {
				vecMulU8(tdata, odata)
			}
		}
	case reflect.Uint16:
		tdata := t.Uint16s()
		odata := other.Uint16s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = tdata[i] * odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = tdata[i] * odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = tdata[i] * odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecMulU16Masked(tdata, odata, t.mask)
			} else {
				vecMulU16(tdata, odata)
			}
		}
	case reflect.Uint32:
		tdata := t.Uint32s()
		odata := other.Uint32s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = tdata[i] * odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = tdata[i] * odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = tdata[i] * odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecMulU32Masked(tdata, odata, t.mask)
			} else {
				vecMulU32(tdata, odata)
			}
		}
	case reflect.Uint64:
		tdata := t.Uint64s()
		odata := other.Uint64s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = tdata[i] * odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = tdata[i] * odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = tdata[i] * odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecMulU64Masked(tdata, odata, t.mask)
			} else {
				vecMulU64(tdata, odata)
			}
		}
	case reflect.Float32:
		tdata := t.Float32s()
		odata := other.Float32s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = tdata[i] * odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = tdata[i] * odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = tdata[i] * odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecMulF32Masked(tdata, odata, t.mask)
			} else {
				vecMulF32(tdata, odata)
			}
		}
	case reflect.Float64:
		tdata := t.Float64s()
		odata := other.Float64s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = tdata[i] * odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = tdata[i] * odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = tdata[i] * odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecMulF64Masked(tdata, odata, t.mask)
			} else {
				vecMulF64(tdata, odata)
			}
		}
	case reflect.Complex64:
		tdata := t.Complex64s()
		odata := other.Complex64s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = tdata[i] * odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = tdata[i] * odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = tdata[i] * odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecMulC64Masked(tdata, odata, t.mask)
			} else {
				vecMulC64(tdata, odata)
			}
		}
	case reflect.Complex128:
		tdata := t.Complex128s()
		odata := other.Complex128s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = tdata[i] * odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = tdata[i] * odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = tdata[i] * odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecMulC128Masked(tdata, odata, t.mask)
			} else {
				vecMulC128(tdata, odata)
			}
		}
	default:
		// TODO: Handle Number interface
	}

	return
}

/* Div */

// Div performs the operation on another *Dense. It takes a list of FuncOpts.
func (t *Dense) Div(other *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	if t.e != nil {
		if div, ok := t.e.(Diver); ok {
			// if safe, then make a copy
			var ret Tensor
			if ret, err = div.Div(t, other, opts...); err != nil {
				goto attemptGo
			}
			retVal = ret.(*Dense)
			return
		}
	}

attemptGo:
	reuse, safe, toReuse, incr, err := prepBinaryDense(t, other, opts...)
	if err != nil {
		return nil, err
	}

	// check if the tensors are accessible
	if !t.isNativeAccessible() {
		err = errors.Errorf(inaccessibleData, t)
		return
	}

	if !other.isNativeAccessible() {
		err = errors.Errorf(inaccessibleData, reuse)
		return
	}

	if reuse != nil && !reuse.isNativeAccessible() {
		err = errors.Errorf(inaccessibleData, reuse)
		return
	}

	var errs errorIndices
	var it, ot *FlatMaskedIterator
	if t.IsMaterializable() {
		it = NewFlatMaskedIterator(t.AP, t.mask)
	}
	if other.IsMaterializable() {
		ot = NewFlatMaskedIterator(other.AP, other.mask)
	}
	switch {
	case incr:
		// when incr returned, the reuse is the *Dense to be incremented
		retVal = reuse
		retVal.MaskFromDense(t, other)
		if it != nil {
			it.mask = retVal.mask
		}
		if ot != nil {
			ot.mask = retVal.mask
		}
		isMasked := retVal.IsMasked()
		switch reuse.t.Kind() {
		case reflect.Int:
			data := reuse.Ints()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						if other.GetI(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetI(i) / other.GetI(j)
						i += iterStep
						j += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if other.GetI(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetI(i) / other.GetI(j)
						j += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if other.GetI(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetI(i) / other.GetI(j)
						i += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						if other.GetI(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetI(i) / other.GetI(j)
					}
					if err != nil {
						return
					}
					err = errs
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecDivIMasked(t.Ints(), other.Ints(), reuse.Ints(), reuse.mask)
					} else {
						err = incrVecDivI(t.Ints(), other.Ints(), reuse.Ints())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						if other.GetI(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetI(i) / other.GetI(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						if other.GetI(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetI(i) / other.GetI(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if other.GetI(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetI(i) / other.GetI(j)
						incrI += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				}
			}
		case reflect.Int8:
			data := reuse.Int8s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						if other.GetI8(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetI8(i) / other.GetI8(j)
						i += iterStep
						j += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if other.GetI8(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetI8(i) / other.GetI8(j)
						j += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if other.GetI8(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetI8(i) / other.GetI8(j)
						i += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						if other.GetI8(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetI8(i) / other.GetI8(j)
					}
					if err != nil {
						return
					}
					err = errs
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecDivI8Masked(t.Int8s(), other.Int8s(), reuse.Int8s(), reuse.mask)
					} else {
						err = incrVecDivI8(t.Int8s(), other.Int8s(), reuse.Int8s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						if other.GetI8(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetI8(i) / other.GetI8(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						if other.GetI8(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetI8(i) / other.GetI8(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if other.GetI8(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetI8(i) / other.GetI8(j)
						incrI += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				}
			}
		case reflect.Int16:
			data := reuse.Int16s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						if other.GetI16(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetI16(i) / other.GetI16(j)
						i += iterStep
						j += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if other.GetI16(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetI16(i) / other.GetI16(j)
						j += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if other.GetI16(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetI16(i) / other.GetI16(j)
						i += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						if other.GetI16(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetI16(i) / other.GetI16(j)
					}
					if err != nil {
						return
					}
					err = errs
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecDivI16Masked(t.Int16s(), other.Int16s(), reuse.Int16s(), reuse.mask)
					} else {
						err = incrVecDivI16(t.Int16s(), other.Int16s(), reuse.Int16s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						if other.GetI16(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetI16(i) / other.GetI16(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						if other.GetI16(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetI16(i) / other.GetI16(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if other.GetI16(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetI16(i) / other.GetI16(j)
						incrI += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				}
			}
		case reflect.Int32:
			data := reuse.Int32s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						if other.GetI32(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetI32(i) / other.GetI32(j)
						i += iterStep
						j += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if other.GetI32(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetI32(i) / other.GetI32(j)
						j += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if other.GetI32(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetI32(i) / other.GetI32(j)
						i += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						if other.GetI32(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetI32(i) / other.GetI32(j)
					}
					if err != nil {
						return
					}
					err = errs
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecDivI32Masked(t.Int32s(), other.Int32s(), reuse.Int32s(), reuse.mask)
					} else {
						err = incrVecDivI32(t.Int32s(), other.Int32s(), reuse.Int32s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						if other.GetI32(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetI32(i) / other.GetI32(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						if other.GetI32(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetI32(i) / other.GetI32(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if other.GetI32(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetI32(i) / other.GetI32(j)
						incrI += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				}
			}
		case reflect.Int64:
			data := reuse.Int64s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						if other.GetI64(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetI64(i) / other.GetI64(j)
						i += iterStep
						j += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if other.GetI64(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetI64(i) / other.GetI64(j)
						j += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if other.GetI64(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetI64(i) / other.GetI64(j)
						i += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						if other.GetI64(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetI64(i) / other.GetI64(j)
					}
					if err != nil {
						return
					}
					err = errs
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecDivI64Masked(t.Int64s(), other.Int64s(), reuse.Int64s(), reuse.mask)
					} else {
						err = incrVecDivI64(t.Int64s(), other.Int64s(), reuse.Int64s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						if other.GetI64(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetI64(i) / other.GetI64(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						if other.GetI64(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetI64(i) / other.GetI64(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if other.GetI64(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetI64(i) / other.GetI64(j)
						incrI += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				}
			}
		case reflect.Uint:
			data := reuse.Uints()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						if other.GetU(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetU(i) / other.GetU(j)
						i += iterStep
						j += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if other.GetU(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetU(i) / other.GetU(j)
						j += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if other.GetU(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetU(i) / other.GetU(j)
						i += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						if other.GetU(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetU(i) / other.GetU(j)
					}
					if err != nil {
						return
					}
					err = errs
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecDivUMasked(t.Uints(), other.Uints(), reuse.Uints(), reuse.mask)
					} else {
						err = incrVecDivU(t.Uints(), other.Uints(), reuse.Uints())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						if other.GetU(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetU(i) / other.GetU(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						if other.GetU(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetU(i) / other.GetU(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if other.GetU(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetU(i) / other.GetU(j)
						incrI += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				}
			}
		case reflect.Uint8:
			data := reuse.Uint8s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						if other.GetU8(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetU8(i) / other.GetU8(j)
						i += iterStep
						j += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if other.GetU8(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetU8(i) / other.GetU8(j)
						j += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if other.GetU8(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetU8(i) / other.GetU8(j)
						i += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						if other.GetU8(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetU8(i) / other.GetU8(j)
					}
					if err != nil {
						return
					}
					err = errs
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecDivU8Masked(t.Uint8s(), other.Uint8s(), reuse.Uint8s(), reuse.mask)
					} else {
						err = incrVecDivU8(t.Uint8s(), other.Uint8s(), reuse.Uint8s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						if other.GetU8(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetU8(i) / other.GetU8(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						if other.GetU8(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetU8(i) / other.GetU8(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if other.GetU8(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetU8(i) / other.GetU8(j)
						incrI += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				}
			}
		case reflect.Uint16:
			data := reuse.Uint16s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						if other.GetU16(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetU16(i) / other.GetU16(j)
						i += iterStep
						j += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if other.GetU16(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetU16(i) / other.GetU16(j)
						j += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if other.GetU16(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetU16(i) / other.GetU16(j)
						i += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						if other.GetU16(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetU16(i) / other.GetU16(j)
					}
					if err != nil {
						return
					}
					err = errs
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecDivU16Masked(t.Uint16s(), other.Uint16s(), reuse.Uint16s(), reuse.mask)
					} else {
						err = incrVecDivU16(t.Uint16s(), other.Uint16s(), reuse.Uint16s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						if other.GetU16(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetU16(i) / other.GetU16(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						if other.GetU16(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetU16(i) / other.GetU16(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if other.GetU16(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetU16(i) / other.GetU16(j)
						incrI += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				}
			}
		case reflect.Uint32:
			data := reuse.Uint32s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						if other.GetU32(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetU32(i) / other.GetU32(j)
						i += iterStep
						j += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if other.GetU32(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetU32(i) / other.GetU32(j)
						j += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if other.GetU32(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetU32(i) / other.GetU32(j)
						i += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						if other.GetU32(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetU32(i) / other.GetU32(j)
					}
					if err != nil {
						return
					}
					err = errs
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecDivU32Masked(t.Uint32s(), other.Uint32s(), reuse.Uint32s(), reuse.mask)
					} else {
						err = incrVecDivU32(t.Uint32s(), other.Uint32s(), reuse.Uint32s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						if other.GetU32(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetU32(i) / other.GetU32(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						if other.GetU32(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetU32(i) / other.GetU32(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if other.GetU32(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetU32(i) / other.GetU32(j)
						incrI += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				}
			}
		case reflect.Uint64:
			data := reuse.Uint64s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						if other.GetU64(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetU64(i) / other.GetU64(j)
						i += iterStep
						j += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if other.GetU64(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetU64(i) / other.GetU64(j)
						j += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if other.GetU64(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetU64(i) / other.GetU64(j)
						i += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						if other.GetU64(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetU64(i) / other.GetU64(j)
					}
					if err != nil {
						return
					}
					err = errs
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecDivU64Masked(t.Uint64s(), other.Uint64s(), reuse.Uint64s(), reuse.mask)
					} else {
						err = incrVecDivU64(t.Uint64s(), other.Uint64s(), reuse.Uint64s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						if other.GetU64(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetU64(i) / other.GetU64(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						if other.GetU64(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetU64(i) / other.GetU64(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if other.GetU64(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetU64(i) / other.GetU64(j)
						incrI += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				}
			}
		case reflect.Float32:
			data := reuse.Float32s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += t.GetF32(i) / other.GetF32(j)
						i += iterStep
						j += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetF32(i) / other.GetF32(j)
						j += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetF32(i) / other.GetF32(j)
						i += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += t.GetF32(i) / other.GetF32(j)
					}
					if err != nil {
						return
					}
					err = errs
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecDivF32Masked(t.Float32s(), other.Float32s(), reuse.Float32s(), reuse.mask)
					} else {
						err = incrVecDivF32(t.Float32s(), other.Float32s(), reuse.Float32s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += t.GetF32(i) / other.GetF32(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += t.GetF32(i) / other.GetF32(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetF32(i) / other.GetF32(j)
						incrI += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				}
			}
		case reflect.Float64:
			data := reuse.Float64s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += t.GetF64(i) / other.GetF64(j)
						i += iterStep
						j += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetF64(i) / other.GetF64(j)
						j += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetF64(i) / other.GetF64(j)
						i += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += t.GetF64(i) / other.GetF64(j)
					}
					if err != nil {
						return
					}
					err = errs
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecDivF64Masked(t.Float64s(), other.Float64s(), reuse.Float64s(), reuse.mask)
					} else {
						err = incrVecDivF64(t.Float64s(), other.Float64s(), reuse.Float64s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += t.GetF64(i) / other.GetF64(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += t.GetF64(i) / other.GetF64(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += t.GetF64(i) / other.GetF64(j)
						incrI += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				}
			}
		case reflect.Complex64:
			data := reuse.Complex64s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						if other.GetC64(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetC64(i) / other.GetC64(j)
						i += iterStep
						j += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if other.GetC64(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetC64(i) / other.GetC64(j)
						j += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if other.GetC64(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetC64(i) / other.GetC64(j)
						i += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						if other.GetC64(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetC64(i) / other.GetC64(j)
					}
					if err != nil {
						return
					}
					err = errs
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecDivC64Masked(t.Complex64s(), other.Complex64s(), reuse.Complex64s(), reuse.mask)
					} else {
						err = incrVecDivC64(t.Complex64s(), other.Complex64s(), reuse.Complex64s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						if other.GetC64(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetC64(i) / other.GetC64(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						if other.GetC64(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetC64(i) / other.GetC64(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if other.GetC64(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetC64(i) / other.GetC64(j)
						incrI += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				}
			}
		case reflect.Complex128:
			data := reuse.Complex128s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						if other.GetC128(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetC128(i) / other.GetC128(j)
						i += iterStep
						j += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if other.GetC128(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetC128(i) / other.GetC128(j)
						j += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if other.GetC128(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetC128(i) / other.GetC128(j)
						i += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						if other.GetC128(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetC128(i) / other.GetC128(j)
					}
					if err != nil {
						return
					}
					err = errs
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecDivC128Masked(t.Complex128s(), other.Complex128s(), reuse.Complex128s(), reuse.mask)
					} else {
						err = incrVecDivC128(t.Complex128s(), other.Complex128s(), reuse.Complex128s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						if other.GetC128(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetC128(i) / other.GetC128(j)
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						if other.GetC128(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetC128(i) / other.GetC128(j)
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if other.GetC128(j) == 0 {
							errs = append(errs, j)
							continue
						}
						data[incrI] += t.GetC128(i) / other.GetC128(j)
						incrI += iterStep
					}
					if err != nil {
						return
					}
					err = errs
				}
			}
		}
		if errs != nil {
			err = errs
		}
	case toReuse:
		if t.IsMaterializable() {
			copyDenseIter(reuse, t, nil, it.FlatIterator)
		} else {
			copyDense(reuse, t) // technically copyDenseIter would have done the same but it's much slower
		}
		err = reuse.div(other, nil, ot)
		retVal = reuse
	case safe:
		if t.IsMaterializable() {
			retVal = t.Materialize().(*Dense)
		} else {
			retVal = t.Clone().(*Dense)
		}
		err = retVal.div(other, nil, ot)
	case !safe:
		err = t.div(other, it, ot)
		retVal = t
	}
	if it != nil {
		it.mask = nil
	}
	if ot != nil {
		ot.mask = nil
	}
	return
}
func (t *Dense) div(other *Dense, itt, ott Iterator) (err error) {
	var it, ot *FlatMaskedIterator
	if itt != nil {
		it = new(FlatMaskedIterator)
		switch iter := itt.(type) {
		case *FlatIterator:
			it.FlatIterator = iter
		case *FlatMaskedIterator:
			it = iter
		case *MultIterator:
			it.FlatIterator = iter.fit0
			it.mask = iter.mask
		}
	}
	if ott != nil {
		ot = new(FlatMaskedIterator)
		switch iter := ott.(type) {
		case *FlatIterator:
			ot.FlatIterator = iter
		case *FlatMaskedIterator:
			ot = iter
		case *MultIterator:
			ot.FlatIterator = iter.fit0
			ot.mask = iter.mask
		}
	}

	t.MaskFromDense(t, other)

	if it != nil {
		it.mask = t.mask
	}
	if ot != nil {
		ot.mask = t.mask
	}

	var errs errorIndices
	switch t.t.Kind() {
	case reflect.Int:
		tdata := t.Ints()
		odata := other.Ints()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecDivIMasked(tdata, odata, t.mask)
			} else {
				vecDivI(tdata, odata)
			}
		}
	case reflect.Int8:
		tdata := t.Int8s()
		odata := other.Int8s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecDivI8Masked(tdata, odata, t.mask)
			} else {
				vecDivI8(tdata, odata)
			}
		}
	case reflect.Int16:
		tdata := t.Int16s()
		odata := other.Int16s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecDivI16Masked(tdata, odata, t.mask)
			} else {
				vecDivI16(tdata, odata)
			}
		}
	case reflect.Int32:
		tdata := t.Int32s()
		odata := other.Int32s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecDivI32Masked(tdata, odata, t.mask)
			} else {
				vecDivI32(tdata, odata)
			}
		}
	case reflect.Int64:
		tdata := t.Int64s()
		odata := other.Int64s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecDivI64Masked(tdata, odata, t.mask)
			} else {
				vecDivI64(tdata, odata)
			}
		}
	case reflect.Uint:
		tdata := t.Uints()
		odata := other.Uints()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecDivUMasked(tdata, odata, t.mask)
			} else {
				vecDivU(tdata, odata)
			}
		}
	case reflect.Uint8:
		tdata := t.Uint8s()
		odata := other.Uint8s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecDivU8Masked(tdata, odata, t.mask)
			} else {
				vecDivU8(tdata, odata)
			}
		}
	case reflect.Uint16:
		tdata := t.Uint16s()
		odata := other.Uint16s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecDivU16Masked(tdata, odata, t.mask)
			} else {
				vecDivU16(tdata, odata)
			}
		}
	case reflect.Uint32:
		tdata := t.Uint32s()
		odata := other.Uint32s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecDivU32Masked(tdata, odata, t.mask)
			} else {
				vecDivU32(tdata, odata)
			}
		}
	case reflect.Uint64:
		tdata := t.Uint64s()
		odata := other.Uint64s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecDivU64Masked(tdata, odata, t.mask)
			} else {
				vecDivU64(tdata, odata)
			}
		}
	case reflect.Float32:
		tdata := t.Float32s()
		odata := other.Float32s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = tdata[i] / odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = tdata[i] / odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = tdata[i] / odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecDivF32Masked(tdata, odata, t.mask)
			} else {
				vecDivF32(tdata, odata)
			}
		}
	case reflect.Float64:
		tdata := t.Float64s()
		odata := other.Float64s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = tdata[i] / odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = tdata[i] / odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = tdata[i] / odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecDivF64Masked(tdata, odata, t.mask)
			} else {
				vecDivF64(tdata, odata)
			}
		}
	case reflect.Complex64:
		tdata := t.Complex64s()
		odata := other.Complex64s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecDivC64Masked(tdata, odata, t.mask)
			} else {
				vecDivC64(tdata, odata)
			}
		}
	case reflect.Complex128:
		tdata := t.Complex128s()
		odata := other.Complex128s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecDivC128Masked(tdata, odata, t.mask)
			} else {
				vecDivC128(tdata, odata)
			}
		}
	default:
		// TODO: Handle Number interface
	}

	if err != nil {
		return
	}

	if errs != nil {
		err = errs
	}
	return
}

/* Pow */

// Pow performs the operation on another *Dense. It takes a list of FuncOpts.
func (t *Dense) Pow(other *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	if t.e != nil {
		if pow, ok := t.e.(Power); ok {
			// if safe, then make a copy
			var ret Tensor
			if ret, err = pow.Pow(t, other, opts...); err != nil {
				goto attemptGo
			}
			retVal = ret.(*Dense)
			return
		}
	}

attemptGo:
	reuse, safe, toReuse, incr, err := prepBinaryDense(t, other, opts...)
	if err != nil {
		return nil, err
	}

	// check if the tensors are accessible
	if !t.isNativeAccessible() {
		err = errors.Errorf(inaccessibleData, t)
		return
	}

	if !other.isNativeAccessible() {
		err = errors.Errorf(inaccessibleData, reuse)
		return
	}

	if reuse != nil && !reuse.isNativeAccessible() {
		err = errors.Errorf(inaccessibleData, reuse)
		return
	}

	var it, ot *FlatMaskedIterator
	if t.IsMaterializable() {
		it = NewFlatMaskedIterator(t.AP, t.mask)
	}
	if other.IsMaterializable() {
		ot = NewFlatMaskedIterator(other.AP, other.mask)
	}
	switch {
	case incr:
		// when incr returned, the reuse is the *Dense to be incremented
		retVal = reuse
		retVal.MaskFromDense(t, other)
		if it != nil {
			it.mask = retVal.mask
		}
		if ot != nil {
			ot.mask = retVal.mask
		}
		isMasked := retVal.IsMasked()
		switch reuse.t.Kind() {
		case reflect.Int:
			data := reuse.Ints()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += int(math.Pow(float64(t.GetI(i)), float64(other.GetI(j))))
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += int(math.Pow(float64(t.GetI(i)), float64(other.GetI(j))))
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += int(math.Pow(float64(t.GetI(i)), float64(other.GetI(j))))
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += int(math.Pow(float64(t.GetI(i)), float64(other.GetI(j))))
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecPowIMasked(t.Ints(), other.Ints(), reuse.Ints(), reuse.mask)
					} else {
						err = incrVecPowI(t.Ints(), other.Ints(), reuse.Ints())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += int(math.Pow(float64(t.GetI(i)), float64(other.GetI(j))))
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += int(math.Pow(float64(t.GetI(i)), float64(other.GetI(j))))
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += int(math.Pow(float64(t.GetI(i)), float64(other.GetI(j))))
						incrI += iterStep
					}
				}
			}
		case reflect.Int8:
			data := reuse.Int8s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += int8(math.Pow(float64(t.GetI8(i)), float64(other.GetI8(j))))
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += int8(math.Pow(float64(t.GetI8(i)), float64(other.GetI8(j))))
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += int8(math.Pow(float64(t.GetI8(i)), float64(other.GetI8(j))))
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += int8(math.Pow(float64(t.GetI8(i)), float64(other.GetI8(j))))
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecPowI8Masked(t.Int8s(), other.Int8s(), reuse.Int8s(), reuse.mask)
					} else {
						err = incrVecPowI8(t.Int8s(), other.Int8s(), reuse.Int8s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += int8(math.Pow(float64(t.GetI8(i)), float64(other.GetI8(j))))
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += int8(math.Pow(float64(t.GetI8(i)), float64(other.GetI8(j))))
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += int8(math.Pow(float64(t.GetI8(i)), float64(other.GetI8(j))))
						incrI += iterStep
					}
				}
			}
		case reflect.Int16:
			data := reuse.Int16s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += int16(math.Pow(float64(t.GetI16(i)), float64(other.GetI16(j))))
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += int16(math.Pow(float64(t.GetI16(i)), float64(other.GetI16(j))))
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += int16(math.Pow(float64(t.GetI16(i)), float64(other.GetI16(j))))
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += int16(math.Pow(float64(t.GetI16(i)), float64(other.GetI16(j))))
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecPowI16Masked(t.Int16s(), other.Int16s(), reuse.Int16s(), reuse.mask)
					} else {
						err = incrVecPowI16(t.Int16s(), other.Int16s(), reuse.Int16s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += int16(math.Pow(float64(t.GetI16(i)), float64(other.GetI16(j))))
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += int16(math.Pow(float64(t.GetI16(i)), float64(other.GetI16(j))))
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += int16(math.Pow(float64(t.GetI16(i)), float64(other.GetI16(j))))
						incrI += iterStep
					}
				}
			}
		case reflect.Int32:
			data := reuse.Int32s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += int32(math.Pow(float64(t.GetI32(i)), float64(other.GetI32(j))))
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += int32(math.Pow(float64(t.GetI32(i)), float64(other.GetI32(j))))
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += int32(math.Pow(float64(t.GetI32(i)), float64(other.GetI32(j))))
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += int32(math.Pow(float64(t.GetI32(i)), float64(other.GetI32(j))))
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecPowI32Masked(t.Int32s(), other.Int32s(), reuse.Int32s(), reuse.mask)
					} else {
						err = incrVecPowI32(t.Int32s(), other.Int32s(), reuse.Int32s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += int32(math.Pow(float64(t.GetI32(i)), float64(other.GetI32(j))))
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += int32(math.Pow(float64(t.GetI32(i)), float64(other.GetI32(j))))
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += int32(math.Pow(float64(t.GetI32(i)), float64(other.GetI32(j))))
						incrI += iterStep
					}
				}
			}
		case reflect.Int64:
			data := reuse.Int64s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += int64(math.Pow(float64(t.GetI64(i)), float64(other.GetI64(j))))
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += int64(math.Pow(float64(t.GetI64(i)), float64(other.GetI64(j))))
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += int64(math.Pow(float64(t.GetI64(i)), float64(other.GetI64(j))))
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += int64(math.Pow(float64(t.GetI64(i)), float64(other.GetI64(j))))
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecPowI64Masked(t.Int64s(), other.Int64s(), reuse.Int64s(), reuse.mask)
					} else {
						err = incrVecPowI64(t.Int64s(), other.Int64s(), reuse.Int64s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += int64(math.Pow(float64(t.GetI64(i)), float64(other.GetI64(j))))
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += int64(math.Pow(float64(t.GetI64(i)), float64(other.GetI64(j))))
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += int64(math.Pow(float64(t.GetI64(i)), float64(other.GetI64(j))))
						incrI += iterStep
					}
				}
			}
		case reflect.Uint:
			data := reuse.Uints()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += uint(math.Pow(float64(t.GetU(i)), float64(other.GetU(j))))
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += uint(math.Pow(float64(t.GetU(i)), float64(other.GetU(j))))
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += uint(math.Pow(float64(t.GetU(i)), float64(other.GetU(j))))
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += uint(math.Pow(float64(t.GetU(i)), float64(other.GetU(j))))
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecPowUMasked(t.Uints(), other.Uints(), reuse.Uints(), reuse.mask)
					} else {
						err = incrVecPowU(t.Uints(), other.Uints(), reuse.Uints())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += uint(math.Pow(float64(t.GetU(i)), float64(other.GetU(j))))
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += uint(math.Pow(float64(t.GetU(i)), float64(other.GetU(j))))
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += uint(math.Pow(float64(t.GetU(i)), float64(other.GetU(j))))
						incrI += iterStep
					}
				}
			}
		case reflect.Uint8:
			data := reuse.Uint8s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += uint8(math.Pow(float64(t.GetU8(i)), float64(other.GetU8(j))))
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += uint8(math.Pow(float64(t.GetU8(i)), float64(other.GetU8(j))))
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += uint8(math.Pow(float64(t.GetU8(i)), float64(other.GetU8(j))))
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += uint8(math.Pow(float64(t.GetU8(i)), float64(other.GetU8(j))))
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecPowU8Masked(t.Uint8s(), other.Uint8s(), reuse.Uint8s(), reuse.mask)
					} else {
						err = incrVecPowU8(t.Uint8s(), other.Uint8s(), reuse.Uint8s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += uint8(math.Pow(float64(t.GetU8(i)), float64(other.GetU8(j))))
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += uint8(math.Pow(float64(t.GetU8(i)), float64(other.GetU8(j))))
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += uint8(math.Pow(float64(t.GetU8(i)), float64(other.GetU8(j))))
						incrI += iterStep
					}
				}
			}
		case reflect.Uint16:
			data := reuse.Uint16s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += uint16(math.Pow(float64(t.GetU16(i)), float64(other.GetU16(j))))
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += uint16(math.Pow(float64(t.GetU16(i)), float64(other.GetU16(j))))
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += uint16(math.Pow(float64(t.GetU16(i)), float64(other.GetU16(j))))
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += uint16(math.Pow(float64(t.GetU16(i)), float64(other.GetU16(j))))
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecPowU16Masked(t.Uint16s(), other.Uint16s(), reuse.Uint16s(), reuse.mask)
					} else {
						err = incrVecPowU16(t.Uint16s(), other.Uint16s(), reuse.Uint16s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += uint16(math.Pow(float64(t.GetU16(i)), float64(other.GetU16(j))))
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += uint16(math.Pow(float64(t.GetU16(i)), float64(other.GetU16(j))))
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += uint16(math.Pow(float64(t.GetU16(i)), float64(other.GetU16(j))))
						incrI += iterStep
					}
				}
			}
		case reflect.Uint32:
			data := reuse.Uint32s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += uint32(math.Pow(float64(t.GetU32(i)), float64(other.GetU32(j))))
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += uint32(math.Pow(float64(t.GetU32(i)), float64(other.GetU32(j))))
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += uint32(math.Pow(float64(t.GetU32(i)), float64(other.GetU32(j))))
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += uint32(math.Pow(float64(t.GetU32(i)), float64(other.GetU32(j))))
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecPowU32Masked(t.Uint32s(), other.Uint32s(), reuse.Uint32s(), reuse.mask)
					} else {
						err = incrVecPowU32(t.Uint32s(), other.Uint32s(), reuse.Uint32s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += uint32(math.Pow(float64(t.GetU32(i)), float64(other.GetU32(j))))
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += uint32(math.Pow(float64(t.GetU32(i)), float64(other.GetU32(j))))
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += uint32(math.Pow(float64(t.GetU32(i)), float64(other.GetU32(j))))
						incrI += iterStep
					}
				}
			}
		case reflect.Uint64:
			data := reuse.Uint64s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += uint64(math.Pow(float64(t.GetU64(i)), float64(other.GetU64(j))))
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += uint64(math.Pow(float64(t.GetU64(i)), float64(other.GetU64(j))))
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += uint64(math.Pow(float64(t.GetU64(i)), float64(other.GetU64(j))))
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += uint64(math.Pow(float64(t.GetU64(i)), float64(other.GetU64(j))))
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecPowU64Masked(t.Uint64s(), other.Uint64s(), reuse.Uint64s(), reuse.mask)
					} else {
						err = incrVecPowU64(t.Uint64s(), other.Uint64s(), reuse.Uint64s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += uint64(math.Pow(float64(t.GetU64(i)), float64(other.GetU64(j))))
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += uint64(math.Pow(float64(t.GetU64(i)), float64(other.GetU64(j))))
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += uint64(math.Pow(float64(t.GetU64(i)), float64(other.GetU64(j))))
						incrI += iterStep
					}
				}
			}
		case reflect.Float32:
			data := reuse.Float32s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += math32.Pow(t.GetF32(i), other.GetF32(j))
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += math32.Pow(t.GetF32(i), other.GetF32(j))
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += math32.Pow(t.GetF32(i), other.GetF32(j))
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += math32.Pow(t.GetF32(i), other.GetF32(j))
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecPowF32Masked(t.Float32s(), other.Float32s(), reuse.Float32s(), reuse.mask)
					} else {
						err = incrVecPowF32(t.Float32s(), other.Float32s(), reuse.Float32s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += math32.Pow(t.GetF32(i), other.GetF32(j))
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += math32.Pow(t.GetF32(i), other.GetF32(j))
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += math32.Pow(t.GetF32(i), other.GetF32(j))
						incrI += iterStep
					}
				}
			}
		case reflect.Float64:
			data := reuse.Float64s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += math.Pow(t.GetF64(i), other.GetF64(j))
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += math.Pow(t.GetF64(i), other.GetF64(j))
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += math.Pow(t.GetF64(i), other.GetF64(j))
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += math.Pow(t.GetF64(i), other.GetF64(j))
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecPowF64Masked(t.Float64s(), other.Float64s(), reuse.Float64s(), reuse.mask)
					} else {
						err = incrVecPowF64(t.Float64s(), other.Float64s(), reuse.Float64s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += math.Pow(t.GetF64(i), other.GetF64(j))
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += math.Pow(t.GetF64(i), other.GetF64(j))
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += math.Pow(t.GetF64(i), other.GetF64(j))
						incrI += iterStep
					}
				}
			}
		case reflect.Complex64:
			data := reuse.Complex64s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += complex64(cmplx.Pow(complex128(t.GetC64(i)), complex128(other.GetC64(j))))
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += complex64(cmplx.Pow(complex128(t.GetC64(i)), complex128(other.GetC64(j))))
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += complex64(cmplx.Pow(complex128(t.GetC64(i)), complex128(other.GetC64(j))))
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += complex64(cmplx.Pow(complex128(t.GetC64(i)), complex128(other.GetC64(j))))
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecPowC64Masked(t.Complex64s(), other.Complex64s(), reuse.Complex64s(), reuse.mask)
					} else {
						err = incrVecPowC64(t.Complex64s(), other.Complex64s(), reuse.Complex64s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += complex64(cmplx.Pow(complex128(t.GetC64(i)), complex128(other.GetC64(j))))
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += complex64(cmplx.Pow(complex128(t.GetC64(i)), complex128(other.GetC64(j))))
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += complex64(cmplx.Pow(complex128(t.GetC64(i)), complex128(other.GetC64(j))))
						incrI += iterStep
					}
				}
			}
		case reflect.Complex128:
			data := reuse.Complex128s()
			switch {
			case reuse.IsMaterializable():
				incrIter := FlatMaskedIteratorFromDense(retVal)
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					for incrI, iterStep, err = incrIter.NextValid(); err == nil; incrI, iterStep, err = incrIter.NextValid() {
						data[incrI] += cmplx.Pow(t.GetC128(i), other.GetC128(j))
						i += iterStep
						j += iterStep
					}
				case it != nil && ot == nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += cmplx.Pow(t.GetC128(i), other.GetC128(j))
						j += iterStep
					}
				case it == nil && ot != nil:
					for {
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, iterStep, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += cmplx.Pow(t.GetC128(i), other.GetC128(j))
						i += iterStep
					}
				case it != nil && ot != nil:
					for {
						if i, _, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if incrI, _, err = incrIter.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}

						data[incrI] += cmplx.Pow(t.GetC128(i), other.GetC128(j))
					}
				}
			case !reuse.IsMaterializable():
				var i, j, incrI, iterStep int
				switch {
				case it == nil && ot == nil:
					if isMasked {
						err = incrVecPowC128Masked(t.Complex128s(), other.Complex128s(), reuse.Complex128s(), reuse.mask)
					} else {
						err = incrVecPowC128(t.Complex128s(), other.Complex128s(), reuse.Complex128s())
					}
				case it != nil && ot == nil:
					for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
						data[incrI] += cmplx.Pow(t.GetC128(i), other.GetC128(j))
						j += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it == nil && ot != nil:
					for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
						data[incrI] += cmplx.Pow(t.GetC128(i), other.GetC128(j))
						i += iterStep
						incrI += iterStep
					}
					err = handleNoOp(err)
				case it != nil && ot != nil:
					for {
						if i, iterStep, err = it.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						if j, _, err = ot.NextValid(); err != nil {
							err = handleNoOp(err)
							break
						}
						data[incrI] += cmplx.Pow(t.GetC128(i), other.GetC128(j))
						incrI += iterStep
					}
				}
			}
		}
	case toReuse:
		if t.IsMaterializable() {
			copyDenseIter(reuse, t, nil, it.FlatIterator)
		} else {
			copyDense(reuse, t) // technically copyDenseIter would have done the same but it's much slower
		}
		err = reuse.pow(other, nil, ot)
		retVal = reuse
	case safe:
		if t.IsMaterializable() {
			retVal = t.Materialize().(*Dense)
		} else {
			retVal = t.Clone().(*Dense)
		}
		err = retVal.pow(other, nil, ot)
	case !safe:
		err = t.pow(other, it, ot)
		retVal = t
	}
	if it != nil {
		it.mask = nil
	}
	if ot != nil {
		ot.mask = nil
	}
	return
}
func (t *Dense) pow(other *Dense, itt, ott Iterator) (err error) {
	var it, ot *FlatMaskedIterator
	if itt != nil {
		it = new(FlatMaskedIterator)
		switch iter := itt.(type) {
		case *FlatIterator:
			it.FlatIterator = iter
		case *FlatMaskedIterator:
			it = iter
		case *MultIterator:
			it.FlatIterator = iter.fit0
			it.mask = iter.mask
		}
	}
	if ott != nil {
		ot = new(FlatMaskedIterator)
		switch iter := ott.(type) {
		case *FlatIterator:
			ot.FlatIterator = iter
		case *FlatMaskedIterator:
			ot = iter
		case *MultIterator:
			ot.FlatIterator = iter.fit0
			ot.mask = iter.mask
		}
	}

	t.MaskFromDense(t, other)

	if it != nil {
		it.mask = t.mask
	}
	if ot != nil {
		ot.mask = t.mask
	}

	switch t.t.Kind() {
	case reflect.Int:
		tdata := t.Ints()
		odata := other.Ints()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = int(math.Pow(float64(tdata[i]), float64(odata[j])))
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = int(math.Pow(float64(tdata[i]), float64(odata[j])))
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = int(math.Pow(float64(tdata[i]), float64(odata[j])))
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecPowIMasked(tdata, odata, t.mask)
			} else {
				vecPowI(tdata, odata)
			}
		}
	case reflect.Int8:
		tdata := t.Int8s()
		odata := other.Int8s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = int8(math.Pow(float64(tdata[i]), float64(odata[j])))
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = int8(math.Pow(float64(tdata[i]), float64(odata[j])))
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = int8(math.Pow(float64(tdata[i]), float64(odata[j])))
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecPowI8Masked(tdata, odata, t.mask)
			} else {
				vecPowI8(tdata, odata)
			}
		}
	case reflect.Int16:
		tdata := t.Int16s()
		odata := other.Int16s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = int16(math.Pow(float64(tdata[i]), float64(odata[j])))
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = int16(math.Pow(float64(tdata[i]), float64(odata[j])))
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = int16(math.Pow(float64(tdata[i]), float64(odata[j])))
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecPowI16Masked(tdata, odata, t.mask)
			} else {
				vecPowI16(tdata, odata)
			}
		}
	case reflect.Int32:
		tdata := t.Int32s()
		odata := other.Int32s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = int32(math.Pow(float64(tdata[i]), float64(odata[j])))
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = int32(math.Pow(float64(tdata[i]), float64(odata[j])))
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = int32(math.Pow(float64(tdata[i]), float64(odata[j])))
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecPowI32Masked(tdata, odata, t.mask)
			} else {
				vecPowI32(tdata, odata)
			}
		}
	case reflect.Int64:
		tdata := t.Int64s()
		odata := other.Int64s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = int64(math.Pow(float64(tdata[i]), float64(odata[j])))
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = int64(math.Pow(float64(tdata[i]), float64(odata[j])))
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = int64(math.Pow(float64(tdata[i]), float64(odata[j])))
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecPowI64Masked(tdata, odata, t.mask)
			} else {
				vecPowI64(tdata, odata)
			}
		}
	case reflect.Uint:
		tdata := t.Uints()
		odata := other.Uints()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = uint(math.Pow(float64(tdata[i]), float64(odata[j])))
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = uint(math.Pow(float64(tdata[i]), float64(odata[j])))
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = uint(math.Pow(float64(tdata[i]), float64(odata[j])))
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecPowUMasked(tdata, odata, t.mask)
			} else {
				vecPowU(tdata, odata)
			}
		}
	case reflect.Uint8:
		tdata := t.Uint8s()
		odata := other.Uint8s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = uint8(math.Pow(float64(tdata[i]), float64(odata[j])))
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = uint8(math.Pow(float64(tdata[i]), float64(odata[j])))
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = uint8(math.Pow(float64(tdata[i]), float64(odata[j])))
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecPowU8Masked(tdata, odata, t.mask)
			} else {
				vecPowU8(tdata, odata)
			}
		}
	case reflect.Uint16:
		tdata := t.Uint16s()
		odata := other.Uint16s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = uint16(math.Pow(float64(tdata[i]), float64(odata[j])))
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = uint16(math.Pow(float64(tdata[i]), float64(odata[j])))
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = uint16(math.Pow(float64(tdata[i]), float64(odata[j])))
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecPowU16Masked(tdata, odata, t.mask)
			} else {
				vecPowU16(tdata, odata)
			}
		}
	case reflect.Uint32:
		tdata := t.Uint32s()
		odata := other.Uint32s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = uint32(math.Pow(float64(tdata[i]), float64(odata[j])))
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = uint32(math.Pow(float64(tdata[i]), float64(odata[j])))
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = uint32(math.Pow(float64(tdata[i]), float64(odata[j])))
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecPowU32Masked(tdata, odata, t.mask)
			} else {
				vecPowU32(tdata, odata)
			}
		}
	case reflect.Uint64:
		tdata := t.Uint64s()
		odata := other.Uint64s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = uint64(math.Pow(float64(tdata[i]), float64(odata[j])))
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = uint64(math.Pow(float64(tdata[i]), float64(odata[j])))
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = uint64(math.Pow(float64(tdata[i]), float64(odata[j])))
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecPowU64Masked(tdata, odata, t.mask)
			} else {
				vecPowU64(tdata, odata)
			}
		}
	case reflect.Float32:
		tdata := t.Float32s()
		odata := other.Float32s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = math32.Pow(tdata[i], odata[j])
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = math32.Pow(tdata[i], odata[j])
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = math32.Pow(tdata[i], odata[j])
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecPowF32Masked(tdata, odata, t.mask)
			} else {
				vecPowF32(tdata, odata)
			}
		}
	case reflect.Float64:
		tdata := t.Float64s()
		odata := other.Float64s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = math.Pow(tdata[i], odata[j])
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = math.Pow(tdata[i], odata[j])
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = math.Pow(tdata[i], odata[j])
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecPowF64Masked(tdata, odata, t.mask)
			} else {
				vecPowF64(tdata, odata)
			}
		}
	case reflect.Complex64:
		tdata := t.Complex64s()
		odata := other.Complex64s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = complex64(cmplx.Pow(complex128(tdata[i]), complex128(odata[j])))
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = complex64(cmplx.Pow(complex128(tdata[i]), complex128(odata[j])))
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = complex64(cmplx.Pow(complex128(tdata[i]), complex128(odata[j])))
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecPowC64Masked(tdata, odata, t.mask)
			} else {
				vecPowC64(tdata, odata)
			}
		}
	case reflect.Complex128:
		tdata := t.Complex128s()
		odata := other.Complex128s()
		var i, j, iterStep int
		switch {
		case it != nil && ot != nil:
			for {
				if i, _, err = it.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, _, err = ot.NextValid(); err != nil {
					err = handleNoOp(err)
					break
				}
				tdata[i] = cmplx.Pow(tdata[i], odata[j])
			}
		case it != nil && ot == nil:
			for i, iterStep, err = it.NextValid(); err == nil; i, iterStep, err = it.NextValid() {
				tdata[i] = cmplx.Pow(tdata[i], odata[j])
				j += iterStep
			}
			err = handleNoOp(err)
		case it == nil && ot != nil:
			for j, iterStep, err = ot.NextValid(); err == nil; j, iterStep, err = ot.NextValid() {
				tdata[i] = cmplx.Pow(tdata[i], odata[j])
				i += iterStep
			}
		default:
			if t.IsMasked() {
				vecPowC128Masked(tdata, odata, t.mask)
			} else {
				vecPowC128(tdata, odata)
			}
		}
	default:
		// TODO: Handle Number interface
	}

	return
}

/* Trans */

// Trans performs addition on a *Dense and a scalar value. The scalar value has to be of the same
// type as defined in the *Dense, otherwise an error will be returned.
func (t *Dense) Trans(other interface{}, opts ...FuncOpt) (retVal *Dense, err error) {
	reuse, safe, toReuse, incr, err := prepUnaryDense(t, opts...)
	if err != nil {
		return nil, err
	}
	if t.IsMasked() && (reuse != nil) {
		reuse.MaskFromDense(t)
	}

	switch {
	case incr:
		switch t.t.Kind() {
		case reflect.Int:
			if t.IsMasked() {
				err = incrTransIMasked(t.Ints(), reuse.Ints(), other.(int), t.mask)
			} else {
				err = incrTransI(t.Ints(), reuse.Ints(), other.(int))
			}
			retVal = reuse
		case reflect.Int8:
			if t.IsMasked() {
				err = incrTransI8Masked(t.Int8s(), reuse.Int8s(), other.(int8), t.mask)
			} else {
				err = incrTransI8(t.Int8s(), reuse.Int8s(), other.(int8))
			}
			retVal = reuse
		case reflect.Int16:
			if t.IsMasked() {
				err = incrTransI16Masked(t.Int16s(), reuse.Int16s(), other.(int16), t.mask)
			} else {
				err = incrTransI16(t.Int16s(), reuse.Int16s(), other.(int16))
			}
			retVal = reuse
		case reflect.Int32:
			if t.IsMasked() {
				err = incrTransI32Masked(t.Int32s(), reuse.Int32s(), other.(int32), t.mask)
			} else {
				err = incrTransI32(t.Int32s(), reuse.Int32s(), other.(int32))
			}
			retVal = reuse
		case reflect.Int64:
			if t.IsMasked() {
				err = incrTransI64Masked(t.Int64s(), reuse.Int64s(), other.(int64), t.mask)
			} else {
				err = incrTransI64(t.Int64s(), reuse.Int64s(), other.(int64))
			}
			retVal = reuse
		case reflect.Uint:
			if t.IsMasked() {
				err = incrTransUMasked(t.Uints(), reuse.Uints(), other.(uint), t.mask)
			} else {
				err = incrTransU(t.Uints(), reuse.Uints(), other.(uint))
			}
			retVal = reuse
		case reflect.Uint8:
			if t.IsMasked() {
				err = incrTransU8Masked(t.Uint8s(), reuse.Uint8s(), other.(uint8), t.mask)
			} else {
				err = incrTransU8(t.Uint8s(), reuse.Uint8s(), other.(uint8))
			}
			retVal = reuse
		case reflect.Uint16:
			if t.IsMasked() {
				err = incrTransU16Masked(t.Uint16s(), reuse.Uint16s(), other.(uint16), t.mask)
			} else {
				err = incrTransU16(t.Uint16s(), reuse.Uint16s(), other.(uint16))
			}
			retVal = reuse
		case reflect.Uint32:
			if t.IsMasked() {
				err = incrTransU32Masked(t.Uint32s(), reuse.Uint32s(), other.(uint32), t.mask)
			} else {
				err = incrTransU32(t.Uint32s(), reuse.Uint32s(), other.(uint32))
			}
			retVal = reuse
		case reflect.Uint64:
			if t.IsMasked() {
				err = incrTransU64Masked(t.Uint64s(), reuse.Uint64s(), other.(uint64), t.mask)
			} else {
				err = incrTransU64(t.Uint64s(), reuse.Uint64s(), other.(uint64))
			}
			retVal = reuse
		case reflect.Float32:
			if t.IsMasked() {
				err = incrTransF32Masked(t.Float32s(), reuse.Float32s(), other.(float32), t.mask)
			} else {
				err = incrTransF32(t.Float32s(), reuse.Float32s(), other.(float32))
			}
			retVal = reuse
		case reflect.Float64:
			if t.IsMasked() {
				err = incrTransF64Masked(t.Float64s(), reuse.Float64s(), other.(float64), t.mask)
			} else {
				err = incrTransF64(t.Float64s(), reuse.Float64s(), other.(float64))
			}
			retVal = reuse
		case reflect.Complex64:
			if t.IsMasked() {
				err = incrTransC64Masked(t.Complex64s(), reuse.Complex64s(), other.(complex64), t.mask)
			} else {
				err = incrTransC64(t.Complex64s(), reuse.Complex64s(), other.(complex64))
			}
			retVal = reuse
		case reflect.Complex128:
			if t.IsMasked() {
				err = incrTransC128Masked(t.Complex128s(), reuse.Complex128s(), other.(complex128), t.mask)
			} else {
				err = incrTransC128(t.Complex128s(), reuse.Complex128s(), other.(complex128))
			}
			retVal = reuse
		}
	case toReuse:
		if t.IsMaterializable() {
			it := NewFlatIterator(t.AP)
			copyDenseIter(reuse, t, nil, it)
		} else {
			copyDense(reuse, t)
		}
		err = reuse.trans(other)
		retVal = reuse
	case safe:
		if t.IsMaterializable() {
			retVal = t.Materialize().(*Dense)
		} else {
			retVal = t.Clone().(*Dense)
		}
		err = retVal.trans(other)
	case !safe:
		err = t.trans(other)
		retVal = t
	}
	return
}
func (t *Dense) trans(other interface{}) (err error) {
	switch t.t.Kind() {
	case reflect.Int:
		b := other.(int)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Ints()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] + b
			}
			return nil
		}
		return transI(t.Ints(), b)
	case reflect.Int8:
		b := other.(int8)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Int8s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] + b
			}
			return nil
		}
		return transI8(t.Int8s(), b)
	case reflect.Int16:
		b := other.(int16)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Int16s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] + b
			}
			return nil
		}
		return transI16(t.Int16s(), b)
	case reflect.Int32:
		b := other.(int32)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Int32s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] + b
			}
			return nil
		}
		return transI32(t.Int32s(), b)
	case reflect.Int64:
		b := other.(int64)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Int64s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] + b
			}
			return nil
		}
		return transI64(t.Int64s(), b)
	case reflect.Uint:
		b := other.(uint)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Uints()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] + b
			}
			return nil
		}
		return transU(t.Uints(), b)
	case reflect.Uint8:
		b := other.(uint8)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Uint8s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] + b
			}
			return nil
		}
		return transU8(t.Uint8s(), b)
	case reflect.Uint16:
		b := other.(uint16)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Uint16s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] + b
			}
			return nil
		}
		return transU16(t.Uint16s(), b)
	case reflect.Uint32:
		b := other.(uint32)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Uint32s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] + b
			}
			return nil
		}
		return transU32(t.Uint32s(), b)
	case reflect.Uint64:
		b := other.(uint64)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Uint64s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] + b
			}
			return nil
		}
		return transU64(t.Uint64s(), b)
	case reflect.Float32:
		b := other.(float32)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Float32s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] + b
			}
			return nil
		}
		return transF32(t.Float32s(), b)
	case reflect.Float64:
		b := other.(float64)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Float64s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] + b
			}
			return nil
		}
		return transF64(t.Float64s(), b)
	case reflect.Complex64:
		b := other.(complex64)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Complex64s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] + b
			}
			return nil
		}
		return transC64(t.Complex64s(), b)
	case reflect.Complex128:
		b := other.(complex128)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Complex128s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] + b
			}
			return nil
		}
		return transC128(t.Complex128s(), b)
	}
	return nil
}

/* TransInv */

// TransInv performs subtraction on a *Dense and a scalar value. The scalar value has to be of the same
// type as defined in the *Dense, otherwise an error will be returned.
func (t *Dense) TransInv(other interface{}, opts ...FuncOpt) (retVal *Dense, err error) {
	reuse, safe, toReuse, incr, err := prepUnaryDense(t, opts...)
	if err != nil {
		return nil, err
	}
	if t.IsMasked() && (reuse != nil) {
		reuse.MaskFromDense(t)
	}

	switch {
	case incr:
		switch t.t.Kind() {
		case reflect.Int:
			if t.IsMasked() {
				err = incrTransInvIMasked(t.Ints(), reuse.Ints(), other.(int), t.mask)
			} else {
				err = incrTransInvI(t.Ints(), reuse.Ints(), other.(int))
			}
			retVal = reuse
		case reflect.Int8:
			if t.IsMasked() {
				err = incrTransInvI8Masked(t.Int8s(), reuse.Int8s(), other.(int8), t.mask)
			} else {
				err = incrTransInvI8(t.Int8s(), reuse.Int8s(), other.(int8))
			}
			retVal = reuse
		case reflect.Int16:
			if t.IsMasked() {
				err = incrTransInvI16Masked(t.Int16s(), reuse.Int16s(), other.(int16), t.mask)
			} else {
				err = incrTransInvI16(t.Int16s(), reuse.Int16s(), other.(int16))
			}
			retVal = reuse
		case reflect.Int32:
			if t.IsMasked() {
				err = incrTransInvI32Masked(t.Int32s(), reuse.Int32s(), other.(int32), t.mask)
			} else {
				err = incrTransInvI32(t.Int32s(), reuse.Int32s(), other.(int32))
			}
			retVal = reuse
		case reflect.Int64:
			if t.IsMasked() {
				err = incrTransInvI64Masked(t.Int64s(), reuse.Int64s(), other.(int64), t.mask)
			} else {
				err = incrTransInvI64(t.Int64s(), reuse.Int64s(), other.(int64))
			}
			retVal = reuse
		case reflect.Uint:
			if t.IsMasked() {
				err = incrTransInvUMasked(t.Uints(), reuse.Uints(), other.(uint), t.mask)
			} else {
				err = incrTransInvU(t.Uints(), reuse.Uints(), other.(uint))
			}
			retVal = reuse
		case reflect.Uint8:
			if t.IsMasked() {
				err = incrTransInvU8Masked(t.Uint8s(), reuse.Uint8s(), other.(uint8), t.mask)
			} else {
				err = incrTransInvU8(t.Uint8s(), reuse.Uint8s(), other.(uint8))
			}
			retVal = reuse
		case reflect.Uint16:
			if t.IsMasked() {
				err = incrTransInvU16Masked(t.Uint16s(), reuse.Uint16s(), other.(uint16), t.mask)
			} else {
				err = incrTransInvU16(t.Uint16s(), reuse.Uint16s(), other.(uint16))
			}
			retVal = reuse
		case reflect.Uint32:
			if t.IsMasked() {
				err = incrTransInvU32Masked(t.Uint32s(), reuse.Uint32s(), other.(uint32), t.mask)
			} else {
				err = incrTransInvU32(t.Uint32s(), reuse.Uint32s(), other.(uint32))
			}
			retVal = reuse
		case reflect.Uint64:
			if t.IsMasked() {
				err = incrTransInvU64Masked(t.Uint64s(), reuse.Uint64s(), other.(uint64), t.mask)
			} else {
				err = incrTransInvU64(t.Uint64s(), reuse.Uint64s(), other.(uint64))
			}
			retVal = reuse
		case reflect.Float32:
			if t.IsMasked() {
				err = incrTransInvF32Masked(t.Float32s(), reuse.Float32s(), other.(float32), t.mask)
			} else {
				err = incrTransInvF32(t.Float32s(), reuse.Float32s(), other.(float32))
			}
			retVal = reuse
		case reflect.Float64:
			if t.IsMasked() {
				err = incrTransInvF64Masked(t.Float64s(), reuse.Float64s(), other.(float64), t.mask)
			} else {
				err = incrTransInvF64(t.Float64s(), reuse.Float64s(), other.(float64))
			}
			retVal = reuse
		case reflect.Complex64:
			if t.IsMasked() {
				err = incrTransInvC64Masked(t.Complex64s(), reuse.Complex64s(), other.(complex64), t.mask)
			} else {
				err = incrTransInvC64(t.Complex64s(), reuse.Complex64s(), other.(complex64))
			}
			retVal = reuse
		case reflect.Complex128:
			if t.IsMasked() {
				err = incrTransInvC128Masked(t.Complex128s(), reuse.Complex128s(), other.(complex128), t.mask)
			} else {
				err = incrTransInvC128(t.Complex128s(), reuse.Complex128s(), other.(complex128))
			}
			retVal = reuse
		}
	case toReuse:
		if t.IsMaterializable() {
			it := NewFlatIterator(t.AP)
			copyDenseIter(reuse, t, nil, it)
		} else {
			copyDense(reuse, t)
		}
		err = reuse.transinv(other)
		retVal = reuse
	case safe:
		if t.IsMaterializable() {
			retVal = t.Materialize().(*Dense)
		} else {
			retVal = t.Clone().(*Dense)
		}
		err = retVal.transinv(other)
	case !safe:
		err = t.transinv(other)
		retVal = t
	}
	return
}
func (t *Dense) transinv(other interface{}) (err error) {
	switch t.t.Kind() {
	case reflect.Int:
		b := other.(int)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Ints()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] - b
			}
			return nil
		}
		return transinvI(t.Ints(), b)
	case reflect.Int8:
		b := other.(int8)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Int8s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] - b
			}
			return nil
		}
		return transinvI8(t.Int8s(), b)
	case reflect.Int16:
		b := other.(int16)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Int16s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] - b
			}
			return nil
		}
		return transinvI16(t.Int16s(), b)
	case reflect.Int32:
		b := other.(int32)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Int32s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] - b
			}
			return nil
		}
		return transinvI32(t.Int32s(), b)
	case reflect.Int64:
		b := other.(int64)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Int64s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] - b
			}
			return nil
		}
		return transinvI64(t.Int64s(), b)
	case reflect.Uint:
		b := other.(uint)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Uints()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] - b
			}
			return nil
		}
		return transinvU(t.Uints(), b)
	case reflect.Uint8:
		b := other.(uint8)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Uint8s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] - b
			}
			return nil
		}
		return transinvU8(t.Uint8s(), b)
	case reflect.Uint16:
		b := other.(uint16)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Uint16s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] - b
			}
			return nil
		}
		return transinvU16(t.Uint16s(), b)
	case reflect.Uint32:
		b := other.(uint32)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Uint32s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] - b
			}
			return nil
		}
		return transinvU32(t.Uint32s(), b)
	case reflect.Uint64:
		b := other.(uint64)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Uint64s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] - b
			}
			return nil
		}
		return transinvU64(t.Uint64s(), b)
	case reflect.Float32:
		b := other.(float32)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Float32s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] - b
			}
			return nil
		}
		return transinvF32(t.Float32s(), b)
	case reflect.Float64:
		b := other.(float64)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Float64s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] - b
			}
			return nil
		}
		return transinvF64(t.Float64s(), b)
	case reflect.Complex64:
		b := other.(complex64)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Complex64s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] - b
			}
			return nil
		}
		return transinvC64(t.Complex64s(), b)
	case reflect.Complex128:
		b := other.(complex128)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Complex128s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] - b
			}
			return nil
		}
		return transinvC128(t.Complex128s(), b)
	}
	return nil
}

/* TransInvR */

// TransInvR performs subtraction on a *Dense and a scalar value. The scalar value has to be of the same
// type as defined in the *Dense, otherwise an error will be returned.
func (t *Dense) TransInvR(other interface{}, opts ...FuncOpt) (retVal *Dense, err error) {
	reuse, safe, toReuse, incr, err := prepUnaryDense(t, opts...)
	if err != nil {
		return nil, err
	}
	if t.IsMasked() && (reuse != nil) {
		reuse.MaskFromDense(t)
	}

	switch {
	case incr:
		switch t.t.Kind() {
		case reflect.Int:
			if t.IsMasked() {
				err = incrTransInvRIMasked(t.Ints(), reuse.Ints(), other.(int), t.mask)
			} else {
				err = incrTransInvRI(t.Ints(), reuse.Ints(), other.(int))
			}
			retVal = reuse
		case reflect.Int8:
			if t.IsMasked() {
				err = incrTransInvRI8Masked(t.Int8s(), reuse.Int8s(), other.(int8), t.mask)
			} else {
				err = incrTransInvRI8(t.Int8s(), reuse.Int8s(), other.(int8))
			}
			retVal = reuse
		case reflect.Int16:
			if t.IsMasked() {
				err = incrTransInvRI16Masked(t.Int16s(), reuse.Int16s(), other.(int16), t.mask)
			} else {
				err = incrTransInvRI16(t.Int16s(), reuse.Int16s(), other.(int16))
			}
			retVal = reuse
		case reflect.Int32:
			if t.IsMasked() {
				err = incrTransInvRI32Masked(t.Int32s(), reuse.Int32s(), other.(int32), t.mask)
			} else {
				err = incrTransInvRI32(t.Int32s(), reuse.Int32s(), other.(int32))
			}
			retVal = reuse
		case reflect.Int64:
			if t.IsMasked() {
				err = incrTransInvRI64Masked(t.Int64s(), reuse.Int64s(), other.(int64), t.mask)
			} else {
				err = incrTransInvRI64(t.Int64s(), reuse.Int64s(), other.(int64))
			}
			retVal = reuse
		case reflect.Uint:
			if t.IsMasked() {
				err = incrTransInvRUMasked(t.Uints(), reuse.Uints(), other.(uint), t.mask)
			} else {
				err = incrTransInvRU(t.Uints(), reuse.Uints(), other.(uint))
			}
			retVal = reuse
		case reflect.Uint8:
			if t.IsMasked() {
				err = incrTransInvRU8Masked(t.Uint8s(), reuse.Uint8s(), other.(uint8), t.mask)
			} else {
				err = incrTransInvRU8(t.Uint8s(), reuse.Uint8s(), other.(uint8))
			}
			retVal = reuse
		case reflect.Uint16:
			if t.IsMasked() {
				err = incrTransInvRU16Masked(t.Uint16s(), reuse.Uint16s(), other.(uint16), t.mask)
			} else {
				err = incrTransInvRU16(t.Uint16s(), reuse.Uint16s(), other.(uint16))
			}
			retVal = reuse
		case reflect.Uint32:
			if t.IsMasked() {
				err = incrTransInvRU32Masked(t.Uint32s(), reuse.Uint32s(), other.(uint32), t.mask)
			} else {
				err = incrTransInvRU32(t.Uint32s(), reuse.Uint32s(), other.(uint32))
			}
			retVal = reuse
		case reflect.Uint64:
			if t.IsMasked() {
				err = incrTransInvRU64Masked(t.Uint64s(), reuse.Uint64s(), other.(uint64), t.mask)
			} else {
				err = incrTransInvRU64(t.Uint64s(), reuse.Uint64s(), other.(uint64))
			}
			retVal = reuse
		case reflect.Float32:
			if t.IsMasked() {
				err = incrTransInvRF32Masked(t.Float32s(), reuse.Float32s(), other.(float32), t.mask)
			} else {
				err = incrTransInvRF32(t.Float32s(), reuse.Float32s(), other.(float32))
			}
			retVal = reuse
		case reflect.Float64:
			if t.IsMasked() {
				err = incrTransInvRF64Masked(t.Float64s(), reuse.Float64s(), other.(float64), t.mask)
			} else {
				err = incrTransInvRF64(t.Float64s(), reuse.Float64s(), other.(float64))
			}
			retVal = reuse
		case reflect.Complex64:
			if t.IsMasked() {
				err = incrTransInvRC64Masked(t.Complex64s(), reuse.Complex64s(), other.(complex64), t.mask)
			} else {
				err = incrTransInvRC64(t.Complex64s(), reuse.Complex64s(), other.(complex64))
			}
			retVal = reuse
		case reflect.Complex128:
			if t.IsMasked() {
				err = incrTransInvRC128Masked(t.Complex128s(), reuse.Complex128s(), other.(complex128), t.mask)
			} else {
				err = incrTransInvRC128(t.Complex128s(), reuse.Complex128s(), other.(complex128))
			}
			retVal = reuse
		}
	case toReuse:
		if t.IsMaterializable() {
			it := NewFlatIterator(t.AP)
			copyDenseIter(reuse, t, nil, it)
		} else {
			copyDense(reuse, t)
		}
		err = reuse.transinvr(other)
		retVal = reuse
	case safe:
		if t.IsMaterializable() {
			retVal = t.Materialize().(*Dense)
		} else {
			retVal = t.Clone().(*Dense)
		}
		err = retVal.transinvr(other)
	case !safe:
		err = t.transinvr(other)
		retVal = t
	}
	return
}
func (t *Dense) transinvr(other interface{}) (err error) {
	switch t.t.Kind() {
	case reflect.Int:
		b := other.(int)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Ints()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = b - data[i]
			}
			return nil
		}
		return transinvrI(t.Ints(), b)
	case reflect.Int8:
		b := other.(int8)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Int8s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = b - data[i]
			}
			return nil
		}
		return transinvrI8(t.Int8s(), b)
	case reflect.Int16:
		b := other.(int16)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Int16s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = b - data[i]
			}
			return nil
		}
		return transinvrI16(t.Int16s(), b)
	case reflect.Int32:
		b := other.(int32)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Int32s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = b - data[i]
			}
			return nil
		}
		return transinvrI32(t.Int32s(), b)
	case reflect.Int64:
		b := other.(int64)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Int64s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = b - data[i]
			}
			return nil
		}
		return transinvrI64(t.Int64s(), b)
	case reflect.Uint:
		b := other.(uint)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Uints()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = b - data[i]
			}
			return nil
		}
		return transinvrU(t.Uints(), b)
	case reflect.Uint8:
		b := other.(uint8)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Uint8s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = b - data[i]
			}
			return nil
		}
		return transinvrU8(t.Uint8s(), b)
	case reflect.Uint16:
		b := other.(uint16)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Uint16s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = b - data[i]
			}
			return nil
		}
		return transinvrU16(t.Uint16s(), b)
	case reflect.Uint32:
		b := other.(uint32)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Uint32s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = b - data[i]
			}
			return nil
		}
		return transinvrU32(t.Uint32s(), b)
	case reflect.Uint64:
		b := other.(uint64)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Uint64s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = b - data[i]
			}
			return nil
		}
		return transinvrU64(t.Uint64s(), b)
	case reflect.Float32:
		b := other.(float32)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Float32s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = b - data[i]
			}
			return nil
		}
		return transinvrF32(t.Float32s(), b)
	case reflect.Float64:
		b := other.(float64)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Float64s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = b - data[i]
			}
			return nil
		}
		return transinvrF64(t.Float64s(), b)
	case reflect.Complex64:
		b := other.(complex64)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Complex64s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = b - data[i]
			}
			return nil
		}
		return transinvrC64(t.Complex64s(), b)
	case reflect.Complex128:
		b := other.(complex128)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Complex128s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = b - data[i]
			}
			return nil
		}
		return transinvrC128(t.Complex128s(), b)
	}
	return nil
}

/* Scale */

// Scale performs multiplication on a *Dense and a scalar value. The scalar value has to be of the same
// type as defined in the *Dense, otherwise an error will be returned.
func (t *Dense) Scale(other interface{}, opts ...FuncOpt) (retVal *Dense, err error) {
	reuse, safe, toReuse, incr, err := prepUnaryDense(t, opts...)
	if err != nil {
		return nil, err
	}
	if t.IsMasked() && (reuse != nil) {
		reuse.MaskFromDense(t)
	}

	switch {
	case incr:
		switch t.t.Kind() {
		case reflect.Int:
			if t.IsMasked() {
				err = incrScaleIMasked(t.Ints(), reuse.Ints(), other.(int), t.mask)
			} else {
				err = incrScaleI(t.Ints(), reuse.Ints(), other.(int))
			}
			retVal = reuse
		case reflect.Int8:
			if t.IsMasked() {
				err = incrScaleI8Masked(t.Int8s(), reuse.Int8s(), other.(int8), t.mask)
			} else {
				err = incrScaleI8(t.Int8s(), reuse.Int8s(), other.(int8))
			}
			retVal = reuse
		case reflect.Int16:
			if t.IsMasked() {
				err = incrScaleI16Masked(t.Int16s(), reuse.Int16s(), other.(int16), t.mask)
			} else {
				err = incrScaleI16(t.Int16s(), reuse.Int16s(), other.(int16))
			}
			retVal = reuse
		case reflect.Int32:
			if t.IsMasked() {
				err = incrScaleI32Masked(t.Int32s(), reuse.Int32s(), other.(int32), t.mask)
			} else {
				err = incrScaleI32(t.Int32s(), reuse.Int32s(), other.(int32))
			}
			retVal = reuse
		case reflect.Int64:
			if t.IsMasked() {
				err = incrScaleI64Masked(t.Int64s(), reuse.Int64s(), other.(int64), t.mask)
			} else {
				err = incrScaleI64(t.Int64s(), reuse.Int64s(), other.(int64))
			}
			retVal = reuse
		case reflect.Uint:
			if t.IsMasked() {
				err = incrScaleUMasked(t.Uints(), reuse.Uints(), other.(uint), t.mask)
			} else {
				err = incrScaleU(t.Uints(), reuse.Uints(), other.(uint))
			}
			retVal = reuse
		case reflect.Uint8:
			if t.IsMasked() {
				err = incrScaleU8Masked(t.Uint8s(), reuse.Uint8s(), other.(uint8), t.mask)
			} else {
				err = incrScaleU8(t.Uint8s(), reuse.Uint8s(), other.(uint8))
			}
			retVal = reuse
		case reflect.Uint16:
			if t.IsMasked() {
				err = incrScaleU16Masked(t.Uint16s(), reuse.Uint16s(), other.(uint16), t.mask)
			} else {
				err = incrScaleU16(t.Uint16s(), reuse.Uint16s(), other.(uint16))
			}
			retVal = reuse
		case reflect.Uint32:
			if t.IsMasked() {
				err = incrScaleU32Masked(t.Uint32s(), reuse.Uint32s(), other.(uint32), t.mask)
			} else {
				err = incrScaleU32(t.Uint32s(), reuse.Uint32s(), other.(uint32))
			}
			retVal = reuse
		case reflect.Uint64:
			if t.IsMasked() {
				err = incrScaleU64Masked(t.Uint64s(), reuse.Uint64s(), other.(uint64), t.mask)
			} else {
				err = incrScaleU64(t.Uint64s(), reuse.Uint64s(), other.(uint64))
			}
			retVal = reuse
		case reflect.Float32:
			if t.IsMasked() {
				err = incrScaleF32Masked(t.Float32s(), reuse.Float32s(), other.(float32), t.mask)
			} else {
				err = incrScaleF32(t.Float32s(), reuse.Float32s(), other.(float32))
			}
			retVal = reuse
		case reflect.Float64:
			if t.IsMasked() {
				err = incrScaleF64Masked(t.Float64s(), reuse.Float64s(), other.(float64), t.mask)
			} else {
				err = incrScaleF64(t.Float64s(), reuse.Float64s(), other.(float64))
			}
			retVal = reuse
		case reflect.Complex64:
			if t.IsMasked() {
				err = incrScaleC64Masked(t.Complex64s(), reuse.Complex64s(), other.(complex64), t.mask)
			} else {
				err = incrScaleC64(t.Complex64s(), reuse.Complex64s(), other.(complex64))
			}
			retVal = reuse
		case reflect.Complex128:
			if t.IsMasked() {
				err = incrScaleC128Masked(t.Complex128s(), reuse.Complex128s(), other.(complex128), t.mask)
			} else {
				err = incrScaleC128(t.Complex128s(), reuse.Complex128s(), other.(complex128))
			}
			retVal = reuse
		}
	case toReuse:
		if t.IsMaterializable() {
			it := NewFlatIterator(t.AP)
			copyDenseIter(reuse, t, nil, it)
		} else {
			copyDense(reuse, t)
		}
		err = reuse.scale(other)
		retVal = reuse
	case safe:
		if t.IsMaterializable() {
			retVal = t.Materialize().(*Dense)
		} else {
			retVal = t.Clone().(*Dense)
		}
		err = retVal.scale(other)
	case !safe:
		err = t.scale(other)
		retVal = t
	}
	return
}
func (t *Dense) scale(other interface{}) (err error) {
	switch t.t.Kind() {
	case reflect.Int:
		b := other.(int)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Ints()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] * b
			}
			return nil
		}
		return scaleI(t.Ints(), b)
	case reflect.Int8:
		b := other.(int8)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Int8s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] * b
			}
			return nil
		}
		return scaleI8(t.Int8s(), b)
	case reflect.Int16:
		b := other.(int16)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Int16s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] * b
			}
			return nil
		}
		return scaleI16(t.Int16s(), b)
	case reflect.Int32:
		b := other.(int32)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Int32s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] * b
			}
			return nil
		}
		return scaleI32(t.Int32s(), b)
	case reflect.Int64:
		b := other.(int64)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Int64s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] * b
			}
			return nil
		}
		return scaleI64(t.Int64s(), b)
	case reflect.Uint:
		b := other.(uint)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Uints()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] * b
			}
			return nil
		}
		return scaleU(t.Uints(), b)
	case reflect.Uint8:
		b := other.(uint8)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Uint8s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] * b
			}
			return nil
		}
		return scaleU8(t.Uint8s(), b)
	case reflect.Uint16:
		b := other.(uint16)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Uint16s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] * b
			}
			return nil
		}
		return scaleU16(t.Uint16s(), b)
	case reflect.Uint32:
		b := other.(uint32)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Uint32s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] * b
			}
			return nil
		}
		return scaleU32(t.Uint32s(), b)
	case reflect.Uint64:
		b := other.(uint64)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Uint64s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] * b
			}
			return nil
		}
		return scaleU64(t.Uint64s(), b)
	case reflect.Float32:
		b := other.(float32)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Float32s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] * b
			}
			return nil
		}
		return scaleF32(t.Float32s(), b)
	case reflect.Float64:
		b := other.(float64)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Float64s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] * b
			}
			return nil
		}
		return scaleF64(t.Float64s(), b)
	case reflect.Complex64:
		b := other.(complex64)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Complex64s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] * b
			}
			return nil
		}
		return scaleC64(t.Complex64s(), b)
	case reflect.Complex128:
		b := other.(complex128)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Complex128s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] * b
			}
			return nil
		}
		return scaleC128(t.Complex128s(), b)
	}
	return nil
}

/* ScaleInv */

// ScaleInv performs division on a *Dense and a scalar value. The scalar value has to be of the same
// type as defined in the *Dense, otherwise an error will be returned.
func (t *Dense) ScaleInv(other interface{}, opts ...FuncOpt) (retVal *Dense, err error) {
	reuse, safe, toReuse, incr, err := prepUnaryDense(t, opts...)
	if err != nil {
		return nil, err
	}
	if t.IsMasked() && (reuse != nil) {
		reuse.MaskFromDense(t)
	}

	switch {
	case incr:
		switch t.t.Kind() {
		case reflect.Int:
			if t.IsMasked() {
				err = incrScaleInvIMasked(t.Ints(), reuse.Ints(), other.(int), t.mask)
			} else {
				err = incrScaleInvI(t.Ints(), reuse.Ints(), other.(int))
			}
			retVal = reuse
		case reflect.Int8:
			if t.IsMasked() {
				err = incrScaleInvI8Masked(t.Int8s(), reuse.Int8s(), other.(int8), t.mask)
			} else {
				err = incrScaleInvI8(t.Int8s(), reuse.Int8s(), other.(int8))
			}
			retVal = reuse
		case reflect.Int16:
			if t.IsMasked() {
				err = incrScaleInvI16Masked(t.Int16s(), reuse.Int16s(), other.(int16), t.mask)
			} else {
				err = incrScaleInvI16(t.Int16s(), reuse.Int16s(), other.(int16))
			}
			retVal = reuse
		case reflect.Int32:
			if t.IsMasked() {
				err = incrScaleInvI32Masked(t.Int32s(), reuse.Int32s(), other.(int32), t.mask)
			} else {
				err = incrScaleInvI32(t.Int32s(), reuse.Int32s(), other.(int32))
			}
			retVal = reuse
		case reflect.Int64:
			if t.IsMasked() {
				err = incrScaleInvI64Masked(t.Int64s(), reuse.Int64s(), other.(int64), t.mask)
			} else {
				err = incrScaleInvI64(t.Int64s(), reuse.Int64s(), other.(int64))
			}
			retVal = reuse
		case reflect.Uint:
			if t.IsMasked() {
				err = incrScaleInvUMasked(t.Uints(), reuse.Uints(), other.(uint), t.mask)
			} else {
				err = incrScaleInvU(t.Uints(), reuse.Uints(), other.(uint))
			}
			retVal = reuse
		case reflect.Uint8:
			if t.IsMasked() {
				err = incrScaleInvU8Masked(t.Uint8s(), reuse.Uint8s(), other.(uint8), t.mask)
			} else {
				err = incrScaleInvU8(t.Uint8s(), reuse.Uint8s(), other.(uint8))
			}
			retVal = reuse
		case reflect.Uint16:
			if t.IsMasked() {
				err = incrScaleInvU16Masked(t.Uint16s(), reuse.Uint16s(), other.(uint16), t.mask)
			} else {
				err = incrScaleInvU16(t.Uint16s(), reuse.Uint16s(), other.(uint16))
			}
			retVal = reuse
		case reflect.Uint32:
			if t.IsMasked() {
				err = incrScaleInvU32Masked(t.Uint32s(), reuse.Uint32s(), other.(uint32), t.mask)
			} else {
				err = incrScaleInvU32(t.Uint32s(), reuse.Uint32s(), other.(uint32))
			}
			retVal = reuse
		case reflect.Uint64:
			if t.IsMasked() {
				err = incrScaleInvU64Masked(t.Uint64s(), reuse.Uint64s(), other.(uint64), t.mask)
			} else {
				err = incrScaleInvU64(t.Uint64s(), reuse.Uint64s(), other.(uint64))
			}
			retVal = reuse
		case reflect.Float32:
			if t.IsMasked() {
				err = incrScaleInvF32Masked(t.Float32s(), reuse.Float32s(), other.(float32), t.mask)
			} else {
				err = incrScaleInvF32(t.Float32s(), reuse.Float32s(), other.(float32))
			}
			retVal = reuse
		case reflect.Float64:
			if t.IsMasked() {
				err = incrScaleInvF64Masked(t.Float64s(), reuse.Float64s(), other.(float64), t.mask)
			} else {
				err = incrScaleInvF64(t.Float64s(), reuse.Float64s(), other.(float64))
			}
			retVal = reuse
		case reflect.Complex64:
			if t.IsMasked() {
				err = incrScaleInvC64Masked(t.Complex64s(), reuse.Complex64s(), other.(complex64), t.mask)
			} else {
				err = incrScaleInvC64(t.Complex64s(), reuse.Complex64s(), other.(complex64))
			}
			retVal = reuse
		case reflect.Complex128:
			if t.IsMasked() {
				err = incrScaleInvC128Masked(t.Complex128s(), reuse.Complex128s(), other.(complex128), t.mask)
			} else {
				err = incrScaleInvC128(t.Complex128s(), reuse.Complex128s(), other.(complex128))
			}
			retVal = reuse
		}
	case toReuse:
		if t.IsMaterializable() {
			it := NewFlatIterator(t.AP)
			copyDenseIter(reuse, t, nil, it)
		} else {
			copyDense(reuse, t)
		}
		err = reuse.scaleinv(other)
		retVal = reuse
	case safe:
		if t.IsMaterializable() {
			retVal = t.Materialize().(*Dense)
		} else {
			retVal = t.Clone().(*Dense)
		}
		err = retVal.scaleinv(other)
	case !safe:
		err = t.scaleinv(other)
		retVal = t
	}
	return
}
func (t *Dense) scaleinv(other interface{}) (err error) {
	switch t.t.Kind() {
	case reflect.Int:
		b := other.(int)
		if t.IsMaterializable() {
			if b == 0 {
				err = t.zeroIter()
				if err != nil {
					err = errors.Wrapf(err, div0, -1)
					return
				}
				err = errors.Errorf(div0, -1)
				return
			}
			it := IteratorFromDense(t)
			var i int
			data := t.Ints()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] / b
			}
			return nil
		}
		return scaleinvI(t.Ints(), b)
	case reflect.Int8:
		b := other.(int8)
		if t.IsMaterializable() {
			if b == 0 {
				err = t.zeroIter()
				if err != nil {
					err = errors.Wrapf(err, div0, -1)
					return
				}
				err = errors.Errorf(div0, -1)
				return
			}
			it := IteratorFromDense(t)
			var i int
			data := t.Int8s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] / b
			}
			return nil
		}
		return scaleinvI8(t.Int8s(), b)
	case reflect.Int16:
		b := other.(int16)
		if t.IsMaterializable() {
			if b == 0 {
				err = t.zeroIter()
				if err != nil {
					err = errors.Wrapf(err, div0, -1)
					return
				}
				err = errors.Errorf(div0, -1)
				return
			}
			it := IteratorFromDense(t)
			var i int
			data := t.Int16s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] / b
			}
			return nil
		}
		return scaleinvI16(t.Int16s(), b)
	case reflect.Int32:
		b := other.(int32)
		if t.IsMaterializable() {
			if b == 0 {
				err = t.zeroIter()
				if err != nil {
					err = errors.Wrapf(err, div0, -1)
					return
				}
				err = errors.Errorf(div0, -1)
				return
			}
			it := IteratorFromDense(t)
			var i int
			data := t.Int32s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] / b
			}
			return nil
		}
		return scaleinvI32(t.Int32s(), b)
	case reflect.Int64:
		b := other.(int64)
		if t.IsMaterializable() {
			if b == 0 {
				err = t.zeroIter()
				if err != nil {
					err = errors.Wrapf(err, div0, -1)
					return
				}
				err = errors.Errorf(div0, -1)
				return
			}
			it := IteratorFromDense(t)
			var i int
			data := t.Int64s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] / b
			}
			return nil
		}
		return scaleinvI64(t.Int64s(), b)
	case reflect.Uint:
		b := other.(uint)
		if t.IsMaterializable() {
			if b == 0 {
				err = t.zeroIter()
				if err != nil {
					err = errors.Wrapf(err, div0, -1)
					return
				}
				err = errors.Errorf(div0, -1)
				return
			}
			it := IteratorFromDense(t)
			var i int
			data := t.Uints()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] / b
			}
			return nil
		}
		return scaleinvU(t.Uints(), b)
	case reflect.Uint8:
		b := other.(uint8)
		if t.IsMaterializable() {
			if b == 0 {
				err = t.zeroIter()
				if err != nil {
					err = errors.Wrapf(err, div0, -1)
					return
				}
				err = errors.Errorf(div0, -1)
				return
			}
			it := IteratorFromDense(t)
			var i int
			data := t.Uint8s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] / b
			}
			return nil
		}
		return scaleinvU8(t.Uint8s(), b)
	case reflect.Uint16:
		b := other.(uint16)
		if t.IsMaterializable() {
			if b == 0 {
				err = t.zeroIter()
				if err != nil {
					err = errors.Wrapf(err, div0, -1)
					return
				}
				err = errors.Errorf(div0, -1)
				return
			}
			it := IteratorFromDense(t)
			var i int
			data := t.Uint16s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] / b
			}
			return nil
		}
		return scaleinvU16(t.Uint16s(), b)
	case reflect.Uint32:
		b := other.(uint32)
		if t.IsMaterializable() {
			if b == 0 {
				err = t.zeroIter()
				if err != nil {
					err = errors.Wrapf(err, div0, -1)
					return
				}
				err = errors.Errorf(div0, -1)
				return
			}
			it := IteratorFromDense(t)
			var i int
			data := t.Uint32s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] / b
			}
			return nil
		}
		return scaleinvU32(t.Uint32s(), b)
	case reflect.Uint64:
		b := other.(uint64)
		if t.IsMaterializable() {
			if b == 0 {
				err = t.zeroIter()
				if err != nil {
					err = errors.Wrapf(err, div0, -1)
					return
				}
				err = errors.Errorf(div0, -1)
				return
			}
			it := IteratorFromDense(t)
			var i int
			data := t.Uint64s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] / b
			}
			return nil
		}
		return scaleinvU64(t.Uint64s(), b)
	case reflect.Float32:
		b := other.(float32)
		if t.IsMaterializable() {
			if b == 0 {
				err = t.zeroIter()
				if err != nil {
					err = errors.Wrapf(err, div0, -1)
					return
				}
				err = errors.Errorf(div0, -1)
				return
			}
			it := IteratorFromDense(t)
			var i int
			data := t.Float32s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] / b
			}
			return nil
		}
		return scaleinvF32(t.Float32s(), b)
	case reflect.Float64:
		b := other.(float64)
		if t.IsMaterializable() {
			if b == 0 {
				err = t.zeroIter()
				if err != nil {
					err = errors.Wrapf(err, div0, -1)
					return
				}
				err = errors.Errorf(div0, -1)
				return
			}
			it := IteratorFromDense(t)
			var i int
			data := t.Float64s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] / b
			}
			return nil
		}
		return scaleinvF64(t.Float64s(), b)
	case reflect.Complex64:
		b := other.(complex64)
		if t.IsMaterializable() {
			if b == 0 {
				err = t.zeroIter()
				if err != nil {
					err = errors.Wrapf(err, div0, -1)
					return
				}
				err = errors.Errorf(div0, -1)
				return
			}
			it := IteratorFromDense(t)
			var i int
			data := t.Complex64s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] / b
			}
			return nil
		}
		return scaleinvC64(t.Complex64s(), b)
	case reflect.Complex128:
		b := other.(complex128)
		if t.IsMaterializable() {
			if b == 0 {
				err = t.zeroIter()
				if err != nil {
					err = errors.Wrapf(err, div0, -1)
					return
				}
				err = errors.Errorf(div0, -1)
				return
			}
			it := IteratorFromDense(t)
			var i int
			data := t.Complex128s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = data[i] / b
			}
			return nil
		}
		return scaleinvC128(t.Complex128s(), b)
	}
	return nil
}

/* ScaleInvR */

// ScaleInvR performs division on a *Dense and a scalar value. The scalar value has to be of the same
// type as defined in the *Dense, otherwise an error will be returned.
func (t *Dense) ScaleInvR(other interface{}, opts ...FuncOpt) (retVal *Dense, err error) {
	reuse, safe, toReuse, incr, err := prepUnaryDense(t, opts...)
	if err != nil {
		return nil, err
	}
	if t.IsMasked() && (reuse != nil) {
		reuse.MaskFromDense(t)
	}

	switch {
	case incr:
		switch t.t.Kind() {
		case reflect.Int:
			if t.IsMasked() {
				err = incrScaleInvRIMasked(t.Ints(), reuse.Ints(), other.(int), t.mask)
			} else {
				err = incrScaleInvRI(t.Ints(), reuse.Ints(), other.(int))
			}
			retVal = reuse
		case reflect.Int8:
			if t.IsMasked() {
				err = incrScaleInvRI8Masked(t.Int8s(), reuse.Int8s(), other.(int8), t.mask)
			} else {
				err = incrScaleInvRI8(t.Int8s(), reuse.Int8s(), other.(int8))
			}
			retVal = reuse
		case reflect.Int16:
			if t.IsMasked() {
				err = incrScaleInvRI16Masked(t.Int16s(), reuse.Int16s(), other.(int16), t.mask)
			} else {
				err = incrScaleInvRI16(t.Int16s(), reuse.Int16s(), other.(int16))
			}
			retVal = reuse
		case reflect.Int32:
			if t.IsMasked() {
				err = incrScaleInvRI32Masked(t.Int32s(), reuse.Int32s(), other.(int32), t.mask)
			} else {
				err = incrScaleInvRI32(t.Int32s(), reuse.Int32s(), other.(int32))
			}
			retVal = reuse
		case reflect.Int64:
			if t.IsMasked() {
				err = incrScaleInvRI64Masked(t.Int64s(), reuse.Int64s(), other.(int64), t.mask)
			} else {
				err = incrScaleInvRI64(t.Int64s(), reuse.Int64s(), other.(int64))
			}
			retVal = reuse
		case reflect.Uint:
			if t.IsMasked() {
				err = incrScaleInvRUMasked(t.Uints(), reuse.Uints(), other.(uint), t.mask)
			} else {
				err = incrScaleInvRU(t.Uints(), reuse.Uints(), other.(uint))
			}
			retVal = reuse
		case reflect.Uint8:
			if t.IsMasked() {
				err = incrScaleInvRU8Masked(t.Uint8s(), reuse.Uint8s(), other.(uint8), t.mask)
			} else {
				err = incrScaleInvRU8(t.Uint8s(), reuse.Uint8s(), other.(uint8))
			}
			retVal = reuse
		case reflect.Uint16:
			if t.IsMasked() {
				err = incrScaleInvRU16Masked(t.Uint16s(), reuse.Uint16s(), other.(uint16), t.mask)
			} else {
				err = incrScaleInvRU16(t.Uint16s(), reuse.Uint16s(), other.(uint16))
			}
			retVal = reuse
		case reflect.Uint32:
			if t.IsMasked() {
				err = incrScaleInvRU32Masked(t.Uint32s(), reuse.Uint32s(), other.(uint32), t.mask)
			} else {
				err = incrScaleInvRU32(t.Uint32s(), reuse.Uint32s(), other.(uint32))
			}
			retVal = reuse
		case reflect.Uint64:
			if t.IsMasked() {
				err = incrScaleInvRU64Masked(t.Uint64s(), reuse.Uint64s(), other.(uint64), t.mask)
			} else {
				err = incrScaleInvRU64(t.Uint64s(), reuse.Uint64s(), other.(uint64))
			}
			retVal = reuse
		case reflect.Float32:
			if t.IsMasked() {
				err = incrScaleInvRF32Masked(t.Float32s(), reuse.Float32s(), other.(float32), t.mask)
			} else {
				err = incrScaleInvRF32(t.Float32s(), reuse.Float32s(), other.(float32))
			}
			retVal = reuse
		case reflect.Float64:
			if t.IsMasked() {
				err = incrScaleInvRF64Masked(t.Float64s(), reuse.Float64s(), other.(float64), t.mask)
			} else {
				err = incrScaleInvRF64(t.Float64s(), reuse.Float64s(), other.(float64))
			}
			retVal = reuse
		case reflect.Complex64:
			if t.IsMasked() {
				err = incrScaleInvRC64Masked(t.Complex64s(), reuse.Complex64s(), other.(complex64), t.mask)
			} else {
				err = incrScaleInvRC64(t.Complex64s(), reuse.Complex64s(), other.(complex64))
			}
			retVal = reuse
		case reflect.Complex128:
			if t.IsMasked() {
				err = incrScaleInvRC128Masked(t.Complex128s(), reuse.Complex128s(), other.(complex128), t.mask)
			} else {
				err = incrScaleInvRC128(t.Complex128s(), reuse.Complex128s(), other.(complex128))
			}
			retVal = reuse
		}
	case toReuse:
		if t.IsMaterializable() {
			it := NewFlatIterator(t.AP)
			copyDenseIter(reuse, t, nil, it)
		} else {
			copyDense(reuse, t)
		}
		err = reuse.scaleinvr(other)
		retVal = reuse
	case safe:
		if t.IsMaterializable() {
			retVal = t.Materialize().(*Dense)
		} else {
			retVal = t.Clone().(*Dense)
		}
		err = retVal.scaleinvr(other)
	case !safe:
		err = t.scaleinvr(other)
		retVal = t
	}
	return
}
func (t *Dense) scaleinvr(other interface{}) (err error) {
	switch t.t.Kind() {
	case reflect.Int:
		b := other.(int)
		if t.IsMaterializable() {
			if b == 0 {
				err = t.zeroIter()
				if err != nil {
					err = errors.Wrapf(err, div0, -1)
					return
				}
				err = errors.Errorf(div0, -1)
				return
			}
			it := IteratorFromDense(t)
			var i int
			data := t.Ints()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = b / data[i]
			}
			return nil
		}
		return scaleinvrI(t.Ints(), b)
	case reflect.Int8:
		b := other.(int8)
		if t.IsMaterializable() {
			if b == 0 {
				err = t.zeroIter()
				if err != nil {
					err = errors.Wrapf(err, div0, -1)
					return
				}
				err = errors.Errorf(div0, -1)
				return
			}
			it := IteratorFromDense(t)
			var i int
			data := t.Int8s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = b / data[i]
			}
			return nil
		}
		return scaleinvrI8(t.Int8s(), b)
	case reflect.Int16:
		b := other.(int16)
		if t.IsMaterializable() {
			if b == 0 {
				err = t.zeroIter()
				if err != nil {
					err = errors.Wrapf(err, div0, -1)
					return
				}
				err = errors.Errorf(div0, -1)
				return
			}
			it := IteratorFromDense(t)
			var i int
			data := t.Int16s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = b / data[i]
			}
			return nil
		}
		return scaleinvrI16(t.Int16s(), b)
	case reflect.Int32:
		b := other.(int32)
		if t.IsMaterializable() {
			if b == 0 {
				err = t.zeroIter()
				if err != nil {
					err = errors.Wrapf(err, div0, -1)
					return
				}
				err = errors.Errorf(div0, -1)
				return
			}
			it := IteratorFromDense(t)
			var i int
			data := t.Int32s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = b / data[i]
			}
			return nil
		}
		return scaleinvrI32(t.Int32s(), b)
	case reflect.Int64:
		b := other.(int64)
		if t.IsMaterializable() {
			if b == 0 {
				err = t.zeroIter()
				if err != nil {
					err = errors.Wrapf(err, div0, -1)
					return
				}
				err = errors.Errorf(div0, -1)
				return
			}
			it := IteratorFromDense(t)
			var i int
			data := t.Int64s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = b / data[i]
			}
			return nil
		}
		return scaleinvrI64(t.Int64s(), b)
	case reflect.Uint:
		b := other.(uint)
		if t.IsMaterializable() {
			if b == 0 {
				err = t.zeroIter()
				if err != nil {
					err = errors.Wrapf(err, div0, -1)
					return
				}
				err = errors.Errorf(div0, -1)
				return
			}
			it := IteratorFromDense(t)
			var i int
			data := t.Uints()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = b / data[i]
			}
			return nil
		}
		return scaleinvrU(t.Uints(), b)
	case reflect.Uint8:
		b := other.(uint8)
		if t.IsMaterializable() {
			if b == 0 {
				err = t.zeroIter()
				if err != nil {
					err = errors.Wrapf(err, div0, -1)
					return
				}
				err = errors.Errorf(div0, -1)
				return
			}
			it := IteratorFromDense(t)
			var i int
			data := t.Uint8s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = b / data[i]
			}
			return nil
		}
		return scaleinvrU8(t.Uint8s(), b)
	case reflect.Uint16:
		b := other.(uint16)
		if t.IsMaterializable() {
			if b == 0 {
				err = t.zeroIter()
				if err != nil {
					err = errors.Wrapf(err, div0, -1)
					return
				}
				err = errors.Errorf(div0, -1)
				return
			}
			it := IteratorFromDense(t)
			var i int
			data := t.Uint16s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = b / data[i]
			}
			return nil
		}
		return scaleinvrU16(t.Uint16s(), b)
	case reflect.Uint32:
		b := other.(uint32)
		if t.IsMaterializable() {
			if b == 0 {
				err = t.zeroIter()
				if err != nil {
					err = errors.Wrapf(err, div0, -1)
					return
				}
				err = errors.Errorf(div0, -1)
				return
			}
			it := IteratorFromDense(t)
			var i int
			data := t.Uint32s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = b / data[i]
			}
			return nil
		}
		return scaleinvrU32(t.Uint32s(), b)
	case reflect.Uint64:
		b := other.(uint64)
		if t.IsMaterializable() {
			if b == 0 {
				err = t.zeroIter()
				if err != nil {
					err = errors.Wrapf(err, div0, -1)
					return
				}
				err = errors.Errorf(div0, -1)
				return
			}
			it := IteratorFromDense(t)
			var i int
			data := t.Uint64s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = b / data[i]
			}
			return nil
		}
		return scaleinvrU64(t.Uint64s(), b)
	case reflect.Float32:
		b := other.(float32)
		if t.IsMaterializable() {
			if b == 0 {
				err = t.zeroIter()
				if err != nil {
					err = errors.Wrapf(err, div0, -1)
					return
				}
				err = errors.Errorf(div0, -1)
				return
			}
			it := IteratorFromDense(t)
			var i int
			data := t.Float32s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = b / data[i]
			}
			return nil
		}
		return scaleinvrF32(t.Float32s(), b)
	case reflect.Float64:
		b := other.(float64)
		if t.IsMaterializable() {
			if b == 0 {
				err = t.zeroIter()
				if err != nil {
					err = errors.Wrapf(err, div0, -1)
					return
				}
				err = errors.Errorf(div0, -1)
				return
			}
			it := IteratorFromDense(t)
			var i int
			data := t.Float64s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = b / data[i]
			}
			return nil
		}
		return scaleinvrF64(t.Float64s(), b)
	case reflect.Complex64:
		b := other.(complex64)
		if t.IsMaterializable() {
			if b == 0 {
				err = t.zeroIter()
				if err != nil {
					err = errors.Wrapf(err, div0, -1)
					return
				}
				err = errors.Errorf(div0, -1)
				return
			}
			it := IteratorFromDense(t)
			var i int
			data := t.Complex64s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = b / data[i]
			}
			return nil
		}
		return scaleinvrC64(t.Complex64s(), b)
	case reflect.Complex128:
		b := other.(complex128)
		if t.IsMaterializable() {
			if b == 0 {
				err = t.zeroIter()
				if err != nil {
					err = errors.Wrapf(err, div0, -1)
					return
				}
				err = errors.Errorf(div0, -1)
				return
			}
			it := IteratorFromDense(t)
			var i int
			data := t.Complex128s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = b / data[i]
			}
			return nil
		}
		return scaleinvrC128(t.Complex128s(), b)
	}
	return nil
}

/* PowOf */

// PowOf performs exponentiation on a *Dense and a scalar value. The scalar value has to be of the same
// type as defined in the *Dense, otherwise an error will be returned.
func (t *Dense) PowOf(other interface{}, opts ...FuncOpt) (retVal *Dense, err error) {
	reuse, safe, toReuse, incr, err := prepUnaryDense(t, opts...)
	if err != nil {
		return nil, err
	}
	if t.IsMasked() && (reuse != nil) {
		reuse.MaskFromDense(t)
	}

	switch {
	case incr:
		switch t.t.Kind() {
		case reflect.Int:
			if t.IsMasked() {
				err = incrPowOfIMasked(t.Ints(), reuse.Ints(), other.(int), t.mask)
			} else {
				err = incrPowOfI(t.Ints(), reuse.Ints(), other.(int))
			}
			retVal = reuse
		case reflect.Int8:
			if t.IsMasked() {
				err = incrPowOfI8Masked(t.Int8s(), reuse.Int8s(), other.(int8), t.mask)
			} else {
				err = incrPowOfI8(t.Int8s(), reuse.Int8s(), other.(int8))
			}
			retVal = reuse
		case reflect.Int16:
			if t.IsMasked() {
				err = incrPowOfI16Masked(t.Int16s(), reuse.Int16s(), other.(int16), t.mask)
			} else {
				err = incrPowOfI16(t.Int16s(), reuse.Int16s(), other.(int16))
			}
			retVal = reuse
		case reflect.Int32:
			if t.IsMasked() {
				err = incrPowOfI32Masked(t.Int32s(), reuse.Int32s(), other.(int32), t.mask)
			} else {
				err = incrPowOfI32(t.Int32s(), reuse.Int32s(), other.(int32))
			}
			retVal = reuse
		case reflect.Int64:
			if t.IsMasked() {
				err = incrPowOfI64Masked(t.Int64s(), reuse.Int64s(), other.(int64), t.mask)
			} else {
				err = incrPowOfI64(t.Int64s(), reuse.Int64s(), other.(int64))
			}
			retVal = reuse
		case reflect.Uint:
			if t.IsMasked() {
				err = incrPowOfUMasked(t.Uints(), reuse.Uints(), other.(uint), t.mask)
			} else {
				err = incrPowOfU(t.Uints(), reuse.Uints(), other.(uint))
			}
			retVal = reuse
		case reflect.Uint8:
			if t.IsMasked() {
				err = incrPowOfU8Masked(t.Uint8s(), reuse.Uint8s(), other.(uint8), t.mask)
			} else {
				err = incrPowOfU8(t.Uint8s(), reuse.Uint8s(), other.(uint8))
			}
			retVal = reuse
		case reflect.Uint16:
			if t.IsMasked() {
				err = incrPowOfU16Masked(t.Uint16s(), reuse.Uint16s(), other.(uint16), t.mask)
			} else {
				err = incrPowOfU16(t.Uint16s(), reuse.Uint16s(), other.(uint16))
			}
			retVal = reuse
		case reflect.Uint32:
			if t.IsMasked() {
				err = incrPowOfU32Masked(t.Uint32s(), reuse.Uint32s(), other.(uint32), t.mask)
			} else {
				err = incrPowOfU32(t.Uint32s(), reuse.Uint32s(), other.(uint32))
			}
			retVal = reuse
		case reflect.Uint64:
			if t.IsMasked() {
				err = incrPowOfU64Masked(t.Uint64s(), reuse.Uint64s(), other.(uint64), t.mask)
			} else {
				err = incrPowOfU64(t.Uint64s(), reuse.Uint64s(), other.(uint64))
			}
			retVal = reuse
		case reflect.Float32:
			if t.IsMasked() {
				err = incrPowOfF32Masked(t.Float32s(), reuse.Float32s(), other.(float32), t.mask)
			} else {
				err = incrPowOfF32(t.Float32s(), reuse.Float32s(), other.(float32))
			}
			retVal = reuse
		case reflect.Float64:
			if t.IsMasked() {
				err = incrPowOfF64Masked(t.Float64s(), reuse.Float64s(), other.(float64), t.mask)
			} else {
				err = incrPowOfF64(t.Float64s(), reuse.Float64s(), other.(float64))
			}
			retVal = reuse
		case reflect.Complex64:
			if t.IsMasked() {
				err = incrPowOfC64Masked(t.Complex64s(), reuse.Complex64s(), other.(complex64), t.mask)
			} else {
				err = incrPowOfC64(t.Complex64s(), reuse.Complex64s(), other.(complex64))
			}
			retVal = reuse
		case reflect.Complex128:
			if t.IsMasked() {
				err = incrPowOfC128Masked(t.Complex128s(), reuse.Complex128s(), other.(complex128), t.mask)
			} else {
				err = incrPowOfC128(t.Complex128s(), reuse.Complex128s(), other.(complex128))
			}
			retVal = reuse
		}
	case toReuse:
		if t.IsMaterializable() {
			it := NewFlatIterator(t.AP)
			copyDenseIter(reuse, t, nil, it)
		} else {
			copyDense(reuse, t)
		}
		err = reuse.powof(other)
		retVal = reuse
	case safe:
		if t.IsMaterializable() {
			retVal = t.Materialize().(*Dense)
		} else {
			retVal = t.Clone().(*Dense)
		}
		err = retVal.powof(other)
	case !safe:
		err = t.powof(other)
		retVal = t
	}
	return
}
func (t *Dense) powof(other interface{}) (err error) {
	switch t.t.Kind() {
	case reflect.Int:
		b := other.(int)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Ints()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = int(math.Pow(float64(data[i]), float64(b)))
			}
			return nil
		}
		return powofI(t.Ints(), b)
	case reflect.Int8:
		b := other.(int8)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Int8s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = int8(math.Pow(float64(data[i]), float64(b)))
			}
			return nil
		}
		return powofI8(t.Int8s(), b)
	case reflect.Int16:
		b := other.(int16)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Int16s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = int16(math.Pow(float64(data[i]), float64(b)))
			}
			return nil
		}
		return powofI16(t.Int16s(), b)
	case reflect.Int32:
		b := other.(int32)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Int32s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = int32(math.Pow(float64(data[i]), float64(b)))
			}
			return nil
		}
		return powofI32(t.Int32s(), b)
	case reflect.Int64:
		b := other.(int64)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Int64s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = int64(math.Pow(float64(data[i]), float64(b)))
			}
			return nil
		}
		return powofI64(t.Int64s(), b)
	case reflect.Uint:
		b := other.(uint)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Uints()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = uint(math.Pow(float64(data[i]), float64(b)))
			}
			return nil
		}
		return powofU(t.Uints(), b)
	case reflect.Uint8:
		b := other.(uint8)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Uint8s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = uint8(math.Pow(float64(data[i]), float64(b)))
			}
			return nil
		}
		return powofU8(t.Uint8s(), b)
	case reflect.Uint16:
		b := other.(uint16)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Uint16s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = uint16(math.Pow(float64(data[i]), float64(b)))
			}
			return nil
		}
		return powofU16(t.Uint16s(), b)
	case reflect.Uint32:
		b := other.(uint32)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Uint32s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = uint32(math.Pow(float64(data[i]), float64(b)))
			}
			return nil
		}
		return powofU32(t.Uint32s(), b)
	case reflect.Uint64:
		b := other.(uint64)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Uint64s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = uint64(math.Pow(float64(data[i]), float64(b)))
			}
			return nil
		}
		return powofU64(t.Uint64s(), b)
	case reflect.Float32:
		b := other.(float32)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Float32s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = math32.Pow(data[i], b)
			}
			return nil
		}
		return powofF32(t.Float32s(), b)
	case reflect.Float64:
		b := other.(float64)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Float64s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = math.Pow(data[i], b)
			}
			return nil
		}
		return powofF64(t.Float64s(), b)
	case reflect.Complex64:
		b := other.(complex64)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Complex64s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = complex64(cmplx.Pow(complex128(data[i]), complex128(b)))
			}
			return nil
		}
		return powofC64(t.Complex64s(), b)
	case reflect.Complex128:
		b := other.(complex128)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Complex128s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = cmplx.Pow(data[i], b)
			}
			return nil
		}
		return powofC128(t.Complex128s(), b)
	}
	return nil
}

/* PowOfR */

// PowOfR performs exponentiation on a *Dense and a scalar value. The scalar value has to be of the same
// type as defined in the *Dense, otherwise an error will be returned.
func (t *Dense) PowOfR(other interface{}, opts ...FuncOpt) (retVal *Dense, err error) {
	reuse, safe, toReuse, incr, err := prepUnaryDense(t, opts...)
	if err != nil {
		return nil, err
	}
	if t.IsMasked() && (reuse != nil) {
		reuse.MaskFromDense(t)
	}

	switch {
	case incr:
		switch t.t.Kind() {
		case reflect.Int:
			if t.IsMasked() {
				err = incrPowOfRIMasked(t.Ints(), reuse.Ints(), other.(int), t.mask)
			} else {
				err = incrPowOfRI(t.Ints(), reuse.Ints(), other.(int))
			}
			retVal = reuse
		case reflect.Int8:
			if t.IsMasked() {
				err = incrPowOfRI8Masked(t.Int8s(), reuse.Int8s(), other.(int8), t.mask)
			} else {
				err = incrPowOfRI8(t.Int8s(), reuse.Int8s(), other.(int8))
			}
			retVal = reuse
		case reflect.Int16:
			if t.IsMasked() {
				err = incrPowOfRI16Masked(t.Int16s(), reuse.Int16s(), other.(int16), t.mask)
			} else {
				err = incrPowOfRI16(t.Int16s(), reuse.Int16s(), other.(int16))
			}
			retVal = reuse
		case reflect.Int32:
			if t.IsMasked() {
				err = incrPowOfRI32Masked(t.Int32s(), reuse.Int32s(), other.(int32), t.mask)
			} else {
				err = incrPowOfRI32(t.Int32s(), reuse.Int32s(), other.(int32))
			}
			retVal = reuse
		case reflect.Int64:
			if t.IsMasked() {
				err = incrPowOfRI64Masked(t.Int64s(), reuse.Int64s(), other.(int64), t.mask)
			} else {
				err = incrPowOfRI64(t.Int64s(), reuse.Int64s(), other.(int64))
			}
			retVal = reuse
		case reflect.Uint:
			if t.IsMasked() {
				err = incrPowOfRUMasked(t.Uints(), reuse.Uints(), other.(uint), t.mask)
			} else {
				err = incrPowOfRU(t.Uints(), reuse.Uints(), other.(uint))
			}
			retVal = reuse
		case reflect.Uint8:
			if t.IsMasked() {
				err = incrPowOfRU8Masked(t.Uint8s(), reuse.Uint8s(), other.(uint8), t.mask)
			} else {
				err = incrPowOfRU8(t.Uint8s(), reuse.Uint8s(), other.(uint8))
			}
			retVal = reuse
		case reflect.Uint16:
			if t.IsMasked() {
				err = incrPowOfRU16Masked(t.Uint16s(), reuse.Uint16s(), other.(uint16), t.mask)
			} else {
				err = incrPowOfRU16(t.Uint16s(), reuse.Uint16s(), other.(uint16))
			}
			retVal = reuse
		case reflect.Uint32:
			if t.IsMasked() {
				err = incrPowOfRU32Masked(t.Uint32s(), reuse.Uint32s(), other.(uint32), t.mask)
			} else {
				err = incrPowOfRU32(t.Uint32s(), reuse.Uint32s(), other.(uint32))
			}
			retVal = reuse
		case reflect.Uint64:
			if t.IsMasked() {
				err = incrPowOfRU64Masked(t.Uint64s(), reuse.Uint64s(), other.(uint64), t.mask)
			} else {
				err = incrPowOfRU64(t.Uint64s(), reuse.Uint64s(), other.(uint64))
			}
			retVal = reuse
		case reflect.Float32:
			if t.IsMasked() {
				err = incrPowOfRF32Masked(t.Float32s(), reuse.Float32s(), other.(float32), t.mask)
			} else {
				err = incrPowOfRF32(t.Float32s(), reuse.Float32s(), other.(float32))
			}
			retVal = reuse
		case reflect.Float64:
			if t.IsMasked() {
				err = incrPowOfRF64Masked(t.Float64s(), reuse.Float64s(), other.(float64), t.mask)
			} else {
				err = incrPowOfRF64(t.Float64s(), reuse.Float64s(), other.(float64))
			}
			retVal = reuse
		case reflect.Complex64:
			if t.IsMasked() {
				err = incrPowOfRC64Masked(t.Complex64s(), reuse.Complex64s(), other.(complex64), t.mask)
			} else {
				err = incrPowOfRC64(t.Complex64s(), reuse.Complex64s(), other.(complex64))
			}
			retVal = reuse
		case reflect.Complex128:
			if t.IsMasked() {
				err = incrPowOfRC128Masked(t.Complex128s(), reuse.Complex128s(), other.(complex128), t.mask)
			} else {
				err = incrPowOfRC128(t.Complex128s(), reuse.Complex128s(), other.(complex128))
			}
			retVal = reuse
		}
	case toReuse:
		if t.IsMaterializable() {
			it := NewFlatIterator(t.AP)
			copyDenseIter(reuse, t, nil, it)
		} else {
			copyDense(reuse, t)
		}
		err = reuse.powofr(other)
		retVal = reuse
	case safe:
		if t.IsMaterializable() {
			retVal = t.Materialize().(*Dense)
		} else {
			retVal = t.Clone().(*Dense)
		}
		err = retVal.powofr(other)
	case !safe:
		err = t.powofr(other)
		retVal = t
	}
	return
}
func (t *Dense) powofr(other interface{}) (err error) {
	switch t.t.Kind() {
	case reflect.Int:
		b := other.(int)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Ints()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = int(math.Pow(float64(data[i]), float64(b)))
			}
			return nil
		}
		return powofrI(t.Ints(), b)
	case reflect.Int8:
		b := other.(int8)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Int8s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = int8(math.Pow(float64(data[i]), float64(b)))
			}
			return nil
		}
		return powofrI8(t.Int8s(), b)
	case reflect.Int16:
		b := other.(int16)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Int16s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = int16(math.Pow(float64(data[i]), float64(b)))
			}
			return nil
		}
		return powofrI16(t.Int16s(), b)
	case reflect.Int32:
		b := other.(int32)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Int32s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = int32(math.Pow(float64(data[i]), float64(b)))
			}
			return nil
		}
		return powofrI32(t.Int32s(), b)
	case reflect.Int64:
		b := other.(int64)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Int64s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = int64(math.Pow(float64(data[i]), float64(b)))
			}
			return nil
		}
		return powofrI64(t.Int64s(), b)
	case reflect.Uint:
		b := other.(uint)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Uints()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = uint(math.Pow(float64(data[i]), float64(b)))
			}
			return nil
		}
		return powofrU(t.Uints(), b)
	case reflect.Uint8:
		b := other.(uint8)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Uint8s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = uint8(math.Pow(float64(data[i]), float64(b)))
			}
			return nil
		}
		return powofrU8(t.Uint8s(), b)
	case reflect.Uint16:
		b := other.(uint16)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Uint16s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = uint16(math.Pow(float64(data[i]), float64(b)))
			}
			return nil
		}
		return powofrU16(t.Uint16s(), b)
	case reflect.Uint32:
		b := other.(uint32)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Uint32s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = uint32(math.Pow(float64(data[i]), float64(b)))
			}
			return nil
		}
		return powofrU32(t.Uint32s(), b)
	case reflect.Uint64:
		b := other.(uint64)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Uint64s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = uint64(math.Pow(float64(data[i]), float64(b)))
			}
			return nil
		}
		return powofrU64(t.Uint64s(), b)
	case reflect.Float32:
		b := other.(float32)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Float32s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = math32.Pow(data[i], b)
			}
			return nil
		}
		return powofrF32(t.Float32s(), b)
	case reflect.Float64:
		b := other.(float64)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Float64s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = math.Pow(data[i], b)
			}
			return nil
		}
		return powofrF64(t.Float64s(), b)
	case reflect.Complex64:
		b := other.(complex64)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Complex64s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = complex64(cmplx.Pow(complex128(data[i]), complex128(b)))
			}
			return nil
		}
		return powofrC64(t.Complex64s(), b)
	case reflect.Complex128:
		b := other.(complex128)
		if t.IsMaterializable() {
			it := IteratorFromDense(t)
			var i int
			data := t.Complex128s()
			for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
				data[i] = cmplx.Pow(data[i], b)
			}
			return nil
		}
		return powofrC128(t.Complex128s(), b)
	}
	return nil
}
