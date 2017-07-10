package stdeng

import (
	"reflect"

	"github.com/chewxy/vecf32"
	"github.com/chewxy/vecf64"
	"github.com/pkg/errors"
)

/*
GENERATED FILE. DO NOT EDIT
*/

// Sub performs subtraction: a -= b.
// If the element type are numbers, it will fall to using standard number subtraction.
// If the element type are strings, concatenation will happen instead.
// If the element type is unknown, it will attempt to look for the Suber interface, and use the Suber method.
// Failing that, it produces an error.
//
// a and b may be scalars. In the case where one of which is vector and the other is a scalar, it's the vector that will
// be clobbered. If both are scalars, a will be clobbered
func (e E) Sub(t reflect.Type, a, b Array) (err error) {
	as := isScalar(a)
	bs := isScalar(b)
	switch t {
	case Int:
		at := a.Ints()
		bt := b.Ints()

		switch {
		case as && bs:
			at[0] -= bt[0]
		case as && !bs:
			subIVS(bt, at[0])
		case bs && !as:
			subIVS(at, bt[0])
		default:
			subI(at, bt)
		}
		return nil
	case Int8:
		at := a.Int8s()
		bt := b.Int8s()

		switch {
		case as && bs:
			at[0] -= bt[0]
		case as && !bs:
			subI8VS(bt, at[0])
		case bs && !as:
			subI8VS(at, bt[0])
		default:
			subI8(at, bt)
		}
		return nil
	case Int16:
		at := a.Int16s()
		bt := b.Int16s()

		switch {
		case as && bs:
			at[0] -= bt[0]
		case as && !bs:
			subI16VS(bt, at[0])
		case bs && !as:
			subI16VS(at, bt[0])
		default:
			subI16(at, bt)
		}
		return nil
	case Int32:
		at := a.Int32s()
		bt := b.Int32s()

		switch {
		case as && bs:
			at[0] -= bt[0]
		case as && !bs:
			subI32VS(bt, at[0])
		case bs && !as:
			subI32VS(at, bt[0])
		default:
			subI32(at, bt)
		}
		return nil
	case Int64:
		at := a.Int64s()
		bt := b.Int64s()

		switch {
		case as && bs:
			at[0] -= bt[0]
		case as && !bs:
			subI64VS(bt, at[0])
		case bs && !as:
			subI64VS(at, bt[0])
		default:
			subI64(at, bt)
		}
		return nil
	case Uint:
		at := a.Uints()
		bt := b.Uints()

		switch {
		case as && bs:
			at[0] -= bt[0]
		case as && !bs:
			subUVS(bt, at[0])
		case bs && !as:
			subUVS(at, bt[0])
		default:
			subU(at, bt)
		}
		return nil
	case Uint8:
		at := a.Uint8s()
		bt := b.Uint8s()

		switch {
		case as && bs:
			at[0] -= bt[0]
		case as && !bs:
			subU8VS(bt, at[0])
		case bs && !as:
			subU8VS(at, bt[0])
		default:
			subU8(at, bt)
		}
		return nil
	case Uint16:
		at := a.Uint16s()
		bt := b.Uint16s()

		switch {
		case as && bs:
			at[0] -= bt[0]
		case as && !bs:
			subU16VS(bt, at[0])
		case bs && !as:
			subU16VS(at, bt[0])
		default:
			subU16(at, bt)
		}
		return nil
	case Uint32:
		at := a.Uint32s()
		bt := b.Uint32s()

		switch {
		case as && bs:
			at[0] -= bt[0]
		case as && !bs:
			subU32VS(bt, at[0])
		case bs && !as:
			subU32VS(at, bt[0])
		default:
			subU32(at, bt)
		}
		return nil
	case Uint64:
		at := a.Uint64s()
		bt := b.Uint64s()

		switch {
		case as && bs:
			at[0] -= bt[0]
		case as && !bs:
			subU64VS(bt, at[0])
		case bs && !as:
			subU64VS(at, bt[0])
		default:
			subU64(at, bt)
		}
		return nil
	case Float32:
		at := a.Float32s()
		bt := b.Float32s()

		switch {
		case as && bs:
			at[0] -= bt[0]
		case as && !bs:
			subF32VS(bt, at[0])
		case bs && !as:
			subF32VS(at, bt[0])
		default:
			subF32(at, bt)
		}
		return nil
	case Float64:
		at := a.Float64s()
		bt := b.Float64s()

		switch {
		case as && bs:
			at[0] -= bt[0]
		case as && !bs:
			subF64VS(bt, at[0])
		case bs && !as:
			subF64VS(at, bt[0])
		default:
			subF64(at, bt)
		}
		return nil
	case Complex64:
		at := a.Complex64s()
		bt := b.Complex64s()

		switch {
		case as && bs:
			at[0] -= bt[0]
		case as && !bs:
			subC64VS(bt, at[0])
		case bs && !as:
			subC64VS(at, bt[0])
		default:
			subC64(at, bt)
		}
		return nil
	case Complex128:
		at := a.Complex128s()
		bt := b.Complex128s()

		switch {
		case as && bs:
			at[0] -= bt[0]
		case as && !bs:
			subC128VS(bt, at[0])
		case bs && !as:
			subC128VS(at, bt[0])
		default:
			subC128(at, bt)
		}
		return nil
	default:
		return errors.Errorf("NYI")
	}
}

// SubIter performs subtraction a -= b, guided by the iterator
func (e E) SubIter(t reflect.Type, a, b Array, ait, bit Iterator) error {
	as := isScalar(a)
	bs := isScalar(b)
	switch t {
	case Int:
		at := a.Ints()
		bt := b.Ints()

		switch {
		case as && bs:
			at[0] -= bt[0]
			return nil
		case as && !bs:
			return subIIterVS(bt, at[0], bit)
		case bs && !as:
			return subIIterVS(at, bt[0], ait)
		default:
			return subIIter(at, bt, ait, bit)
		}
	case Int8:
		at := a.Int8s()
		bt := b.Int8s()

		switch {
		case as && bs:
			at[0] -= bt[0]
			return nil
		case as && !bs:
			return subI8IterVS(bt, at[0], bit)
		case bs && !as:
			return subI8IterVS(at, bt[0], ait)
		default:
			return subI8Iter(at, bt, ait, bit)
		}
	case Int16:
		at := a.Int16s()
		bt := b.Int16s()

		switch {
		case as && bs:
			at[0] -= bt[0]
			return nil
		case as && !bs:
			return subI16IterVS(bt, at[0], bit)
		case bs && !as:
			return subI16IterVS(at, bt[0], ait)
		default:
			return subI16Iter(at, bt, ait, bit)
		}
	case Int32:
		at := a.Int32s()
		bt := b.Int32s()

		switch {
		case as && bs:
			at[0] -= bt[0]
			return nil
		case as && !bs:
			return subI32IterVS(bt, at[0], bit)
		case bs && !as:
			return subI32IterVS(at, bt[0], ait)
		default:
			return subI32Iter(at, bt, ait, bit)
		}
	case Int64:
		at := a.Int64s()
		bt := b.Int64s()

		switch {
		case as && bs:
			at[0] -= bt[0]
			return nil
		case as && !bs:
			return subI64IterVS(bt, at[0], bit)
		case bs && !as:
			return subI64IterVS(at, bt[0], ait)
		default:
			return subI64Iter(at, bt, ait, bit)
		}
	case Uint:
		at := a.Uints()
		bt := b.Uints()

		switch {
		case as && bs:
			at[0] -= bt[0]
			return nil
		case as && !bs:
			return subUIterVS(bt, at[0], bit)
		case bs && !as:
			return subUIterVS(at, bt[0], ait)
		default:
			return subUIter(at, bt, ait, bit)
		}
	case Uint8:
		at := a.Uint8s()
		bt := b.Uint8s()

		switch {
		case as && bs:
			at[0] -= bt[0]
			return nil
		case as && !bs:
			return subU8IterVS(bt, at[0], bit)
		case bs && !as:
			return subU8IterVS(at, bt[0], ait)
		default:
			return subU8Iter(at, bt, ait, bit)
		}
	case Uint16:
		at := a.Uint16s()
		bt := b.Uint16s()

		switch {
		case as && bs:
			at[0] -= bt[0]
			return nil
		case as && !bs:
			return subU16IterVS(bt, at[0], bit)
		case bs && !as:
			return subU16IterVS(at, bt[0], ait)
		default:
			return subU16Iter(at, bt, ait, bit)
		}
	case Uint32:
		at := a.Uint32s()
		bt := b.Uint32s()

		switch {
		case as && bs:
			at[0] -= bt[0]
			return nil
		case as && !bs:
			return subU32IterVS(bt, at[0], bit)
		case bs && !as:
			return subU32IterVS(at, bt[0], ait)
		default:
			return subU32Iter(at, bt, ait, bit)
		}
	case Uint64:
		at := a.Uint64s()
		bt := b.Uint64s()

		switch {
		case as && bs:
			at[0] -= bt[0]
			return nil
		case as && !bs:
			return subU64IterVS(bt, at[0], bit)
		case bs && !as:
			return subU64IterVS(at, bt[0], ait)
		default:
			return subU64Iter(at, bt, ait, bit)
		}
	case Float32:
		at := a.Float32s()
		bt := b.Float32s()

		switch {
		case as && bs:
			at[0] -= bt[0]
			return nil
		case as && !bs:
			return subF32IterVS(bt, at[0], bit)
		case bs && !as:
			return subF32IterVS(at, bt[0], ait)
		default:
			return subF32Iter(at, bt, ait, bit)
		}
	case Float64:
		at := a.Float64s()
		bt := b.Float64s()

		switch {
		case as && bs:
			at[0] -= bt[0]
			return nil
		case as && !bs:
			return subF64IterVS(bt, at[0], bit)
		case bs && !as:
			return subF64IterVS(at, bt[0], ait)
		default:
			return subF64Iter(at, bt, ait, bit)
		}
	case Complex64:
		at := a.Complex64s()
		bt := b.Complex64s()

		switch {
		case as && bs:
			at[0] -= bt[0]
			return nil
		case as && !bs:
			return subC64IterVS(bt, at[0], bit)
		case bs && !as:
			return subC64IterVS(at, bt[0], ait)
		default:
			return subC64Iter(at, bt, ait, bit)
		}
	case Complex128:
		at := a.Complex128s()
		bt := b.Complex128s()

		switch {
		case as && bs:
			at[0] -= bt[0]
			return nil
		case as && !bs:
			return subC128IterVS(bt, at[0], bit)
		case bs && !as:
			return subC128IterVS(at, bt[0], ait)
		default:
			return subC128Iter(at, bt, ait, bit)
		}
	default:
		return errors.Errorf("NYI")
	}
}

// SubIncr performs incr += a - b
func (e E) SubIncr(t reflect.Type, a, b, incr Array) error {
	as := isScalar(a)
	bs := isScalar(b)
	is := isScalar(incr)

	if (as && !bs) || (bs && !as) && is {
		return errors.Errorf("Cannot increment on a scalar increment")
	}

	switch t {
	case Int:
		at := a.Ints()
		bt := b.Ints()
		it := incr.Ints()

		switch {
		case as && bs:
			if !is {
				at[0] -= bt[0]
				return e.Add(t, incr, a)
			}
			it[0] += at[0] - bt[0]
			return nil
		case as && !bs:
			subIncrIVS(bt, at[0], it)
		case bs && !as:
			subIncrIVS(at, bt[0], it)
		default:
			subIncrI(at, bt, it)
		}
	case Int8:
		at := a.Int8s()
		bt := b.Int8s()
		it := incr.Int8s()

		switch {
		case as && bs:
			if !is {
				at[0] -= bt[0]
				return e.Add(t, incr, a)
			}
			it[0] += at[0] - bt[0]
			return nil
		case as && !bs:
			subIncrI8VS(bt, at[0], it)
		case bs && !as:
			subIncrI8VS(at, bt[0], it)
		default:
			subIncrI8(at, bt, it)
		}
	case Int16:
		at := a.Int16s()
		bt := b.Int16s()
		it := incr.Int16s()

		switch {
		case as && bs:
			if !is {
				at[0] -= bt[0]
				return e.Add(t, incr, a)
			}
			it[0] += at[0] - bt[0]
			return nil
		case as && !bs:
			subIncrI16VS(bt, at[0], it)
		case bs && !as:
			subIncrI16VS(at, bt[0], it)
		default:
			subIncrI16(at, bt, it)
		}
	case Int32:
		at := a.Int32s()
		bt := b.Int32s()
		it := incr.Int32s()

		switch {
		case as && bs:
			if !is {
				at[0] -= bt[0]
				return e.Add(t, incr, a)
			}
			it[0] += at[0] - bt[0]
			return nil
		case as && !bs:
			subIncrI32VS(bt, at[0], it)
		case bs && !as:
			subIncrI32VS(at, bt[0], it)
		default:
			subIncrI32(at, bt, it)
		}
	case Int64:
		at := a.Int64s()
		bt := b.Int64s()
		it := incr.Int64s()

		switch {
		case as && bs:
			if !is {
				at[0] -= bt[0]
				return e.Add(t, incr, a)
			}
			it[0] += at[0] - bt[0]
			return nil
		case as && !bs:
			subIncrI64VS(bt, at[0], it)
		case bs && !as:
			subIncrI64VS(at, bt[0], it)
		default:
			subIncrI64(at, bt, it)
		}
	case Uint:
		at := a.Uints()
		bt := b.Uints()
		it := incr.Uints()

		switch {
		case as && bs:
			if !is {
				at[0] -= bt[0]
				return e.Add(t, incr, a)
			}
			it[0] += at[0] - bt[0]
			return nil
		case as && !bs:
			subIncrUVS(bt, at[0], it)
		case bs && !as:
			subIncrUVS(at, bt[0], it)
		default:
			subIncrU(at, bt, it)
		}
	case Uint8:
		at := a.Uint8s()
		bt := b.Uint8s()
		it := incr.Uint8s()

		switch {
		case as && bs:
			if !is {
				at[0] -= bt[0]
				return e.Add(t, incr, a)
			}
			it[0] += at[0] - bt[0]
			return nil
		case as && !bs:
			subIncrU8VS(bt, at[0], it)
		case bs && !as:
			subIncrU8VS(at, bt[0], it)
		default:
			subIncrU8(at, bt, it)
		}
	case Uint16:
		at := a.Uint16s()
		bt := b.Uint16s()
		it := incr.Uint16s()

		switch {
		case as && bs:
			if !is {
				at[0] -= bt[0]
				return e.Add(t, incr, a)
			}
			it[0] += at[0] - bt[0]
			return nil
		case as && !bs:
			subIncrU16VS(bt, at[0], it)
		case bs && !as:
			subIncrU16VS(at, bt[0], it)
		default:
			subIncrU16(at, bt, it)
		}
	case Uint32:
		at := a.Uint32s()
		bt := b.Uint32s()
		it := incr.Uint32s()

		switch {
		case as && bs:
			if !is {
				at[0] -= bt[0]
				return e.Add(t, incr, a)
			}
			it[0] += at[0] - bt[0]
			return nil
		case as && !bs:
			subIncrU32VS(bt, at[0], it)
		case bs && !as:
			subIncrU32VS(at, bt[0], it)
		default:
			subIncrU32(at, bt, it)
		}
	case Uint64:
		at := a.Uint64s()
		bt := b.Uint64s()
		it := incr.Uint64s()

		switch {
		case as && bs:
			if !is {
				at[0] -= bt[0]
				return e.Add(t, incr, a)
			}
			it[0] += at[0] - bt[0]
			return nil
		case as && !bs:
			subIncrU64VS(bt, at[0], it)
		case bs && !as:
			subIncrU64VS(at, bt[0], it)
		default:
			subIncrU64(at, bt, it)
		}
	case Float32:
		at := a.Float32s()
		bt := b.Float32s()
		it := incr.Float32s()

		switch {
		case as && bs:
			if !is {
				at[0] -= bt[0]
				return e.Add(t, incr, a)
			}
			it[0] += at[0] - bt[0]
			return nil
		case as && !bs:
			subIncrF32VS(bt, at[0], it)
		case bs && !as:
			subIncrF32VS(at, bt[0], it)
		default:
			subIncrF32(at, bt, it)
		}
	case Float64:
		at := a.Float64s()
		bt := b.Float64s()
		it := incr.Float64s()

		switch {
		case as && bs:
			if !is {
				at[0] -= bt[0]
				return e.Add(t, incr, a)
			}
			it[0] += at[0] - bt[0]
			return nil
		case as && !bs:
			subIncrF64VS(bt, at[0], it)
		case bs && !as:
			subIncrF64VS(at, bt[0], it)
		default:
			subIncrF64(at, bt, it)
		}
	case Complex64:
		at := a.Complex64s()
		bt := b.Complex64s()
		it := incr.Complex64s()

		switch {
		case as && bs:
			if !is {
				at[0] -= bt[0]
				return e.Add(t, incr, a)
			}
			it[0] += at[0] - bt[0]
			return nil
		case as && !bs:
			subIncrC64VS(bt, at[0], it)
		case bs && !as:
			subIncrC64VS(at, bt[0], it)
		default:
			subIncrC64(at, bt, it)
		}
	case Complex128:
		at := a.Complex128s()
		bt := b.Complex128s()
		it := incr.Complex128s()

		switch {
		case as && bs:
			if !is {
				at[0] -= bt[0]
				return e.Add(t, incr, a)
			}
			it[0] += at[0] - bt[0]
			return nil
		case as && !bs:
			subIncrC128VS(bt, at[0], it)
		case bs && !as:
			subIncrC128VS(at, bt[0], it)
		default:
			subIncrC128(at, bt, it)
		}
	default:
		return errors.Errorf("NYI")
	}
	return nil
}

// SubIterIncr performs incr += a - b, guided by iterators
func (e E) SubIterIncr(t reflect.Type, a, b, incr Array, ait, bit, iit Iterator) error {
	as := isScalar(a)
	bs := isScalar(b)
	is := isScalar(incr)

	if (as && !bs) || (bs && !as) && is {
		return errors.Errorf("Cannot increment on a scalar increment")
	}

	switch t {
	case Int:
		at := a.Ints()
		bt := b.Ints()
		it := incr.Ints()

		switch {
		case as && bs:
			if !is {
				at[0] -= bt[0]
				return e.SubIter(t, incr, a, iit, ait)
			}
		case as && !bs:
			return subIncrIIterVS(bt, at[0], it, bit, iit)
		case bs && !as:
			return subIncrIIterVS(at, bt[0], it, ait, iit)
		default:
			return subIncrIIter(at, bt, it, ait, bit, iit)
		}
	case Int8:
		at := a.Int8s()
		bt := b.Int8s()
		it := incr.Int8s()

		switch {
		case as && bs:
			if !is {
				at[0] -= bt[0]
				return e.SubIter(t, incr, a, iit, ait)
			}
		case as && !bs:
			return subIncrI8IterVS(bt, at[0], it, bit, iit)
		case bs && !as:
			return subIncrI8IterVS(at, bt[0], it, ait, iit)
		default:
			return subIncrI8Iter(at, bt, it, ait, bit, iit)
		}
	case Int16:
		at := a.Int16s()
		bt := b.Int16s()
		it := incr.Int16s()

		switch {
		case as && bs:
			if !is {
				at[0] -= bt[0]
				return e.SubIter(t, incr, a, iit, ait)
			}
		case as && !bs:
			return subIncrI16IterVS(bt, at[0], it, bit, iit)
		case bs && !as:
			return subIncrI16IterVS(at, bt[0], it, ait, iit)
		default:
			return subIncrI16Iter(at, bt, it, ait, bit, iit)
		}
	case Int32:
		at := a.Int32s()
		bt := b.Int32s()
		it := incr.Int32s()

		switch {
		case as && bs:
			if !is {
				at[0] -= bt[0]
				return e.SubIter(t, incr, a, iit, ait)
			}
		case as && !bs:
			return subIncrI32IterVS(bt, at[0], it, bit, iit)
		case bs && !as:
			return subIncrI32IterVS(at, bt[0], it, ait, iit)
		default:
			return subIncrI32Iter(at, bt, it, ait, bit, iit)
		}
	case Int64:
		at := a.Int64s()
		bt := b.Int64s()
		it := incr.Int64s()

		switch {
		case as && bs:
			if !is {
				at[0] -= bt[0]
				return e.SubIter(t, incr, a, iit, ait)
			}
		case as && !bs:
			return subIncrI64IterVS(bt, at[0], it, bit, iit)
		case bs && !as:
			return subIncrI64IterVS(at, bt[0], it, ait, iit)
		default:
			return subIncrI64Iter(at, bt, it, ait, bit, iit)
		}
	case Uint:
		at := a.Uints()
		bt := b.Uints()
		it := incr.Uints()

		switch {
		case as && bs:
			if !is {
				at[0] -= bt[0]
				return e.SubIter(t, incr, a, iit, ait)
			}
		case as && !bs:
			return subIncrUIterVS(bt, at[0], it, bit, iit)
		case bs && !as:
			return subIncrUIterVS(at, bt[0], it, ait, iit)
		default:
			return subIncrUIter(at, bt, it, ait, bit, iit)
		}
	case Uint8:
		at := a.Uint8s()
		bt := b.Uint8s()
		it := incr.Uint8s()

		switch {
		case as && bs:
			if !is {
				at[0] -= bt[0]
				return e.SubIter(t, incr, a, iit, ait)
			}
		case as && !bs:
			return subIncrU8IterVS(bt, at[0], it, bit, iit)
		case bs && !as:
			return subIncrU8IterVS(at, bt[0], it, ait, iit)
		default:
			return subIncrU8Iter(at, bt, it, ait, bit, iit)
		}
	case Uint16:
		at := a.Uint16s()
		bt := b.Uint16s()
		it := incr.Uint16s()

		switch {
		case as && bs:
			if !is {
				at[0] -= bt[0]
				return e.SubIter(t, incr, a, iit, ait)
			}
		case as && !bs:
			return subIncrU16IterVS(bt, at[0], it, bit, iit)
		case bs && !as:
			return subIncrU16IterVS(at, bt[0], it, ait, iit)
		default:
			return subIncrU16Iter(at, bt, it, ait, bit, iit)
		}
	case Uint32:
		at := a.Uint32s()
		bt := b.Uint32s()
		it := incr.Uint32s()

		switch {
		case as && bs:
			if !is {
				at[0] -= bt[0]
				return e.SubIter(t, incr, a, iit, ait)
			}
		case as && !bs:
			return subIncrU32IterVS(bt, at[0], it, bit, iit)
		case bs && !as:
			return subIncrU32IterVS(at, bt[0], it, ait, iit)
		default:
			return subIncrU32Iter(at, bt, it, ait, bit, iit)
		}
	case Uint64:
		at := a.Uint64s()
		bt := b.Uint64s()
		it := incr.Uint64s()

		switch {
		case as && bs:
			if !is {
				at[0] -= bt[0]
				return e.SubIter(t, incr, a, iit, ait)
			}
		case as && !bs:
			return subIncrU64IterVS(bt, at[0], it, bit, iit)
		case bs && !as:
			return subIncrU64IterVS(at, bt[0], it, ait, iit)
		default:
			return subIncrU64Iter(at, bt, it, ait, bit, iit)
		}
	case Float32:
		at := a.Float32s()
		bt := b.Float32s()
		it := incr.Float32s()

		switch {
		case as && bs:
			if !is {
				at[0] -= bt[0]
				return e.SubIter(t, incr, a, iit, ait)
			}
		case as && !bs:
			return subIncrF32IterVS(bt, at[0], it, bit, iit)
		case bs && !as:
			return subIncrF32IterVS(at, bt[0], it, ait, iit)
		default:
			return subIncrF32Iter(at, bt, it, ait, bit, iit)
		}
	case Float64:
		at := a.Float64s()
		bt := b.Float64s()
		it := incr.Float64s()

		switch {
		case as && bs:
			if !is {
				at[0] -= bt[0]
				return e.SubIter(t, incr, a, iit, ait)
			}
		case as && !bs:
			return subIncrF64IterVS(bt, at[0], it, bit, iit)
		case bs && !as:
			return subIncrF64IterVS(at, bt[0], it, ait, iit)
		default:
			return subIncrF64Iter(at, bt, it, ait, bit, iit)
		}
	case Complex64:
		at := a.Complex64s()
		bt := b.Complex64s()
		it := incr.Complex64s()

		switch {
		case as && bs:
			if !is {
				at[0] -= bt[0]
				return e.SubIter(t, incr, a, iit, ait)
			}
		case as && !bs:
			return subIncrC64IterVS(bt, at[0], it, bit, iit)
		case bs && !as:
			return subIncrC64IterVS(at, bt[0], it, ait, iit)
		default:
			return subIncrC64Iter(at, bt, it, ait, bit, iit)
		}
	case Complex128:
		at := a.Complex128s()
		bt := b.Complex128s()
		it := incr.Complex128s()

		switch {
		case as && bs:
			if !is {
				at[0] -= bt[0]
				return e.SubIter(t, incr, a, iit, ait)
			}
		case as && !bs:
			return subIncrC128IterVS(bt, at[0], it, bit, iit)
		case bs && !as:
			return subIncrC128IterVS(at, bt[0], it, ait, iit)
		default:
			return subIncrC128Iter(at, bt, it, ait, bit, iit)
		}
	default:
		return errors.Errorf("NYI")
	}
	return nil
}

func subI(a, b []int) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range b {
		a[i] -= v
	}
}

func subIVS(a []int, b int) {
	a = a[:]
	for i := range a {
		a[i] -= b
	}
}

func subISV(a int, b []int) {
	for i := range b {
		b[i] = a - b[i]
	}
}

func subIIter(a, b []int, ait, bit Iterator) (err error) {
	var i, j int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = bit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		a[i] -= b[j]
	}
	return
}

func subIIterVS(a []int, b int, it Iterator) (err error) {
	var i int
	for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
		a[i] -= b
	}
	err = handleNoOp(err)
	return
}

func subIIterSV(a int, b []int, it Iterator) (err error) {
	var i int
	for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
		b[i] = a - b[i]
	}
	err = handleNoOp(err)
	return
}

func subIncrI(a, b, incr []int) {
	a = a[:]
	b = b[:len(a)]
	incr = incr[:len(a)]
	for i, v := range a {
		incr[i] += v - b[i]
	}
}

func subIncrIVS(a []int, b int, incr []int) {
	a = a[:]
	incr = incr[:len(a)]
	for i := range a {
		incr[i] += a[i] - b
	}
}

func subIncrISV(a int, b []int, incr []int) {
	b = b[:]
	incr = incr[:len(b)]
	for i := range b {
		incr[i] += a - b[i]
	}
}

func subIncrIIterVS(a []int, b int, incr []int, ait, iit Iterator) (err error) {
	var i, j int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = iit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		incr[j] += a[i] - b
	}
	return nil
}

func subIncrIIterSV(a int, b []int, incr []int, ait, iit Iterator) (err error) {
	var i, j int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = iit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		incr[j] += a - b[i]
	}
	return nil
}

func subIncrIIter(a, b, incr []int, ait, bit, iit Iterator) (err error) {
	var i, j, k int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = bit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if k, _, err = iit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		incr[k] += a[i] - b[j]
	}
	return nil
}

func subI8(a, b []int8) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range b {
		a[i] -= v
	}
}

func subI8VS(a []int8, b int8) {
	a = a[:]
	for i := range a {
		a[i] -= b
	}
}

func subI8SV(a int8, b []int8) {
	for i := range b {
		b[i] = a - b[i]
	}
}

func subI8Iter(a, b []int8, ait, bit Iterator) (err error) {
	var i, j int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = bit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		a[i] -= b[j]
	}
	return
}

func subI8IterVS(a []int8, b int8, it Iterator) (err error) {
	var i int
	for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
		a[i] -= b
	}
	err = handleNoOp(err)
	return
}

func subI8IterSV(a int8, b []int8, it Iterator) (err error) {
	var i int
	for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
		b[i] = a - b[i]
	}
	err = handleNoOp(err)
	return
}

func subIncrI8(a, b, incr []int8) {
	a = a[:]
	b = b[:len(a)]
	incr = incr[:len(a)]
	for i, v := range a {
		incr[i] += v - b[i]
	}
}

func subIncrI8VS(a []int8, b int8, incr []int8) {
	a = a[:]
	incr = incr[:len(a)]
	for i := range a {
		incr[i] += a[i] - b
	}
}

func subIncrI8SV(a int8, b []int8, incr []int8) {
	b = b[:]
	incr = incr[:len(b)]
	for i := range b {
		incr[i] += a - b[i]
	}
}

func subIncrI8IterVS(a []int8, b int8, incr []int8, ait, iit Iterator) (err error) {
	var i, j int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = iit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		incr[j] += a[i] - b
	}
	return nil
}

func subIncrI8IterSV(a int8, b []int8, incr []int8, ait, iit Iterator) (err error) {
	var i, j int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = iit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		incr[j] += a - b[i]
	}
	return nil
}

func subIncrI8Iter(a, b, incr []int8, ait, bit, iit Iterator) (err error) {
	var i, j, k int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = bit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if k, _, err = iit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		incr[k] += a[i] - b[j]
	}
	return nil
}

func subI16(a, b []int16) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range b {
		a[i] -= v
	}
}

func subI16VS(a []int16, b int16) {
	a = a[:]
	for i := range a {
		a[i] -= b
	}
}

func subI16SV(a int16, b []int16) {
	for i := range b {
		b[i] = a - b[i]
	}
}

func subI16Iter(a, b []int16, ait, bit Iterator) (err error) {
	var i, j int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = bit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		a[i] -= b[j]
	}
	return
}

func subI16IterVS(a []int16, b int16, it Iterator) (err error) {
	var i int
	for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
		a[i] -= b
	}
	err = handleNoOp(err)
	return
}

func subI16IterSV(a int16, b []int16, it Iterator) (err error) {
	var i int
	for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
		b[i] = a - b[i]
	}
	err = handleNoOp(err)
	return
}

func subIncrI16(a, b, incr []int16) {
	a = a[:]
	b = b[:len(a)]
	incr = incr[:len(a)]
	for i, v := range a {
		incr[i] += v - b[i]
	}
}

func subIncrI16VS(a []int16, b int16, incr []int16) {
	a = a[:]
	incr = incr[:len(a)]
	for i := range a {
		incr[i] += a[i] - b
	}
}

func subIncrI16SV(a int16, b []int16, incr []int16) {
	b = b[:]
	incr = incr[:len(b)]
	for i := range b {
		incr[i] += a - b[i]
	}
}

func subIncrI16IterVS(a []int16, b int16, incr []int16, ait, iit Iterator) (err error) {
	var i, j int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = iit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		incr[j] += a[i] - b
	}
	return nil
}

func subIncrI16IterSV(a int16, b []int16, incr []int16, ait, iit Iterator) (err error) {
	var i, j int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = iit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		incr[j] += a - b[i]
	}
	return nil
}

func subIncrI16Iter(a, b, incr []int16, ait, bit, iit Iterator) (err error) {
	var i, j, k int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = bit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if k, _, err = iit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		incr[k] += a[i] - b[j]
	}
	return nil
}

func subI32(a, b []int32) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range b {
		a[i] -= v
	}
}

func subI32VS(a []int32, b int32) {
	a = a[:]
	for i := range a {
		a[i] -= b
	}
}

func subI32SV(a int32, b []int32) {
	for i := range b {
		b[i] = a - b[i]
	}
}

func subI32Iter(a, b []int32, ait, bit Iterator) (err error) {
	var i, j int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = bit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		a[i] -= b[j]
	}
	return
}

func subI32IterVS(a []int32, b int32, it Iterator) (err error) {
	var i int
	for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
		a[i] -= b
	}
	err = handleNoOp(err)
	return
}

func subI32IterSV(a int32, b []int32, it Iterator) (err error) {
	var i int
	for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
		b[i] = a - b[i]
	}
	err = handleNoOp(err)
	return
}

func subIncrI32(a, b, incr []int32) {
	a = a[:]
	b = b[:len(a)]
	incr = incr[:len(a)]
	for i, v := range a {
		incr[i] += v - b[i]
	}
}

func subIncrI32VS(a []int32, b int32, incr []int32) {
	a = a[:]
	incr = incr[:len(a)]
	for i := range a {
		incr[i] += a[i] - b
	}
}

func subIncrI32SV(a int32, b []int32, incr []int32) {
	b = b[:]
	incr = incr[:len(b)]
	for i := range b {
		incr[i] += a - b[i]
	}
}

func subIncrI32IterVS(a []int32, b int32, incr []int32, ait, iit Iterator) (err error) {
	var i, j int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = iit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		incr[j] += a[i] - b
	}
	return nil
}

func subIncrI32IterSV(a int32, b []int32, incr []int32, ait, iit Iterator) (err error) {
	var i, j int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = iit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		incr[j] += a - b[i]
	}
	return nil
}

func subIncrI32Iter(a, b, incr []int32, ait, bit, iit Iterator) (err error) {
	var i, j, k int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = bit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if k, _, err = iit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		incr[k] += a[i] - b[j]
	}
	return nil
}

func subI64(a, b []int64) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range b {
		a[i] -= v
	}
}

func subI64VS(a []int64, b int64) {
	a = a[:]
	for i := range a {
		a[i] -= b
	}
}

func subI64SV(a int64, b []int64) {
	for i := range b {
		b[i] = a - b[i]
	}
}

func subI64Iter(a, b []int64, ait, bit Iterator) (err error) {
	var i, j int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = bit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		a[i] -= b[j]
	}
	return
}

func subI64IterVS(a []int64, b int64, it Iterator) (err error) {
	var i int
	for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
		a[i] -= b
	}
	err = handleNoOp(err)
	return
}

func subI64IterSV(a int64, b []int64, it Iterator) (err error) {
	var i int
	for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
		b[i] = a - b[i]
	}
	err = handleNoOp(err)
	return
}

func subIncrI64(a, b, incr []int64) {
	a = a[:]
	b = b[:len(a)]
	incr = incr[:len(a)]
	for i, v := range a {
		incr[i] += v - b[i]
	}
}

func subIncrI64VS(a []int64, b int64, incr []int64) {
	a = a[:]
	incr = incr[:len(a)]
	for i := range a {
		incr[i] += a[i] - b
	}
}

func subIncrI64SV(a int64, b []int64, incr []int64) {
	b = b[:]
	incr = incr[:len(b)]
	for i := range b {
		incr[i] += a - b[i]
	}
}

func subIncrI64IterVS(a []int64, b int64, incr []int64, ait, iit Iterator) (err error) {
	var i, j int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = iit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		incr[j] += a[i] - b
	}
	return nil
}

func subIncrI64IterSV(a int64, b []int64, incr []int64, ait, iit Iterator) (err error) {
	var i, j int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = iit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		incr[j] += a - b[i]
	}
	return nil
}

func subIncrI64Iter(a, b, incr []int64, ait, bit, iit Iterator) (err error) {
	var i, j, k int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = bit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if k, _, err = iit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		incr[k] += a[i] - b[j]
	}
	return nil
}

func subU(a, b []uint) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range b {
		a[i] -= v
	}
}

func subUVS(a []uint, b uint) {
	a = a[:]
	for i := range a {
		a[i] -= b
	}
}

func subUSV(a uint, b []uint) {
	for i := range b {
		b[i] = a - b[i]
	}
}

func subUIter(a, b []uint, ait, bit Iterator) (err error) {
	var i, j int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = bit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		a[i] -= b[j]
	}
	return
}

func subUIterVS(a []uint, b uint, it Iterator) (err error) {
	var i int
	for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
		a[i] -= b
	}
	err = handleNoOp(err)
	return
}

func subUIterSV(a uint, b []uint, it Iterator) (err error) {
	var i int
	for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
		b[i] = a - b[i]
	}
	err = handleNoOp(err)
	return
}

func subIncrU(a, b, incr []uint) {
	a = a[:]
	b = b[:len(a)]
	incr = incr[:len(a)]
	for i, v := range a {
		incr[i] += v - b[i]
	}
}

func subIncrUVS(a []uint, b uint, incr []uint) {
	a = a[:]
	incr = incr[:len(a)]
	for i := range a {
		incr[i] += a[i] - b
	}
}

func subIncrUSV(a uint, b []uint, incr []uint) {
	b = b[:]
	incr = incr[:len(b)]
	for i := range b {
		incr[i] += a - b[i]
	}
}

func subIncrUIterVS(a []uint, b uint, incr []uint, ait, iit Iterator) (err error) {
	var i, j int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = iit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		incr[j] += a[i] - b
	}
	return nil
}

func subIncrUIterSV(a uint, b []uint, incr []uint, ait, iit Iterator) (err error) {
	var i, j int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = iit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		incr[j] += a - b[i]
	}
	return nil
}

func subIncrUIter(a, b, incr []uint, ait, bit, iit Iterator) (err error) {
	var i, j, k int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = bit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if k, _, err = iit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		incr[k] += a[i] - b[j]
	}
	return nil
}

func subU8(a, b []uint8) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range b {
		a[i] -= v
	}
}

func subU8VS(a []uint8, b uint8) {
	a = a[:]
	for i := range a {
		a[i] -= b
	}
}

func subU8SV(a uint8, b []uint8) {
	for i := range b {
		b[i] = a - b[i]
	}
}

func subU8Iter(a, b []uint8, ait, bit Iterator) (err error) {
	var i, j int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = bit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		a[i] -= b[j]
	}
	return
}

func subU8IterVS(a []uint8, b uint8, it Iterator) (err error) {
	var i int
	for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
		a[i] -= b
	}
	err = handleNoOp(err)
	return
}

func subU8IterSV(a uint8, b []uint8, it Iterator) (err error) {
	var i int
	for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
		b[i] = a - b[i]
	}
	err = handleNoOp(err)
	return
}

func subIncrU8(a, b, incr []uint8) {
	a = a[:]
	b = b[:len(a)]
	incr = incr[:len(a)]
	for i, v := range a {
		incr[i] += v - b[i]
	}
}

func subIncrU8VS(a []uint8, b uint8, incr []uint8) {
	a = a[:]
	incr = incr[:len(a)]
	for i := range a {
		incr[i] += a[i] - b
	}
}

func subIncrU8SV(a uint8, b []uint8, incr []uint8) {
	b = b[:]
	incr = incr[:len(b)]
	for i := range b {
		incr[i] += a - b[i]
	}
}

func subIncrU8IterVS(a []uint8, b uint8, incr []uint8, ait, iit Iterator) (err error) {
	var i, j int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = iit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		incr[j] += a[i] - b
	}
	return nil
}

func subIncrU8IterSV(a uint8, b []uint8, incr []uint8, ait, iit Iterator) (err error) {
	var i, j int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = iit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		incr[j] += a - b[i]
	}
	return nil
}

func subIncrU8Iter(a, b, incr []uint8, ait, bit, iit Iterator) (err error) {
	var i, j, k int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = bit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if k, _, err = iit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		incr[k] += a[i] - b[j]
	}
	return nil
}

func subU16(a, b []uint16) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range b {
		a[i] -= v
	}
}

func subU16VS(a []uint16, b uint16) {
	a = a[:]
	for i := range a {
		a[i] -= b
	}
}

func subU16SV(a uint16, b []uint16) {
	for i := range b {
		b[i] = a - b[i]
	}
}

func subU16Iter(a, b []uint16, ait, bit Iterator) (err error) {
	var i, j int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = bit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		a[i] -= b[j]
	}
	return
}

func subU16IterVS(a []uint16, b uint16, it Iterator) (err error) {
	var i int
	for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
		a[i] -= b
	}
	err = handleNoOp(err)
	return
}

func subU16IterSV(a uint16, b []uint16, it Iterator) (err error) {
	var i int
	for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
		b[i] = a - b[i]
	}
	err = handleNoOp(err)
	return
}

func subIncrU16(a, b, incr []uint16) {
	a = a[:]
	b = b[:len(a)]
	incr = incr[:len(a)]
	for i, v := range a {
		incr[i] += v - b[i]
	}
}

func subIncrU16VS(a []uint16, b uint16, incr []uint16) {
	a = a[:]
	incr = incr[:len(a)]
	for i := range a {
		incr[i] += a[i] - b
	}
}

func subIncrU16SV(a uint16, b []uint16, incr []uint16) {
	b = b[:]
	incr = incr[:len(b)]
	for i := range b {
		incr[i] += a - b[i]
	}
}

func subIncrU16IterVS(a []uint16, b uint16, incr []uint16, ait, iit Iterator) (err error) {
	var i, j int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = iit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		incr[j] += a[i] - b
	}
	return nil
}

func subIncrU16IterSV(a uint16, b []uint16, incr []uint16, ait, iit Iterator) (err error) {
	var i, j int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = iit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		incr[j] += a - b[i]
	}
	return nil
}

func subIncrU16Iter(a, b, incr []uint16, ait, bit, iit Iterator) (err error) {
	var i, j, k int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = bit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if k, _, err = iit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		incr[k] += a[i] - b[j]
	}
	return nil
}

func subU32(a, b []uint32) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range b {
		a[i] -= v
	}
}

func subU32VS(a []uint32, b uint32) {
	a = a[:]
	for i := range a {
		a[i] -= b
	}
}

func subU32SV(a uint32, b []uint32) {
	for i := range b {
		b[i] = a - b[i]
	}
}

func subU32Iter(a, b []uint32, ait, bit Iterator) (err error) {
	var i, j int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = bit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		a[i] -= b[j]
	}
	return
}

func subU32IterVS(a []uint32, b uint32, it Iterator) (err error) {
	var i int
	for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
		a[i] -= b
	}
	err = handleNoOp(err)
	return
}

func subU32IterSV(a uint32, b []uint32, it Iterator) (err error) {
	var i int
	for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
		b[i] = a - b[i]
	}
	err = handleNoOp(err)
	return
}

func subIncrU32(a, b, incr []uint32) {
	a = a[:]
	b = b[:len(a)]
	incr = incr[:len(a)]
	for i, v := range a {
		incr[i] += v - b[i]
	}
}

func subIncrU32VS(a []uint32, b uint32, incr []uint32) {
	a = a[:]
	incr = incr[:len(a)]
	for i := range a {
		incr[i] += a[i] - b
	}
}

func subIncrU32SV(a uint32, b []uint32, incr []uint32) {
	b = b[:]
	incr = incr[:len(b)]
	for i := range b {
		incr[i] += a - b[i]
	}
}

func subIncrU32IterVS(a []uint32, b uint32, incr []uint32, ait, iit Iterator) (err error) {
	var i, j int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = iit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		incr[j] += a[i] - b
	}
	return nil
}

func subIncrU32IterSV(a uint32, b []uint32, incr []uint32, ait, iit Iterator) (err error) {
	var i, j int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = iit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		incr[j] += a - b[i]
	}
	return nil
}

func subIncrU32Iter(a, b, incr []uint32, ait, bit, iit Iterator) (err error) {
	var i, j, k int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = bit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if k, _, err = iit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		incr[k] += a[i] - b[j]
	}
	return nil
}

func subU64(a, b []uint64) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range b {
		a[i] -= v
	}
}

func subU64VS(a []uint64, b uint64) {
	a = a[:]
	for i := range a {
		a[i] -= b
	}
}

func subU64SV(a uint64, b []uint64) {
	for i := range b {
		b[i] = a - b[i]
	}
}

func subU64Iter(a, b []uint64, ait, bit Iterator) (err error) {
	var i, j int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = bit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		a[i] -= b[j]
	}
	return
}

func subU64IterVS(a []uint64, b uint64, it Iterator) (err error) {
	var i int
	for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
		a[i] -= b
	}
	err = handleNoOp(err)
	return
}

func subU64IterSV(a uint64, b []uint64, it Iterator) (err error) {
	var i int
	for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
		b[i] = a - b[i]
	}
	err = handleNoOp(err)
	return
}

func subIncrU64(a, b, incr []uint64) {
	a = a[:]
	b = b[:len(a)]
	incr = incr[:len(a)]
	for i, v := range a {
		incr[i] += v - b[i]
	}
}

func subIncrU64VS(a []uint64, b uint64, incr []uint64) {
	a = a[:]
	incr = incr[:len(a)]
	for i := range a {
		incr[i] += a[i] - b
	}
}

func subIncrU64SV(a uint64, b []uint64, incr []uint64) {
	b = b[:]
	incr = incr[:len(b)]
	for i := range b {
		incr[i] += a - b[i]
	}
}

func subIncrU64IterVS(a []uint64, b uint64, incr []uint64, ait, iit Iterator) (err error) {
	var i, j int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = iit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		incr[j] += a[i] - b
	}
	return nil
}

func subIncrU64IterSV(a uint64, b []uint64, incr []uint64, ait, iit Iterator) (err error) {
	var i, j int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = iit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		incr[j] += a - b[i]
	}
	return nil
}

func subIncrU64Iter(a, b, incr []uint64, ait, bit, iit Iterator) (err error) {
	var i, j, k int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = bit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if k, _, err = iit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		incr[k] += a[i] - b[j]
	}
	return nil
}

func subF32(a, b []float32) {
	vecf32.Sub(a, b)
}

func subF32VS(a []float32, b float32) {
	vecf32.TransInv(a, b)
}

func subF32SV(a float32, b []float32) {
	vecf32.TransInvR(b, a)
}

func subF32Iter(a, b []float32, ait, bit Iterator) (err error) {
	var i, j int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = bit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		a[i] -= b[j]
	}
	return
}

func subF32IterVS(a []float32, b float32, it Iterator) (err error) {
	var i int
	for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
		a[i] -= b
	}
	err = handleNoOp(err)
	return
}

func subF32IterSV(a float32, b []float32, it Iterator) (err error) {
	var i int
	for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
		b[i] = a - b[i]
	}
	err = handleNoOp(err)
	return
}

func subIncrF32(a, b, incr []float32) {
	vecf32.IncrSub(a, b, incr)
}

func subIncrF32VS(a []float32, b float32, incr []float32) {
	vecf32.IncrTransInv(a, b, incr)
}

func subIncrF32SV(a float32, b []float32, incr []float32) {
	vecf32.IncrTransInvR(b, a, incr)
}

func subIncrF32IterVS(a []float32, b float32, incr []float32, ait, iit Iterator) (err error) {
	var i, j int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = iit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		incr[j] += a[i] - b
	}
	return nil
}

func subIncrF32IterSV(a float32, b []float32, incr []float32, ait, iit Iterator) (err error) {
	var i, j int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = iit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		incr[j] += a - b[i]
	}
	return nil
}

func subIncrF32Iter(a, b, incr []float32, ait, bit, iit Iterator) (err error) {
	var i, j, k int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = bit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if k, _, err = iit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		incr[k] += a[i] - b[j]
	}
	return nil
}

func subF64(a, b []float64) {
	vecf64.Sub(a, b)
}

func subF64VS(a []float64, b float64) {
	vecf64.TransInv(a, b)
}

func subF64SV(a float64, b []float64) {
	vecf64.TransInvR(b, a)
}

func subF64Iter(a, b []float64, ait, bit Iterator) (err error) {
	var i, j int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = bit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		a[i] -= b[j]
	}
	return
}

func subF64IterVS(a []float64, b float64, it Iterator) (err error) {
	var i int
	for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
		a[i] -= b
	}
	err = handleNoOp(err)
	return
}

func subF64IterSV(a float64, b []float64, it Iterator) (err error) {
	var i int
	for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
		b[i] = a - b[i]
	}
	err = handleNoOp(err)
	return
}

func subIncrF64(a, b, incr []float64) {
	vecf64.IncrSub(a, b, incr)
}

func subIncrF64VS(a []float64, b float64, incr []float64) {
	vecf64.IncrTransInv(a, b, incr)
}

func subIncrF64SV(a float64, b []float64, incr []float64) {
	vecf64.IncrTransInvR(b, a, incr)
}

func subIncrF64IterVS(a []float64, b float64, incr []float64, ait, iit Iterator) (err error) {
	var i, j int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = iit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		incr[j] += a[i] - b
	}
	return nil
}

func subIncrF64IterSV(a float64, b []float64, incr []float64, ait, iit Iterator) (err error) {
	var i, j int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = iit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		incr[j] += a - b[i]
	}
	return nil
}

func subIncrF64Iter(a, b, incr []float64, ait, bit, iit Iterator) (err error) {
	var i, j, k int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = bit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if k, _, err = iit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		incr[k] += a[i] - b[j]
	}
	return nil
}

func subC64(a, b []complex64) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range b {
		a[i] -= v
	}
}

func subC64VS(a []complex64, b complex64) {
	a = a[:]
	for i := range a {
		a[i] -= b
	}
}

func subC64SV(a complex64, b []complex64) {
	for i := range b {
		b[i] = a - b[i]
	}
}

func subC64Iter(a, b []complex64, ait, bit Iterator) (err error) {
	var i, j int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = bit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		a[i] -= b[j]
	}
	return
}

func subC64IterVS(a []complex64, b complex64, it Iterator) (err error) {
	var i int
	for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
		a[i] -= b
	}
	err = handleNoOp(err)
	return
}

func subC64IterSV(a complex64, b []complex64, it Iterator) (err error) {
	var i int
	for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
		b[i] = a - b[i]
	}
	err = handleNoOp(err)
	return
}

func subIncrC64(a, b, incr []complex64) {
	a = a[:]
	b = b[:len(a)]
	incr = incr[:len(a)]
	for i, v := range a {
		incr[i] += v - b[i]
	}
}

func subIncrC64VS(a []complex64, b complex64, incr []complex64) {
	a = a[:]
	incr = incr[:len(a)]
	for i := range a {
		incr[i] += a[i] - b
	}
}

func subIncrC64SV(a complex64, b []complex64, incr []complex64) {
	b = b[:]
	incr = incr[:len(b)]
	for i := range b {
		incr[i] += a - b[i]
	}
}

func subIncrC64IterVS(a []complex64, b complex64, incr []complex64, ait, iit Iterator) (err error) {
	var i, j int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = iit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		incr[j] += a[i] - b
	}
	return nil
}

func subIncrC64IterSV(a complex64, b []complex64, incr []complex64, ait, iit Iterator) (err error) {
	var i, j int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = iit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		incr[j] += a - b[i]
	}
	return nil
}

func subIncrC64Iter(a, b, incr []complex64, ait, bit, iit Iterator) (err error) {
	var i, j, k int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = bit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if k, _, err = iit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		incr[k] += a[i] - b[j]
	}
	return nil
}

func subC128(a, b []complex128) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range b {
		a[i] -= v
	}
}

func subC128VS(a []complex128, b complex128) {
	a = a[:]
	for i := range a {
		a[i] -= b
	}
}

func subC128SV(a complex128, b []complex128) {
	for i := range b {
		b[i] = a - b[i]
	}
}

func subC128Iter(a, b []complex128, ait, bit Iterator) (err error) {
	var i, j int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = bit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		a[i] -= b[j]
	}
	return
}

func subC128IterVS(a []complex128, b complex128, it Iterator) (err error) {
	var i int
	for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
		a[i] -= b
	}
	err = handleNoOp(err)
	return
}

func subC128IterSV(a complex128, b []complex128, it Iterator) (err error) {
	var i int
	for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
		b[i] = a - b[i]
	}
	err = handleNoOp(err)
	return
}

func subIncrC128(a, b, incr []complex128) {
	a = a[:]
	b = b[:len(a)]
	incr = incr[:len(a)]
	for i, v := range a {
		incr[i] += v - b[i]
	}
}

func subIncrC128VS(a []complex128, b complex128, incr []complex128) {
	a = a[:]
	incr = incr[:len(a)]
	for i := range a {
		incr[i] += a[i] - b
	}
}

func subIncrC128SV(a complex128, b []complex128, incr []complex128) {
	b = b[:]
	incr = incr[:len(b)]
	for i := range b {
		incr[i] += a - b[i]
	}
}

func subIncrC128IterVS(a []complex128, b complex128, incr []complex128, ait, iit Iterator) (err error) {
	var i, j int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = iit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		incr[j] += a[i] - b
	}
	return nil
}

func subIncrC128IterSV(a complex128, b []complex128, incr []complex128, ait, iit Iterator) (err error) {
	var i, j int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = iit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		incr[j] += a - b[i]
	}
	return nil
}

func subIncrC128Iter(a, b, incr []complex128, ait, bit, iit Iterator) (err error) {
	var i, j, k int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = bit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if k, _, err = iit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		incr[k] += a[i] - b[j]
	}
	return nil
}
