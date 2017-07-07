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

// Add performs addition: a += b.
// If the element type are numbers, it will fall to using standard number addition.
// If the element type are strings, concatenation will happen instead.
// If the element type is unknown, it will attempt to look for the Adder interface, and use the Adder method.
// Failing that, it produces an error.
//
// a and b may be scalars. In the case where one of which is vector and the other is a scalar, it's the vector that will
// be clobbered. If both are scalars, a will be clobbered
func (e E) Add(t reflect.Type, a, b Array) (err error) {
	as := isScalar(a)
	bs := isScalar(b)
	switch t {
	case Int:
		at := a.Ints()
		bt := b.Ints()

		switch {
		case as && bs:
			at[0] += bt[0]
		case as && !bs:
			addIVS(bt, at[0])
		case bs && !as:
			addIVS(at, bt[0])
		default:
			addI(at, bt)
		}
		return nil
	case Int8:
		at := a.Int8s()
		bt := b.Int8s()

		switch {
		case as && bs:
			at[0] += bt[0]
		case as && !bs:
			addI8VS(bt, at[0])
		case bs && !as:
			addI8VS(at, bt[0])
		default:
			addI8(at, bt)
		}
		return nil
	case Int16:
		at := a.Int16s()
		bt := b.Int16s()

		switch {
		case as && bs:
			at[0] += bt[0]
		case as && !bs:
			addI16VS(bt, at[0])
		case bs && !as:
			addI16VS(at, bt[0])
		default:
			addI16(at, bt)
		}
		return nil
	case Int32:
		at := a.Int32s()
		bt := b.Int32s()

		switch {
		case as && bs:
			at[0] += bt[0]
		case as && !bs:
			addI32VS(bt, at[0])
		case bs && !as:
			addI32VS(at, bt[0])
		default:
			addI32(at, bt)
		}
		return nil
	case Int64:
		at := a.Int64s()
		bt := b.Int64s()

		switch {
		case as && bs:
			at[0] += bt[0]
		case as && !bs:
			addI64VS(bt, at[0])
		case bs && !as:
			addI64VS(at, bt[0])
		default:
			addI64(at, bt)
		}
		return nil
	case Uint:
		at := a.Uints()
		bt := b.Uints()

		switch {
		case as && bs:
			at[0] += bt[0]
		case as && !bs:
			addUVS(bt, at[0])
		case bs && !as:
			addUVS(at, bt[0])
		default:
			addU(at, bt)
		}
		return nil
	case Uint8:
		at := a.Uint8s()
		bt := b.Uint8s()

		switch {
		case as && bs:
			at[0] += bt[0]
		case as && !bs:
			addU8VS(bt, at[0])
		case bs && !as:
			addU8VS(at, bt[0])
		default:
			addU8(at, bt)
		}
		return nil
	case Uint16:
		at := a.Uint16s()
		bt := b.Uint16s()

		switch {
		case as && bs:
			at[0] += bt[0]
		case as && !bs:
			addU16VS(bt, at[0])
		case bs && !as:
			addU16VS(at, bt[0])
		default:
			addU16(at, bt)
		}
		return nil
	case Uint32:
		at := a.Uint32s()
		bt := b.Uint32s()

		switch {
		case as && bs:
			at[0] += bt[0]
		case as && !bs:
			addU32VS(bt, at[0])
		case bs && !as:
			addU32VS(at, bt[0])
		default:
			addU32(at, bt)
		}
		return nil
	case Uint64:
		at := a.Uint64s()
		bt := b.Uint64s()

		switch {
		case as && bs:
			at[0] += bt[0]
		case as && !bs:
			addU64VS(bt, at[0])
		case bs && !as:
			addU64VS(at, bt[0])
		default:
			addU64(at, bt)
		}
		return nil
	case Float32:
		at := a.Float32s()
		bt := b.Float32s()

		switch {
		case as && bs:
			at[0] += bt[0]
		case as && !bs:
			addF32VS(bt, at[0])
		case bs && !as:
			addF32VS(at, bt[0])
		default:
			addF32(at, bt)
		}
		return nil
	case Float64:
		at := a.Float64s()
		bt := b.Float64s()

		switch {
		case as && bs:
			at[0] += bt[0]
		case as && !bs:
			addF64VS(bt, at[0])
		case bs && !as:
			addF64VS(at, bt[0])
		default:
			addF64(at, bt)
		}
		return nil
	case Complex64:
		at := a.Complex64s()
		bt := b.Complex64s()

		switch {
		case as && bs:
			at[0] += bt[0]
		case as && !bs:
			addC64VS(bt, at[0])
		case bs && !as:
			addC64VS(at, bt[0])
		default:
			addC64(at, bt)
		}
		return nil
	case Complex128:
		at := a.Complex128s()
		bt := b.Complex128s()

		switch {
		case as && bs:
			at[0] += bt[0]
		case as && !bs:
			addC128VS(bt, at[0])
		case bs && !as:
			addC128VS(at, bt[0])
		default:
			addC128(at, bt)
		}
		return nil
	case String:
		at := a.Strings()
		bt := b.Strings()

		switch {
		case as && bs:
			at[0] += bt[0]
		case as && !bs:
			addStrVS(bt, at[0])
		case bs && !as:
			addStrVS(at, bt[0])
		default:
			addStr(at, bt)
		}
		return nil
	default:
		return errors.Errorf("NYI")
	}
}

// AddIter performs addition a += b, guided by the iterator
func (e E) AddIter(t reflect.Type, a, b Array, ait, bit Iterator) error {
	as := isScalar(a)
	bs := isScalar(b)
	switch t {
	case Int:
		at := a.Ints()
		bt := b.Ints()

		switch {
		case as && bs:
			at[0] += bt[0]
			return nil
		case as && !bs:
			return addIIterVS(bt, at[0], bit)
		case bs && !as:
			return addIIterVS(at, bt[0], ait)
		default:
			return addIIter(at, bt, ait, bit)
		}
	case Int8:
		at := a.Int8s()
		bt := b.Int8s()

		switch {
		case as && bs:
			at[0] += bt[0]
			return nil
		case as && !bs:
			return addI8IterVS(bt, at[0], bit)
		case bs && !as:
			return addI8IterVS(at, bt[0], ait)
		default:
			return addI8Iter(at, bt, ait, bit)
		}
	case Int16:
		at := a.Int16s()
		bt := b.Int16s()

		switch {
		case as && bs:
			at[0] += bt[0]
			return nil
		case as && !bs:
			return addI16IterVS(bt, at[0], bit)
		case bs && !as:
			return addI16IterVS(at, bt[0], ait)
		default:
			return addI16Iter(at, bt, ait, bit)
		}
	case Int32:
		at := a.Int32s()
		bt := b.Int32s()

		switch {
		case as && bs:
			at[0] += bt[0]
			return nil
		case as && !bs:
			return addI32IterVS(bt, at[0], bit)
		case bs && !as:
			return addI32IterVS(at, bt[0], ait)
		default:
			return addI32Iter(at, bt, ait, bit)
		}
	case Int64:
		at := a.Int64s()
		bt := b.Int64s()

		switch {
		case as && bs:
			at[0] += bt[0]
			return nil
		case as && !bs:
			return addI64IterVS(bt, at[0], bit)
		case bs && !as:
			return addI64IterVS(at, bt[0], ait)
		default:
			return addI64Iter(at, bt, ait, bit)
		}
	case Uint:
		at := a.Uints()
		bt := b.Uints()

		switch {
		case as && bs:
			at[0] += bt[0]
			return nil
		case as && !bs:
			return addUIterVS(bt, at[0], bit)
		case bs && !as:
			return addUIterVS(at, bt[0], ait)
		default:
			return addUIter(at, bt, ait, bit)
		}
	case Uint8:
		at := a.Uint8s()
		bt := b.Uint8s()

		switch {
		case as && bs:
			at[0] += bt[0]
			return nil
		case as && !bs:
			return addU8IterVS(bt, at[0], bit)
		case bs && !as:
			return addU8IterVS(at, bt[0], ait)
		default:
			return addU8Iter(at, bt, ait, bit)
		}
	case Uint16:
		at := a.Uint16s()
		bt := b.Uint16s()

		switch {
		case as && bs:
			at[0] += bt[0]
			return nil
		case as && !bs:
			return addU16IterVS(bt, at[0], bit)
		case bs && !as:
			return addU16IterVS(at, bt[0], ait)
		default:
			return addU16Iter(at, bt, ait, bit)
		}
	case Uint32:
		at := a.Uint32s()
		bt := b.Uint32s()

		switch {
		case as && bs:
			at[0] += bt[0]
			return nil
		case as && !bs:
			return addU32IterVS(bt, at[0], bit)
		case bs && !as:
			return addU32IterVS(at, bt[0], ait)
		default:
			return addU32Iter(at, bt, ait, bit)
		}
	case Uint64:
		at := a.Uint64s()
		bt := b.Uint64s()

		switch {
		case as && bs:
			at[0] += bt[0]
			return nil
		case as && !bs:
			return addU64IterVS(bt, at[0], bit)
		case bs && !as:
			return addU64IterVS(at, bt[0], ait)
		default:
			return addU64Iter(at, bt, ait, bit)
		}
	case Float32:
		at := a.Float32s()
		bt := b.Float32s()

		switch {
		case as && bs:
			at[0] += bt[0]
			return nil
		case as && !bs:
			return addF32IterVS(bt, at[0], bit)
		case bs && !as:
			return addF32IterVS(at, bt[0], ait)
		default:
			return addF32Iter(at, bt, ait, bit)
		}
	case Float64:
		at := a.Float64s()
		bt := b.Float64s()

		switch {
		case as && bs:
			at[0] += bt[0]
			return nil
		case as && !bs:
			return addF64IterVS(bt, at[0], bit)
		case bs && !as:
			return addF64IterVS(at, bt[0], ait)
		default:
			return addF64Iter(at, bt, ait, bit)
		}
	case Complex64:
		at := a.Complex64s()
		bt := b.Complex64s()

		switch {
		case as && bs:
			at[0] += bt[0]
			return nil
		case as && !bs:
			return addC64IterVS(bt, at[0], bit)
		case bs && !as:
			return addC64IterVS(at, bt[0], ait)
		default:
			return addC64Iter(at, bt, ait, bit)
		}
	case Complex128:
		at := a.Complex128s()
		bt := b.Complex128s()

		switch {
		case as && bs:
			at[0] += bt[0]
			return nil
		case as && !bs:
			return addC128IterVS(bt, at[0], bit)
		case bs && !as:
			return addC128IterVS(at, bt[0], ait)
		default:
			return addC128Iter(at, bt, ait, bit)
		}
	case String:
		at := a.Strings()
		bt := b.Strings()

		switch {
		case as && bs:
			at[0] += bt[0]
			return nil
		case as && !bs:
			return addStrIterVS(bt, at[0], bit)
		case bs && !as:
			return addStrIterVS(at, bt[0], ait)
		default:
			return addStrIter(at, bt, ait, bit)
		}
	default:
		return errors.Errorf("NYI")
	}
}

// AddIncr performs incr += a + b
func (e E) AddIncr(t reflect.Type, a, b, incr Array) error {
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
				at[0] += bt[0]
				return e.Add(t, incr, a)
			}
			it[0] += at[0] + bt[0]
			return nil
		case as && !bs:
			addIncrIVS(bt, at[0], it)
		case bs && !as:
			addIncrIVS(at, bt[0], it)
		default:
			addIncrI(at, bt, it)
		}
	case Int8:
		at := a.Int8s()
		bt := b.Int8s()
		it := incr.Int8s()

		switch {
		case as && bs:
			if !is {
				at[0] += bt[0]
				return e.Add(t, incr, a)
			}
			it[0] += at[0] + bt[0]
			return nil
		case as && !bs:
			addIncrI8VS(bt, at[0], it)
		case bs && !as:
			addIncrI8VS(at, bt[0], it)
		default:
			addIncrI8(at, bt, it)
		}
	case Int16:
		at := a.Int16s()
		bt := b.Int16s()
		it := incr.Int16s()

		switch {
		case as && bs:
			if !is {
				at[0] += bt[0]
				return e.Add(t, incr, a)
			}
			it[0] += at[0] + bt[0]
			return nil
		case as && !bs:
			addIncrI16VS(bt, at[0], it)
		case bs && !as:
			addIncrI16VS(at, bt[0], it)
		default:
			addIncrI16(at, bt, it)
		}
	case Int32:
		at := a.Int32s()
		bt := b.Int32s()
		it := incr.Int32s()

		switch {
		case as && bs:
			if !is {
				at[0] += bt[0]
				return e.Add(t, incr, a)
			}
			it[0] += at[0] + bt[0]
			return nil
		case as && !bs:
			addIncrI32VS(bt, at[0], it)
		case bs && !as:
			addIncrI32VS(at, bt[0], it)
		default:
			addIncrI32(at, bt, it)
		}
	case Int64:
		at := a.Int64s()
		bt := b.Int64s()
		it := incr.Int64s()

		switch {
		case as && bs:
			if !is {
				at[0] += bt[0]
				return e.Add(t, incr, a)
			}
			it[0] += at[0] + bt[0]
			return nil
		case as && !bs:
			addIncrI64VS(bt, at[0], it)
		case bs && !as:
			addIncrI64VS(at, bt[0], it)
		default:
			addIncrI64(at, bt, it)
		}
	case Uint:
		at := a.Uints()
		bt := b.Uints()
		it := incr.Uints()

		switch {
		case as && bs:
			if !is {
				at[0] += bt[0]
				return e.Add(t, incr, a)
			}
			it[0] += at[0] + bt[0]
			return nil
		case as && !bs:
			addIncrUVS(bt, at[0], it)
		case bs && !as:
			addIncrUVS(at, bt[0], it)
		default:
			addIncrU(at, bt, it)
		}
	case Uint8:
		at := a.Uint8s()
		bt := b.Uint8s()
		it := incr.Uint8s()

		switch {
		case as && bs:
			if !is {
				at[0] += bt[0]
				return e.Add(t, incr, a)
			}
			it[0] += at[0] + bt[0]
			return nil
		case as && !bs:
			addIncrU8VS(bt, at[0], it)
		case bs && !as:
			addIncrU8VS(at, bt[0], it)
		default:
			addIncrU8(at, bt, it)
		}
	case Uint16:
		at := a.Uint16s()
		bt := b.Uint16s()
		it := incr.Uint16s()

		switch {
		case as && bs:
			if !is {
				at[0] += bt[0]
				return e.Add(t, incr, a)
			}
			it[0] += at[0] + bt[0]
			return nil
		case as && !bs:
			addIncrU16VS(bt, at[0], it)
		case bs && !as:
			addIncrU16VS(at, bt[0], it)
		default:
			addIncrU16(at, bt, it)
		}
	case Uint32:
		at := a.Uint32s()
		bt := b.Uint32s()
		it := incr.Uint32s()

		switch {
		case as && bs:
			if !is {
				at[0] += bt[0]
				return e.Add(t, incr, a)
			}
			it[0] += at[0] + bt[0]
			return nil
		case as && !bs:
			addIncrU32VS(bt, at[0], it)
		case bs && !as:
			addIncrU32VS(at, bt[0], it)
		default:
			addIncrU32(at, bt, it)
		}
	case Uint64:
		at := a.Uint64s()
		bt := b.Uint64s()
		it := incr.Uint64s()

		switch {
		case as && bs:
			if !is {
				at[0] += bt[0]
				return e.Add(t, incr, a)
			}
			it[0] += at[0] + bt[0]
			return nil
		case as && !bs:
			addIncrU64VS(bt, at[0], it)
		case bs && !as:
			addIncrU64VS(at, bt[0], it)
		default:
			addIncrU64(at, bt, it)
		}
	case Float32:
		at := a.Float32s()
		bt := b.Float32s()
		it := incr.Float32s()

		switch {
		case as && bs:
			if !is {
				at[0] += bt[0]
				return e.Add(t, incr, a)
			}
			it[0] += at[0] + bt[0]
			return nil
		case as && !bs:
			addIncrF32VS(bt, at[0], it)
		case bs && !as:
			addIncrF32VS(at, bt[0], it)
		default:
			addIncrF32(at, bt, it)
		}
	case Float64:
		at := a.Float64s()
		bt := b.Float64s()
		it := incr.Float64s()

		switch {
		case as && bs:
			if !is {
				at[0] += bt[0]
				return e.Add(t, incr, a)
			}
			it[0] += at[0] + bt[0]
			return nil
		case as && !bs:
			addIncrF64VS(bt, at[0], it)
		case bs && !as:
			addIncrF64VS(at, bt[0], it)
		default:
			addIncrF64(at, bt, it)
		}
	case Complex64:
		at := a.Complex64s()
		bt := b.Complex64s()
		it := incr.Complex64s()

		switch {
		case as && bs:
			if !is {
				at[0] += bt[0]
				return e.Add(t, incr, a)
			}
			it[0] += at[0] + bt[0]
			return nil
		case as && !bs:
			addIncrC64VS(bt, at[0], it)
		case bs && !as:
			addIncrC64VS(at, bt[0], it)
		default:
			addIncrC64(at, bt, it)
		}
	case Complex128:
		at := a.Complex128s()
		bt := b.Complex128s()
		it := incr.Complex128s()

		switch {
		case as && bs:
			if !is {
				at[0] += bt[0]
				return e.Add(t, incr, a)
			}
			it[0] += at[0] + bt[0]
			return nil
		case as && !bs:
			addIncrC128VS(bt, at[0], it)
		case bs && !as:
			addIncrC128VS(at, bt[0], it)
		default:
			addIncrC128(at, bt, it)
		}
	case String:
		at := a.Strings()
		bt := b.Strings()
		it := incr.Strings()

		switch {
		case as && bs:
			if !is {
				at[0] += bt[0]
				return e.Add(t, incr, a)
			}
			it[0] += at[0] + bt[0]
			return nil
		case as && !bs:
			addIncrStrVS(bt, at[0], it)
		case bs && !as:
			addIncrStrVS(at, bt[0], it)
		default:
			addIncrStr(at, bt, it)
		}
	default:
		return errors.Errorf("NYI")
	}
	return nil
}

// AddIterIncr performs incr += a + b, guided by iterators
func (e E) AddIterIncr(t reflect.Type, a, b, incr Array, ait, bit, iit Iterator) error {
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
				at[0] += bt[0]
				return e.AddIter(t, incr, a, iit, ait)
			}
		case as && !bs:
			return addIncrIIterVS(bt, at[0], it, bit, iit)
		case bs && !as:
			return addIncrIIterVS(at, bt[0], it, ait, iit)
		default:
			return addIncrIIter(at, bt, it, ait, bit, iit)
		}
	case Int8:
		at := a.Int8s()
		bt := b.Int8s()
		it := incr.Int8s()

		switch {
		case as && bs:
			if !is {
				at[0] += bt[0]
				return e.AddIter(t, incr, a, iit, ait)
			}
		case as && !bs:
			return addIncrI8IterVS(bt, at[0], it, bit, iit)
		case bs && !as:
			return addIncrI8IterVS(at, bt[0], it, ait, iit)
		default:
			return addIncrI8Iter(at, bt, it, ait, bit, iit)
		}
	case Int16:
		at := a.Int16s()
		bt := b.Int16s()
		it := incr.Int16s()

		switch {
		case as && bs:
			if !is {
				at[0] += bt[0]
				return e.AddIter(t, incr, a, iit, ait)
			}
		case as && !bs:
			return addIncrI16IterVS(bt, at[0], it, bit, iit)
		case bs && !as:
			return addIncrI16IterVS(at, bt[0], it, ait, iit)
		default:
			return addIncrI16Iter(at, bt, it, ait, bit, iit)
		}
	case Int32:
		at := a.Int32s()
		bt := b.Int32s()
		it := incr.Int32s()

		switch {
		case as && bs:
			if !is {
				at[0] += bt[0]
				return e.AddIter(t, incr, a, iit, ait)
			}
		case as && !bs:
			return addIncrI32IterVS(bt, at[0], it, bit, iit)
		case bs && !as:
			return addIncrI32IterVS(at, bt[0], it, ait, iit)
		default:
			return addIncrI32Iter(at, bt, it, ait, bit, iit)
		}
	case Int64:
		at := a.Int64s()
		bt := b.Int64s()
		it := incr.Int64s()

		switch {
		case as && bs:
			if !is {
				at[0] += bt[0]
				return e.AddIter(t, incr, a, iit, ait)
			}
		case as && !bs:
			return addIncrI64IterVS(bt, at[0], it, bit, iit)
		case bs && !as:
			return addIncrI64IterVS(at, bt[0], it, ait, iit)
		default:
			return addIncrI64Iter(at, bt, it, ait, bit, iit)
		}
	case Uint:
		at := a.Uints()
		bt := b.Uints()
		it := incr.Uints()

		switch {
		case as && bs:
			if !is {
				at[0] += bt[0]
				return e.AddIter(t, incr, a, iit, ait)
			}
		case as && !bs:
			return addIncrUIterVS(bt, at[0], it, bit, iit)
		case bs && !as:
			return addIncrUIterVS(at, bt[0], it, ait, iit)
		default:
			return addIncrUIter(at, bt, it, ait, bit, iit)
		}
	case Uint8:
		at := a.Uint8s()
		bt := b.Uint8s()
		it := incr.Uint8s()

		switch {
		case as && bs:
			if !is {
				at[0] += bt[0]
				return e.AddIter(t, incr, a, iit, ait)
			}
		case as && !bs:
			return addIncrU8IterVS(bt, at[0], it, bit, iit)
		case bs && !as:
			return addIncrU8IterVS(at, bt[0], it, ait, iit)
		default:
			return addIncrU8Iter(at, bt, it, ait, bit, iit)
		}
	case Uint16:
		at := a.Uint16s()
		bt := b.Uint16s()
		it := incr.Uint16s()

		switch {
		case as && bs:
			if !is {
				at[0] += bt[0]
				return e.AddIter(t, incr, a, iit, ait)
			}
		case as && !bs:
			return addIncrU16IterVS(bt, at[0], it, bit, iit)
		case bs && !as:
			return addIncrU16IterVS(at, bt[0], it, ait, iit)
		default:
			return addIncrU16Iter(at, bt, it, ait, bit, iit)
		}
	case Uint32:
		at := a.Uint32s()
		bt := b.Uint32s()
		it := incr.Uint32s()

		switch {
		case as && bs:
			if !is {
				at[0] += bt[0]
				return e.AddIter(t, incr, a, iit, ait)
			}
		case as && !bs:
			return addIncrU32IterVS(bt, at[0], it, bit, iit)
		case bs && !as:
			return addIncrU32IterVS(at, bt[0], it, ait, iit)
		default:
			return addIncrU32Iter(at, bt, it, ait, bit, iit)
		}
	case Uint64:
		at := a.Uint64s()
		bt := b.Uint64s()
		it := incr.Uint64s()

		switch {
		case as && bs:
			if !is {
				at[0] += bt[0]
				return e.AddIter(t, incr, a, iit, ait)
			}
		case as && !bs:
			return addIncrU64IterVS(bt, at[0], it, bit, iit)
		case bs && !as:
			return addIncrU64IterVS(at, bt[0], it, ait, iit)
		default:
			return addIncrU64Iter(at, bt, it, ait, bit, iit)
		}
	case Float32:
		at := a.Float32s()
		bt := b.Float32s()
		it := incr.Float32s()

		switch {
		case as && bs:
			if !is {
				at[0] += bt[0]
				return e.AddIter(t, incr, a, iit, ait)
			}
		case as && !bs:
			return addIncrF32IterVS(bt, at[0], it, bit, iit)
		case bs && !as:
			return addIncrF32IterVS(at, bt[0], it, ait, iit)
		default:
			return addIncrF32Iter(at, bt, it, ait, bit, iit)
		}
	case Float64:
		at := a.Float64s()
		bt := b.Float64s()
		it := incr.Float64s()

		switch {
		case as && bs:
			if !is {
				at[0] += bt[0]
				return e.AddIter(t, incr, a, iit, ait)
			}
		case as && !bs:
			return addIncrF64IterVS(bt, at[0], it, bit, iit)
		case bs && !as:
			return addIncrF64IterVS(at, bt[0], it, ait, iit)
		default:
			return addIncrF64Iter(at, bt, it, ait, bit, iit)
		}
	case Complex64:
		at := a.Complex64s()
		bt := b.Complex64s()
		it := incr.Complex64s()

		switch {
		case as && bs:
			if !is {
				at[0] += bt[0]
				return e.AddIter(t, incr, a, iit, ait)
			}
		case as && !bs:
			return addIncrC64IterVS(bt, at[0], it, bit, iit)
		case bs && !as:
			return addIncrC64IterVS(at, bt[0], it, ait, iit)
		default:
			return addIncrC64Iter(at, bt, it, ait, bit, iit)
		}
	case Complex128:
		at := a.Complex128s()
		bt := b.Complex128s()
		it := incr.Complex128s()

		switch {
		case as && bs:
			if !is {
				at[0] += bt[0]
				return e.AddIter(t, incr, a, iit, ait)
			}
		case as && !bs:
			return addIncrC128IterVS(bt, at[0], it, bit, iit)
		case bs && !as:
			return addIncrC128IterVS(at, bt[0], it, ait, iit)
		default:
			return addIncrC128Iter(at, bt, it, ait, bit, iit)
		}
	case String:
		at := a.Strings()
		bt := b.Strings()
		it := incr.Strings()

		switch {
		case as && bs:
			if !is {
				at[0] += bt[0]
				return e.AddIter(t, incr, a, iit, ait)
			}
		case as && !bs:
			return addIncrStrIterVS(bt, at[0], it, bit, iit)
		case bs && !as:
			return addIncrStrIterVS(at, bt[0], it, ait, iit)
		default:
			return addIncrStrIter(at, bt, it, ait, bit, iit)
		}
	default:
		return errors.Errorf("NYI")
	}
	return nil
}

func addI(a, b []int) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range b {
		a[i] += v
	}
}

func addIVS(a []int, b int) {
	a = a[:]
	for i := range a {
		a[i] += b
	}
}

func addIIter(a, b []int, ait, bit Iterator) (err error) {
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
		a[i] += b[j]
	}
	return
}

func addIIterVS(a []int, b int, it Iterator) (err error) {
	var i int
	for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
		a[i] += b
	}
	err = handleNoOp(err)
	return
}

func addIncrI(a, b, incr []int) {
	a = a[:]
	b = b[:len(a)]
	incr = incr[:len(a)]
	for i, v := range a {
		incr[i] += v + b[i]
	}
}

func addIncrIVS(a []int, b int, incr []int) {
	a = a[:]
	incr = incr[:len(a)]
	for i := range a {
		incr[i] += a[i] + b
	}
}

func addIncrIIterVS(a []int, b int, incr []int, ait, iit Iterator) (err error) {
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
		incr[j] += a[i] + b
	}
	return nil
}

func addIncrIIter(a, b, incr []int, ait, bit, iit Iterator) (err error) {
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
		incr[k] += a[i] + b[j]
	}
	return nil
}

func addI8(a, b []int8) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range b {
		a[i] += v
	}
}

func addI8VS(a []int8, b int8) {
	a = a[:]
	for i := range a {
		a[i] += b
	}
}

func addI8Iter(a, b []int8, ait, bit Iterator) (err error) {
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
		a[i] += b[j]
	}
	return
}

func addI8IterVS(a []int8, b int8, it Iterator) (err error) {
	var i int
	for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
		a[i] += b
	}
	err = handleNoOp(err)
	return
}

func addIncrI8(a, b, incr []int8) {
	a = a[:]
	b = b[:len(a)]
	incr = incr[:len(a)]
	for i, v := range a {
		incr[i] += v + b[i]
	}
}

func addIncrI8VS(a []int8, b int8, incr []int8) {
	a = a[:]
	incr = incr[:len(a)]
	for i := range a {
		incr[i] += a[i] + b
	}
}

func addIncrI8IterVS(a []int8, b int8, incr []int8, ait, iit Iterator) (err error) {
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
		incr[j] += a[i] + b
	}
	return nil
}

func addIncrI8Iter(a, b, incr []int8, ait, bit, iit Iterator) (err error) {
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
		incr[k] += a[i] + b[j]
	}
	return nil
}

func addI16(a, b []int16) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range b {
		a[i] += v
	}
}

func addI16VS(a []int16, b int16) {
	a = a[:]
	for i := range a {
		a[i] += b
	}
}

func addI16Iter(a, b []int16, ait, bit Iterator) (err error) {
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
		a[i] += b[j]
	}
	return
}

func addI16IterVS(a []int16, b int16, it Iterator) (err error) {
	var i int
	for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
		a[i] += b
	}
	err = handleNoOp(err)
	return
}

func addIncrI16(a, b, incr []int16) {
	a = a[:]
	b = b[:len(a)]
	incr = incr[:len(a)]
	for i, v := range a {
		incr[i] += v + b[i]
	}
}

func addIncrI16VS(a []int16, b int16, incr []int16) {
	a = a[:]
	incr = incr[:len(a)]
	for i := range a {
		incr[i] += a[i] + b
	}
}

func addIncrI16IterVS(a []int16, b int16, incr []int16, ait, iit Iterator) (err error) {
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
		incr[j] += a[i] + b
	}
	return nil
}

func addIncrI16Iter(a, b, incr []int16, ait, bit, iit Iterator) (err error) {
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
		incr[k] += a[i] + b[j]
	}
	return nil
}

func addI32(a, b []int32) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range b {
		a[i] += v
	}
}

func addI32VS(a []int32, b int32) {
	a = a[:]
	for i := range a {
		a[i] += b
	}
}

func addI32Iter(a, b []int32, ait, bit Iterator) (err error) {
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
		a[i] += b[j]
	}
	return
}

func addI32IterVS(a []int32, b int32, it Iterator) (err error) {
	var i int
	for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
		a[i] += b
	}
	err = handleNoOp(err)
	return
}

func addIncrI32(a, b, incr []int32) {
	a = a[:]
	b = b[:len(a)]
	incr = incr[:len(a)]
	for i, v := range a {
		incr[i] += v + b[i]
	}
}

func addIncrI32VS(a []int32, b int32, incr []int32) {
	a = a[:]
	incr = incr[:len(a)]
	for i := range a {
		incr[i] += a[i] + b
	}
}

func addIncrI32IterVS(a []int32, b int32, incr []int32, ait, iit Iterator) (err error) {
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
		incr[j] += a[i] + b
	}
	return nil
}

func addIncrI32Iter(a, b, incr []int32, ait, bit, iit Iterator) (err error) {
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
		incr[k] += a[i] + b[j]
	}
	return nil
}

func addI64(a, b []int64) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range b {
		a[i] += v
	}
}

func addI64VS(a []int64, b int64) {
	a = a[:]
	for i := range a {
		a[i] += b
	}
}

func addI64Iter(a, b []int64, ait, bit Iterator) (err error) {
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
		a[i] += b[j]
	}
	return
}

func addI64IterVS(a []int64, b int64, it Iterator) (err error) {
	var i int
	for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
		a[i] += b
	}
	err = handleNoOp(err)
	return
}

func addIncrI64(a, b, incr []int64) {
	a = a[:]
	b = b[:len(a)]
	incr = incr[:len(a)]
	for i, v := range a {
		incr[i] += v + b[i]
	}
}

func addIncrI64VS(a []int64, b int64, incr []int64) {
	a = a[:]
	incr = incr[:len(a)]
	for i := range a {
		incr[i] += a[i] + b
	}
}

func addIncrI64IterVS(a []int64, b int64, incr []int64, ait, iit Iterator) (err error) {
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
		incr[j] += a[i] + b
	}
	return nil
}

func addIncrI64Iter(a, b, incr []int64, ait, bit, iit Iterator) (err error) {
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
		incr[k] += a[i] + b[j]
	}
	return nil
}

func addU(a, b []uint) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range b {
		a[i] += v
	}
}

func addUVS(a []uint, b uint) {
	a = a[:]
	for i := range a {
		a[i] += b
	}
}

func addUIter(a, b []uint, ait, bit Iterator) (err error) {
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
		a[i] += b[j]
	}
	return
}

func addUIterVS(a []uint, b uint, it Iterator) (err error) {
	var i int
	for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
		a[i] += b
	}
	err = handleNoOp(err)
	return
}

func addIncrU(a, b, incr []uint) {
	a = a[:]
	b = b[:len(a)]
	incr = incr[:len(a)]
	for i, v := range a {
		incr[i] += v + b[i]
	}
}

func addIncrUVS(a []uint, b uint, incr []uint) {
	a = a[:]
	incr = incr[:len(a)]
	for i := range a {
		incr[i] += a[i] + b
	}
}

func addIncrUIterVS(a []uint, b uint, incr []uint, ait, iit Iterator) (err error) {
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
		incr[j] += a[i] + b
	}
	return nil
}

func addIncrUIter(a, b, incr []uint, ait, bit, iit Iterator) (err error) {
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
		incr[k] += a[i] + b[j]
	}
	return nil
}

func addU8(a, b []uint8) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range b {
		a[i] += v
	}
}

func addU8VS(a []uint8, b uint8) {
	a = a[:]
	for i := range a {
		a[i] += b
	}
}

func addU8Iter(a, b []uint8, ait, bit Iterator) (err error) {
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
		a[i] += b[j]
	}
	return
}

func addU8IterVS(a []uint8, b uint8, it Iterator) (err error) {
	var i int
	for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
		a[i] += b
	}
	err = handleNoOp(err)
	return
}

func addIncrU8(a, b, incr []uint8) {
	a = a[:]
	b = b[:len(a)]
	incr = incr[:len(a)]
	for i, v := range a {
		incr[i] += v + b[i]
	}
}

func addIncrU8VS(a []uint8, b uint8, incr []uint8) {
	a = a[:]
	incr = incr[:len(a)]
	for i := range a {
		incr[i] += a[i] + b
	}
}

func addIncrU8IterVS(a []uint8, b uint8, incr []uint8, ait, iit Iterator) (err error) {
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
		incr[j] += a[i] + b
	}
	return nil
}

func addIncrU8Iter(a, b, incr []uint8, ait, bit, iit Iterator) (err error) {
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
		incr[k] += a[i] + b[j]
	}
	return nil
}

func addU16(a, b []uint16) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range b {
		a[i] += v
	}
}

func addU16VS(a []uint16, b uint16) {
	a = a[:]
	for i := range a {
		a[i] += b
	}
}

func addU16Iter(a, b []uint16, ait, bit Iterator) (err error) {
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
		a[i] += b[j]
	}
	return
}

func addU16IterVS(a []uint16, b uint16, it Iterator) (err error) {
	var i int
	for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
		a[i] += b
	}
	err = handleNoOp(err)
	return
}

func addIncrU16(a, b, incr []uint16) {
	a = a[:]
	b = b[:len(a)]
	incr = incr[:len(a)]
	for i, v := range a {
		incr[i] += v + b[i]
	}
}

func addIncrU16VS(a []uint16, b uint16, incr []uint16) {
	a = a[:]
	incr = incr[:len(a)]
	for i := range a {
		incr[i] += a[i] + b
	}
}

func addIncrU16IterVS(a []uint16, b uint16, incr []uint16, ait, iit Iterator) (err error) {
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
		incr[j] += a[i] + b
	}
	return nil
}

func addIncrU16Iter(a, b, incr []uint16, ait, bit, iit Iterator) (err error) {
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
		incr[k] += a[i] + b[j]
	}
	return nil
}

func addU32(a, b []uint32) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range b {
		a[i] += v
	}
}

func addU32VS(a []uint32, b uint32) {
	a = a[:]
	for i := range a {
		a[i] += b
	}
}

func addU32Iter(a, b []uint32, ait, bit Iterator) (err error) {
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
		a[i] += b[j]
	}
	return
}

func addU32IterVS(a []uint32, b uint32, it Iterator) (err error) {
	var i int
	for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
		a[i] += b
	}
	err = handleNoOp(err)
	return
}

func addIncrU32(a, b, incr []uint32) {
	a = a[:]
	b = b[:len(a)]
	incr = incr[:len(a)]
	for i, v := range a {
		incr[i] += v + b[i]
	}
}

func addIncrU32VS(a []uint32, b uint32, incr []uint32) {
	a = a[:]
	incr = incr[:len(a)]
	for i := range a {
		incr[i] += a[i] + b
	}
}

func addIncrU32IterVS(a []uint32, b uint32, incr []uint32, ait, iit Iterator) (err error) {
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
		incr[j] += a[i] + b
	}
	return nil
}

func addIncrU32Iter(a, b, incr []uint32, ait, bit, iit Iterator) (err error) {
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
		incr[k] += a[i] + b[j]
	}
	return nil
}

func addU64(a, b []uint64) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range b {
		a[i] += v
	}
}

func addU64VS(a []uint64, b uint64) {
	a = a[:]
	for i := range a {
		a[i] += b
	}
}

func addU64Iter(a, b []uint64, ait, bit Iterator) (err error) {
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
		a[i] += b[j]
	}
	return
}

func addU64IterVS(a []uint64, b uint64, it Iterator) (err error) {
	var i int
	for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
		a[i] += b
	}
	err = handleNoOp(err)
	return
}

func addIncrU64(a, b, incr []uint64) {
	a = a[:]
	b = b[:len(a)]
	incr = incr[:len(a)]
	for i, v := range a {
		incr[i] += v + b[i]
	}
}

func addIncrU64VS(a []uint64, b uint64, incr []uint64) {
	a = a[:]
	incr = incr[:len(a)]
	for i := range a {
		incr[i] += a[i] + b
	}
}

func addIncrU64IterVS(a []uint64, b uint64, incr []uint64, ait, iit Iterator) (err error) {
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
		incr[j] += a[i] + b
	}
	return nil
}

func addIncrU64Iter(a, b, incr []uint64, ait, bit, iit Iterator) (err error) {
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
		incr[k] += a[i] + b[j]
	}
	return nil
}

func addF32(a, b []float32) {
	vecf32.Add(a, b)
}

func addF32VS(a []float32, b float32) {
	vecf32.Trans(a, b)
}

func addF32Iter(a, b []float32, ait, bit Iterator) (err error) {
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
		a[i] += b[j]
	}
	return
}

func addF32IterVS(a []float32, b float32, it Iterator) (err error) {
	var i int
	for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
		a[i] += b
	}
	err = handleNoOp(err)
	return
}

func addIncrF32(a, b, incr []float32) {
	vecf32.IncrAdd(a, b, incr)
}

func addIncrF32VS(a []float32, b float32, incr []float32) {
	vecf32.IncrScale(a, b, incr)
}

func addIncrF32IterVS(a []float32, b float32, incr []float32, ait, iit Iterator) (err error) {
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
		incr[j] += a[i] + b
	}
	return nil
}

func addIncrF32Iter(a, b, incr []float32, ait, bit, iit Iterator) (err error) {
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
		incr[k] += a[i] + b[j]
	}
	return nil
}

func addF64(a, b []float64) {
	vecf64.Add(a, b)
}

func addF64VS(a []float64, b float64) {
	vecf64.Trans(a, b)
}

func addF64Iter(a, b []float64, ait, bit Iterator) (err error) {
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
		a[i] += b[j]
	}
	return
}

func addF64IterVS(a []float64, b float64, it Iterator) (err error) {
	var i int
	for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
		a[i] += b
	}
	err = handleNoOp(err)
	return
}

func addIncrF64(a, b, incr []float64) {
	vecf64.IncrAdd(a, b, incr)
}

func addIncrF64VS(a []float64, b float64, incr []float64) {
	vecf64.IncrScale(a, b, incr)
}

func addIncrF64IterVS(a []float64, b float64, incr []float64, ait, iit Iterator) (err error) {
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
		incr[j] += a[i] + b
	}
	return nil
}

func addIncrF64Iter(a, b, incr []float64, ait, bit, iit Iterator) (err error) {
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
		incr[k] += a[i] + b[j]
	}
	return nil
}

func addC64(a, b []complex64) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range b {
		a[i] += v
	}
}

func addC64VS(a []complex64, b complex64) {
	a = a[:]
	for i := range a {
		a[i] += b
	}
}

func addC64Iter(a, b []complex64, ait, bit Iterator) (err error) {
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
		a[i] += b[j]
	}
	return
}

func addC64IterVS(a []complex64, b complex64, it Iterator) (err error) {
	var i int
	for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
		a[i] += b
	}
	err = handleNoOp(err)
	return
}

func addIncrC64(a, b, incr []complex64) {
	a = a[:]
	b = b[:len(a)]
	incr = incr[:len(a)]
	for i, v := range a {
		incr[i] += v + b[i]
	}
}

func addIncrC64VS(a []complex64, b complex64, incr []complex64) {
	a = a[:]
	incr = incr[:len(a)]
	for i := range a {
		incr[i] += a[i] + b
	}
}

func addIncrC64IterVS(a []complex64, b complex64, incr []complex64, ait, iit Iterator) (err error) {
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
		incr[j] += a[i] + b
	}
	return nil
}

func addIncrC64Iter(a, b, incr []complex64, ait, bit, iit Iterator) (err error) {
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
		incr[k] += a[i] + b[j]
	}
	return nil
}

func addC128(a, b []complex128) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range b {
		a[i] += v
	}
}

func addC128VS(a []complex128, b complex128) {
	a = a[:]
	for i := range a {
		a[i] += b
	}
}

func addC128Iter(a, b []complex128, ait, bit Iterator) (err error) {
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
		a[i] += b[j]
	}
	return
}

func addC128IterVS(a []complex128, b complex128, it Iterator) (err error) {
	var i int
	for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
		a[i] += b
	}
	err = handleNoOp(err)
	return
}

func addIncrC128(a, b, incr []complex128) {
	a = a[:]
	b = b[:len(a)]
	incr = incr[:len(a)]
	for i, v := range a {
		incr[i] += v + b[i]
	}
}

func addIncrC128VS(a []complex128, b complex128, incr []complex128) {
	a = a[:]
	incr = incr[:len(a)]
	for i := range a {
		incr[i] += a[i] + b
	}
}

func addIncrC128IterVS(a []complex128, b complex128, incr []complex128, ait, iit Iterator) (err error) {
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
		incr[j] += a[i] + b
	}
	return nil
}

func addIncrC128Iter(a, b, incr []complex128, ait, bit, iit Iterator) (err error) {
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
		incr[k] += a[i] + b[j]
	}
	return nil
}

func addStr(a, b []string) {
	a = a[:]
	b = b[:len(a)]
	for i, v := range b {
		a[i] += v
	}
}

func addStrVS(a []string, b string) {
	a = a[:]
	for i := range a {
		a[i] += b
	}
}

func addStrIter(a, b []string, ait, bit Iterator) (err error) {
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
		a[i] += b[j]
	}
	return
}

func addStrIterVS(a []string, b string, it Iterator) (err error) {
	var i int
	for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
		a[i] += b
	}
	err = handleNoOp(err)
	return
}

func addIncrStr(a, b, incr []string) {
	a = a[:]
	b = b[:len(a)]
	incr = incr[:len(a)]
	for i, v := range a {
		incr[i] += v + b[i]
	}
}

func addIncrStrVS(a []string, b string, incr []string) {
	a = a[:]
	incr = incr[:len(a)]
	for i := range a {
		incr[i] += a[i] + b
	}
}

func addIncrStrIterVS(a []string, b string, incr []string, ait, iit Iterator) (err error) {
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
		incr[j] += a[i] + b
	}
	return nil
}

func addIncrStrIter(a, b, incr []string, ait, bit, iit Iterator) (err error) {
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
		incr[k] += a[i] + b[j]
	}
	return nil
}
