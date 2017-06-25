package tensor

import (
	"reflect"
	"unsafe"

	"github.com/pkg/errors"
)

/*
GENERATED FILE. DO NOT EDIT
*/

func prepBinaryDenseCmp(a, b *Dense, opts ...FuncOpt) (reuse *Dense, safe, same, toReuse bool, err error) {
	if a.t.Kind() != b.t.Kind() {
		err = errors.Errorf(dtypeMismatch, a.t, b.t)
		return
	}

	if !a.Shape().Eq(b.Shape()) {
		err = errors.Errorf(shapeMismatch, a.Shape(), b.Shape())
		return
	}
	fo := ParseFuncOpts(opts...)
	reuseT, _ := fo.IncrReuse()
	safe = fo.Safe()
	same = fo.Same()
	if !safe {
		same = true
	}
	toReuse = reuseT != nil

	if toReuse {
		reuse = reuseT.(*Dense)
		if same {
			if reuse.t.Kind() != a.t.Kind() {
				err = errors.Errorf(dtypeMismatch, a.t, reuse.t)
				return
			}
		} else {
			if reuse.t.Kind() != reflect.Bool {
				err = errors.Errorf(dtypeMismatch, reflect.Bool, reuse.t)
				return
			}
		}

		if err = reuseDenseCheck(reuse, a); err != nil {
			err = errors.Wrap(err, "Cannot use reuse")
			return
		}
	}
	return
}

func prepUnaryDenseCmp(a *Dense, opts ...FuncOpt) (reuse *Dense, safe, same, toReuse bool, err error) {
	fo := ParseFuncOpts(opts...)
	reuseT, _ := fo.IncrReuse()
	safe = fo.Safe()
	same = fo.Same()
	if !safe {
		same = true
	}
	toReuse = reuseT != nil

	if toReuse {
		reuse = reuseT.(*Dense)
		if same {
			if reuse.t.Kind() != a.t.Kind() {
				err = errors.Errorf(dtypeMismatch, a.t, reuse.t)
				return
			}
		} else {
			if reuse.t.Kind() != reflect.Bool {
				err = errors.Errorf(dtypeMismatch, reflect.Bool, reuse.t)
				return
			}
		}

		if err = reuseDenseCheck(reuse, a); err != nil {
			err = errors.Wrap(err, "Cannot use reuse")
			return
		}
	}
	return
}

/* Eq */

func (t *Dense) eqDD(other *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	reuse, safe, same, toReuse, err := prepBinaryDenseCmp(t, other, opts...)
	if err != nil {
		return nil, err
	}

	retVal = recycledDenseNoFix(t.t, t.Shape().Clone())
	var ret interface{} // slice of some sort
	switch t.t.Kind() {

	case reflect.Bool:
		td := t.bools()
		od := other.bools()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			bs := make([]bool, t.Shape().TotalSize())
			ret = bs
			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				bs[k] = td[i] == od[j]
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			bs := make([]bool, t.Shape().TotalSize())
			ret = bs
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				bs[k] = td[i] == od[j]
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			bs := make([]bool, t.Shape().TotalSize())
			ret = bs
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				bs[k] = td[i] == od[j]
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			ret = eqDDBoolsB(td, od)
		}

	case reflect.Int:
		td := t.ints()
		od := other.ints()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []int
			if same {
				ss = make([]int, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] == od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] == od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int
			if same {
				ss = make([]int, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] == od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] == od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int
			if same {
				ss = make([]int, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] == od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] == od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = eqDDSameI(td, od)
			} else {
				ret = eqDDBoolsI(td, od)
			}
		}

	case reflect.Int8:
		td := t.int8s()
		od := other.int8s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []int8
			if same {
				ss = make([]int8, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] == od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] == od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int8
			if same {
				ss = make([]int8, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] == od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] == od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int8
			if same {
				ss = make([]int8, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] == od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] == od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = eqDDSameI8(td, od)
			} else {
				ret = eqDDBoolsI8(td, od)
			}
		}

	case reflect.Int16:
		td := t.int16s()
		od := other.int16s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []int16
			if same {
				ss = make([]int16, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] == od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] == od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int16
			if same {
				ss = make([]int16, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] == od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] == od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int16
			if same {
				ss = make([]int16, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] == od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] == od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = eqDDSameI16(td, od)
			} else {
				ret = eqDDBoolsI16(td, od)
			}
		}

	case reflect.Int32:
		td := t.int32s()
		od := other.int32s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []int32
			if same {
				ss = make([]int32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] == od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] == od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int32
			if same {
				ss = make([]int32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] == od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] == od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int32
			if same {
				ss = make([]int32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] == od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] == od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = eqDDSameI32(td, od)
			} else {
				ret = eqDDBoolsI32(td, od)
			}
		}

	case reflect.Int64:
		td := t.int64s()
		od := other.int64s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []int64
			if same {
				ss = make([]int64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] == od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] == od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int64
			if same {
				ss = make([]int64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] == od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] == od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int64
			if same {
				ss = make([]int64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] == od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] == od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = eqDDSameI64(td, od)
			} else {
				ret = eqDDBoolsI64(td, od)
			}
		}

	case reflect.Uint:
		td := t.uints()
		od := other.uints()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []uint
			if same {
				ss = make([]uint, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] == od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] == od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint
			if same {
				ss = make([]uint, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] == od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] == od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint
			if same {
				ss = make([]uint, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] == od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] == od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = eqDDSameU(td, od)
			} else {
				ret = eqDDBoolsU(td, od)
			}
		}

	case reflect.Uint8:
		td := t.uint8s()
		od := other.uint8s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []uint8
			if same {
				ss = make([]uint8, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] == od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] == od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint8
			if same {
				ss = make([]uint8, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] == od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] == od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint8
			if same {
				ss = make([]uint8, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] == od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] == od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = eqDDSameU8(td, od)
			} else {
				ret = eqDDBoolsU8(td, od)
			}
		}

	case reflect.Uint16:
		td := t.uint16s()
		od := other.uint16s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []uint16
			if same {
				ss = make([]uint16, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] == od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] == od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint16
			if same {
				ss = make([]uint16, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] == od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] == od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint16
			if same {
				ss = make([]uint16, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] == od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] == od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = eqDDSameU16(td, od)
			} else {
				ret = eqDDBoolsU16(td, od)
			}
		}

	case reflect.Uint32:
		td := t.uint32s()
		od := other.uint32s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []uint32
			if same {
				ss = make([]uint32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] == od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] == od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint32
			if same {
				ss = make([]uint32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] == od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] == od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint32
			if same {
				ss = make([]uint32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] == od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] == od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = eqDDSameU32(td, od)
			} else {
				ret = eqDDBoolsU32(td, od)
			}
		}

	case reflect.Uint64:
		td := t.uint64s()
		od := other.uint64s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []uint64
			if same {
				ss = make([]uint64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] == od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] == od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint64
			if same {
				ss = make([]uint64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] == od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] == od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint64
			if same {
				ss = make([]uint64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] == od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] == od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = eqDDSameU64(td, od)
			} else {
				ret = eqDDBoolsU64(td, od)
			}
		}

	case reflect.Uintptr:
		td := t.uintptrs()
		od := other.uintptrs()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			bs := make([]bool, t.Shape().TotalSize())
			ret = bs
			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				bs[k] = td[i] == od[j]
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			bs := make([]bool, t.Shape().TotalSize())
			ret = bs
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				bs[k] = td[i] == od[j]
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			bs := make([]bool, t.Shape().TotalSize())
			ret = bs
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				bs[k] = td[i] == od[j]
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			ret = eqDDBoolsUintptr(td, od)
		}

	case reflect.Float32:
		td := t.float32s()
		od := other.float32s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []float32
			if same {
				ss = make([]float32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] == od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] == od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []float32
			if same {
				ss = make([]float32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] == od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] == od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []float32
			if same {
				ss = make([]float32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] == od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] == od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = eqDDSameF32(td, od)
			} else {
				ret = eqDDBoolsF32(td, od)
			}
		}

	case reflect.Float64:
		td := t.float64s()
		od := other.float64s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []float64
			if same {
				ss = make([]float64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] == od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] == od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []float64
			if same {
				ss = make([]float64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] == od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] == od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []float64
			if same {
				ss = make([]float64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] == od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] == od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = eqDDSameF64(td, od)
			} else {
				ret = eqDDBoolsF64(td, od)
			}
		}

	case reflect.Complex64:
		td := t.complex64s()
		od := other.complex64s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []complex64
			if same {
				ss = make([]complex64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] == od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] == od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []complex64
			if same {
				ss = make([]complex64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] == od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] == od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []complex64
			if same {
				ss = make([]complex64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] == od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] == od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = eqDDSameC64(td, od)
			} else {
				ret = eqDDBoolsC64(td, od)
			}
		}

	case reflect.Complex128:
		td := t.complex128s()
		od := other.complex128s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []complex128
			if same {
				ss = make([]complex128, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] == od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] == od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []complex128
			if same {
				ss = make([]complex128, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] == od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] == od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []complex128
			if same {
				ss = make([]complex128, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] == od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] == od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = eqDDSameC128(td, od)
			} else {
				ret = eqDDBoolsC128(td, od)
			}
		}

	case reflect.String:
		td := t.strings()
		od := other.strings()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			bs := make([]bool, t.Shape().TotalSize())
			ret = bs
			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				bs[k] = td[i] == od[j]
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			bs := make([]bool, t.Shape().TotalSize())
			ret = bs
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				bs[k] = td[i] == od[j]
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			bs := make([]bool, t.Shape().TotalSize())
			ret = bs
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				bs[k] = td[i] == od[j]
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			ret = eqDDBoolsStr(td, od)
		}

	case reflect.UnsafePointer:
		td := t.unsafePointers()
		od := other.unsafePointers()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			bs := make([]bool, t.Shape().TotalSize())
			ret = bs
			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				bs[k] = td[i] == od[j]
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			bs := make([]bool, t.Shape().TotalSize())
			ret = bs
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				bs[k] = td[i] == od[j]
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			bs := make([]bool, t.Shape().TotalSize())
			ret = bs
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				bs[k] = td[i] == od[j]
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			ret = eqDDBoolsUnsafePointer(td, od)
		}

	default:
		err = errors.Errorf(unsupportedDtype, t.t, "eq")
	}

	if err != nil {
		return
	}
	retVal.fromSlice(ret)
	retVal.fix()
	err = retVal.sanity()

	switch {
	case toReuse:
		copyDense(reuse, retVal)
		ReturnTensor(retVal)
		retVal = reuse
	case !safe:
		copyDense(t, retVal)
		ReturnTensor(retVal)
		retVal = t
	}
	return
}

/* Ne */

func (t *Dense) neDD(other *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	reuse, safe, same, toReuse, err := prepBinaryDenseCmp(t, other, opts...)
	if err != nil {
		return nil, err
	}

	retVal = recycledDenseNoFix(t.t, t.Shape().Clone())
	var ret interface{} // slice of some sort
	switch t.t.Kind() {

	case reflect.Bool:
		td := t.bools()
		od := other.bools()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			bs := make([]bool, t.Shape().TotalSize())
			ret = bs
			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				bs[k] = td[i] != od[j]
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			bs := make([]bool, t.Shape().TotalSize())
			ret = bs
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				bs[k] = td[i] != od[j]
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			bs := make([]bool, t.Shape().TotalSize())
			ret = bs
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				bs[k] = td[i] != od[j]
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			ret = neDDBoolsB(td, od)
		}

	case reflect.Int:
		td := t.ints()
		od := other.ints()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []int
			if same {
				ss = make([]int, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] != od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] != od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int
			if same {
				ss = make([]int, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] != od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] != od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int
			if same {
				ss = make([]int, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] != od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] != od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = neDDSameI(td, od)
			} else {
				ret = neDDBoolsI(td, od)
			}
		}

	case reflect.Int8:
		td := t.int8s()
		od := other.int8s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []int8
			if same {
				ss = make([]int8, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] != od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] != od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int8
			if same {
				ss = make([]int8, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] != od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] != od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int8
			if same {
				ss = make([]int8, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] != od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] != od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = neDDSameI8(td, od)
			} else {
				ret = neDDBoolsI8(td, od)
			}
		}

	case reflect.Int16:
		td := t.int16s()
		od := other.int16s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []int16
			if same {
				ss = make([]int16, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] != od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] != od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int16
			if same {
				ss = make([]int16, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] != od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] != od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int16
			if same {
				ss = make([]int16, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] != od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] != od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = neDDSameI16(td, od)
			} else {
				ret = neDDBoolsI16(td, od)
			}
		}

	case reflect.Int32:
		td := t.int32s()
		od := other.int32s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []int32
			if same {
				ss = make([]int32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] != od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] != od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int32
			if same {
				ss = make([]int32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] != od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] != od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int32
			if same {
				ss = make([]int32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] != od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] != od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = neDDSameI32(td, od)
			} else {
				ret = neDDBoolsI32(td, od)
			}
		}

	case reflect.Int64:
		td := t.int64s()
		od := other.int64s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []int64
			if same {
				ss = make([]int64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] != od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] != od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int64
			if same {
				ss = make([]int64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] != od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] != od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int64
			if same {
				ss = make([]int64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] != od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] != od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = neDDSameI64(td, od)
			} else {
				ret = neDDBoolsI64(td, od)
			}
		}

	case reflect.Uint:
		td := t.uints()
		od := other.uints()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []uint
			if same {
				ss = make([]uint, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] != od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] != od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint
			if same {
				ss = make([]uint, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] != od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] != od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint
			if same {
				ss = make([]uint, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] != od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] != od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = neDDSameU(td, od)
			} else {
				ret = neDDBoolsU(td, od)
			}
		}

	case reflect.Uint8:
		td := t.uint8s()
		od := other.uint8s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []uint8
			if same {
				ss = make([]uint8, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] != od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] != od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint8
			if same {
				ss = make([]uint8, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] != od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] != od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint8
			if same {
				ss = make([]uint8, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] != od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] != od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = neDDSameU8(td, od)
			} else {
				ret = neDDBoolsU8(td, od)
			}
		}

	case reflect.Uint16:
		td := t.uint16s()
		od := other.uint16s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []uint16
			if same {
				ss = make([]uint16, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] != od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] != od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint16
			if same {
				ss = make([]uint16, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] != od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] != od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint16
			if same {
				ss = make([]uint16, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] != od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] != od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = neDDSameU16(td, od)
			} else {
				ret = neDDBoolsU16(td, od)
			}
		}

	case reflect.Uint32:
		td := t.uint32s()
		od := other.uint32s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []uint32
			if same {
				ss = make([]uint32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] != od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] != od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint32
			if same {
				ss = make([]uint32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] != od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] != od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint32
			if same {
				ss = make([]uint32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] != od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] != od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = neDDSameU32(td, od)
			} else {
				ret = neDDBoolsU32(td, od)
			}
		}

	case reflect.Uint64:
		td := t.uint64s()
		od := other.uint64s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []uint64
			if same {
				ss = make([]uint64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] != od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] != od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint64
			if same {
				ss = make([]uint64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] != od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] != od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint64
			if same {
				ss = make([]uint64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] != od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] != od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = neDDSameU64(td, od)
			} else {
				ret = neDDBoolsU64(td, od)
			}
		}

	case reflect.Uintptr:
		td := t.uintptrs()
		od := other.uintptrs()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			bs := make([]bool, t.Shape().TotalSize())
			ret = bs
			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				bs[k] = td[i] != od[j]
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			bs := make([]bool, t.Shape().TotalSize())
			ret = bs
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				bs[k] = td[i] != od[j]
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			bs := make([]bool, t.Shape().TotalSize())
			ret = bs
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				bs[k] = td[i] != od[j]
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			ret = neDDBoolsUintptr(td, od)
		}

	case reflect.Float32:
		td := t.float32s()
		od := other.float32s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []float32
			if same {
				ss = make([]float32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] != od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] != od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []float32
			if same {
				ss = make([]float32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] != od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] != od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []float32
			if same {
				ss = make([]float32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] != od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] != od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = neDDSameF32(td, od)
			} else {
				ret = neDDBoolsF32(td, od)
			}
		}

	case reflect.Float64:
		td := t.float64s()
		od := other.float64s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []float64
			if same {
				ss = make([]float64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] != od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] != od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []float64
			if same {
				ss = make([]float64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] != od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] != od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []float64
			if same {
				ss = make([]float64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] != od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] != od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = neDDSameF64(td, od)
			} else {
				ret = neDDBoolsF64(td, od)
			}
		}

	case reflect.Complex64:
		td := t.complex64s()
		od := other.complex64s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []complex64
			if same {
				ss = make([]complex64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] != od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] != od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []complex64
			if same {
				ss = make([]complex64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] != od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] != od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []complex64
			if same {
				ss = make([]complex64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] != od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] != od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = neDDSameC64(td, od)
			} else {
				ret = neDDBoolsC64(td, od)
			}
		}

	case reflect.Complex128:
		td := t.complex128s()
		od := other.complex128s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []complex128
			if same {
				ss = make([]complex128, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] != od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] != od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []complex128
			if same {
				ss = make([]complex128, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] != od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] != od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []complex128
			if same {
				ss = make([]complex128, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] != od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] != od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = neDDSameC128(td, od)
			} else {
				ret = neDDBoolsC128(td, od)
			}
		}

	case reflect.String:
		td := t.strings()
		od := other.strings()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			bs := make([]bool, t.Shape().TotalSize())
			ret = bs
			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				bs[k] = td[i] != od[j]
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			bs := make([]bool, t.Shape().TotalSize())
			ret = bs
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				bs[k] = td[i] != od[j]
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			bs := make([]bool, t.Shape().TotalSize())
			ret = bs
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				bs[k] = td[i] != od[j]
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			ret = neDDBoolsStr(td, od)
		}

	case reflect.UnsafePointer:
		td := t.unsafePointers()
		od := other.unsafePointers()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			bs := make([]bool, t.Shape().TotalSize())
			ret = bs
			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				bs[k] = td[i] != od[j]
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			bs := make([]bool, t.Shape().TotalSize())
			ret = bs
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				bs[k] = td[i] != od[j]
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			bs := make([]bool, t.Shape().TotalSize())
			ret = bs
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				bs[k] = td[i] != od[j]
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			ret = neDDBoolsUnsafePointer(td, od)
		}

	default:
		err = errors.Errorf(unsupportedDtype, t.t, "ne")
	}

	if err != nil {
		return
	}
	retVal.fromSlice(ret)
	retVal.fix()
	err = retVal.sanity()

	switch {
	case toReuse:
		copyDense(reuse, retVal)
		ReturnTensor(retVal)
		retVal = reuse
	case !safe:
		copyDense(t, retVal)
		ReturnTensor(retVal)
		retVal = t
	}
	return
}

/* Gt */

func (t *Dense) gtDD(other *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	reuse, safe, same, toReuse, err := prepBinaryDenseCmp(t, other, opts...)
	if err != nil {
		return nil, err
	}

	retVal = recycledDenseNoFix(t.t, t.Shape().Clone())
	var ret interface{} // slice of some sort
	switch t.t.Kind() {

	case reflect.Int:
		td := t.ints()
		od := other.ints()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []int
			if same {
				ss = make([]int, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] > od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] > od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int
			if same {
				ss = make([]int, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] > od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] > od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int
			if same {
				ss = make([]int, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] > od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] > od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = gtDDSameI(td, od)
			} else {
				ret = gtDDBoolsI(td, od)
			}
		}

	case reflect.Int8:
		td := t.int8s()
		od := other.int8s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []int8
			if same {
				ss = make([]int8, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] > od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] > od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int8
			if same {
				ss = make([]int8, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] > od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] > od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int8
			if same {
				ss = make([]int8, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] > od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] > od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = gtDDSameI8(td, od)
			} else {
				ret = gtDDBoolsI8(td, od)
			}
		}

	case reflect.Int16:
		td := t.int16s()
		od := other.int16s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []int16
			if same {
				ss = make([]int16, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] > od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] > od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int16
			if same {
				ss = make([]int16, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] > od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] > od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int16
			if same {
				ss = make([]int16, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] > od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] > od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = gtDDSameI16(td, od)
			} else {
				ret = gtDDBoolsI16(td, od)
			}
		}

	case reflect.Int32:
		td := t.int32s()
		od := other.int32s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []int32
			if same {
				ss = make([]int32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] > od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] > od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int32
			if same {
				ss = make([]int32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] > od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] > od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int32
			if same {
				ss = make([]int32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] > od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] > od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = gtDDSameI32(td, od)
			} else {
				ret = gtDDBoolsI32(td, od)
			}
		}

	case reflect.Int64:
		td := t.int64s()
		od := other.int64s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []int64
			if same {
				ss = make([]int64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] > od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] > od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int64
			if same {
				ss = make([]int64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] > od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] > od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int64
			if same {
				ss = make([]int64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] > od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] > od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = gtDDSameI64(td, od)
			} else {
				ret = gtDDBoolsI64(td, od)
			}
		}

	case reflect.Uint:
		td := t.uints()
		od := other.uints()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []uint
			if same {
				ss = make([]uint, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] > od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] > od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint
			if same {
				ss = make([]uint, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] > od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] > od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint
			if same {
				ss = make([]uint, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] > od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] > od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = gtDDSameU(td, od)
			} else {
				ret = gtDDBoolsU(td, od)
			}
		}

	case reflect.Uint8:
		td := t.uint8s()
		od := other.uint8s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []uint8
			if same {
				ss = make([]uint8, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] > od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] > od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint8
			if same {
				ss = make([]uint8, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] > od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] > od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint8
			if same {
				ss = make([]uint8, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] > od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] > od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = gtDDSameU8(td, od)
			} else {
				ret = gtDDBoolsU8(td, od)
			}
		}

	case reflect.Uint16:
		td := t.uint16s()
		od := other.uint16s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []uint16
			if same {
				ss = make([]uint16, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] > od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] > od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint16
			if same {
				ss = make([]uint16, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] > od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] > od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint16
			if same {
				ss = make([]uint16, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] > od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] > od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = gtDDSameU16(td, od)
			} else {
				ret = gtDDBoolsU16(td, od)
			}
		}

	case reflect.Uint32:
		td := t.uint32s()
		od := other.uint32s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []uint32
			if same {
				ss = make([]uint32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] > od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] > od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint32
			if same {
				ss = make([]uint32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] > od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] > od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint32
			if same {
				ss = make([]uint32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] > od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] > od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = gtDDSameU32(td, od)
			} else {
				ret = gtDDBoolsU32(td, od)
			}
		}

	case reflect.Uint64:
		td := t.uint64s()
		od := other.uint64s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []uint64
			if same {
				ss = make([]uint64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] > od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] > od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint64
			if same {
				ss = make([]uint64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] > od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] > od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint64
			if same {
				ss = make([]uint64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] > od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] > od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = gtDDSameU64(td, od)
			} else {
				ret = gtDDBoolsU64(td, od)
			}
		}

	case reflect.Float32:
		td := t.float32s()
		od := other.float32s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []float32
			if same {
				ss = make([]float32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] > od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] > od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []float32
			if same {
				ss = make([]float32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] > od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] > od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []float32
			if same {
				ss = make([]float32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] > od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] > od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = gtDDSameF32(td, od)
			} else {
				ret = gtDDBoolsF32(td, od)
			}
		}

	case reflect.Float64:
		td := t.float64s()
		od := other.float64s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []float64
			if same {
				ss = make([]float64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] > od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] > od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []float64
			if same {
				ss = make([]float64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] > od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] > od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []float64
			if same {
				ss = make([]float64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] > od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] > od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = gtDDSameF64(td, od)
			} else {
				ret = gtDDBoolsF64(td, od)
			}
		}

	case reflect.String:
		td := t.strings()
		od := other.strings()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			bs := make([]bool, t.Shape().TotalSize())
			ret = bs
			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				bs[k] = td[i] > od[j]
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			bs := make([]bool, t.Shape().TotalSize())
			ret = bs
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				bs[k] = td[i] > od[j]
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			bs := make([]bool, t.Shape().TotalSize())
			ret = bs
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				bs[k] = td[i] > od[j]
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			ret = gtDDBoolsStr(td, od)
		}

	default:
		err = errors.Errorf(unsupportedDtype, t.t, "gt")
	}

	if err != nil {
		return
	}
	retVal.fromSlice(ret)
	retVal.fix()
	err = retVal.sanity()

	switch {
	case toReuse:
		copyDense(reuse, retVal)
		ReturnTensor(retVal)
		retVal = reuse
	case !safe:
		copyDense(t, retVal)
		ReturnTensor(retVal)
		retVal = t
	}
	return
}

/* Gte */

func (t *Dense) gteDD(other *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	reuse, safe, same, toReuse, err := prepBinaryDenseCmp(t, other, opts...)
	if err != nil {
		return nil, err
	}

	retVal = recycledDenseNoFix(t.t, t.Shape().Clone())
	var ret interface{} // slice of some sort
	switch t.t.Kind() {

	case reflect.Int:
		td := t.ints()
		od := other.ints()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []int
			if same {
				ss = make([]int, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] >= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] >= od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int
			if same {
				ss = make([]int, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] >= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] >= od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int
			if same {
				ss = make([]int, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] >= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] >= od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = gteDDSameI(td, od)
			} else {
				ret = gteDDBoolsI(td, od)
			}
		}

	case reflect.Int8:
		td := t.int8s()
		od := other.int8s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []int8
			if same {
				ss = make([]int8, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] >= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] >= od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int8
			if same {
				ss = make([]int8, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] >= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] >= od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int8
			if same {
				ss = make([]int8, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] >= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] >= od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = gteDDSameI8(td, od)
			} else {
				ret = gteDDBoolsI8(td, od)
			}
		}

	case reflect.Int16:
		td := t.int16s()
		od := other.int16s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []int16
			if same {
				ss = make([]int16, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] >= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] >= od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int16
			if same {
				ss = make([]int16, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] >= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] >= od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int16
			if same {
				ss = make([]int16, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] >= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] >= od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = gteDDSameI16(td, od)
			} else {
				ret = gteDDBoolsI16(td, od)
			}
		}

	case reflect.Int32:
		td := t.int32s()
		od := other.int32s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []int32
			if same {
				ss = make([]int32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] >= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] >= od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int32
			if same {
				ss = make([]int32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] >= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] >= od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int32
			if same {
				ss = make([]int32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] >= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] >= od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = gteDDSameI32(td, od)
			} else {
				ret = gteDDBoolsI32(td, od)
			}
		}

	case reflect.Int64:
		td := t.int64s()
		od := other.int64s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []int64
			if same {
				ss = make([]int64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] >= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] >= od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int64
			if same {
				ss = make([]int64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] >= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] >= od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int64
			if same {
				ss = make([]int64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] >= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] >= od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = gteDDSameI64(td, od)
			} else {
				ret = gteDDBoolsI64(td, od)
			}
		}

	case reflect.Uint:
		td := t.uints()
		od := other.uints()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []uint
			if same {
				ss = make([]uint, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] >= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] >= od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint
			if same {
				ss = make([]uint, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] >= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] >= od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint
			if same {
				ss = make([]uint, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] >= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] >= od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = gteDDSameU(td, od)
			} else {
				ret = gteDDBoolsU(td, od)
			}
		}

	case reflect.Uint8:
		td := t.uint8s()
		od := other.uint8s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []uint8
			if same {
				ss = make([]uint8, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] >= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] >= od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint8
			if same {
				ss = make([]uint8, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] >= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] >= od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint8
			if same {
				ss = make([]uint8, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] >= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] >= od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = gteDDSameU8(td, od)
			} else {
				ret = gteDDBoolsU8(td, od)
			}
		}

	case reflect.Uint16:
		td := t.uint16s()
		od := other.uint16s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []uint16
			if same {
				ss = make([]uint16, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] >= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] >= od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint16
			if same {
				ss = make([]uint16, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] >= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] >= od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint16
			if same {
				ss = make([]uint16, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] >= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] >= od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = gteDDSameU16(td, od)
			} else {
				ret = gteDDBoolsU16(td, od)
			}
		}

	case reflect.Uint32:
		td := t.uint32s()
		od := other.uint32s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []uint32
			if same {
				ss = make([]uint32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] >= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] >= od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint32
			if same {
				ss = make([]uint32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] >= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] >= od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint32
			if same {
				ss = make([]uint32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] >= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] >= od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = gteDDSameU32(td, od)
			} else {
				ret = gteDDBoolsU32(td, od)
			}
		}

	case reflect.Uint64:
		td := t.uint64s()
		od := other.uint64s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []uint64
			if same {
				ss = make([]uint64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] >= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] >= od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint64
			if same {
				ss = make([]uint64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] >= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] >= od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint64
			if same {
				ss = make([]uint64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] >= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] >= od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = gteDDSameU64(td, od)
			} else {
				ret = gteDDBoolsU64(td, od)
			}
		}

	case reflect.Float32:
		td := t.float32s()
		od := other.float32s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []float32
			if same {
				ss = make([]float32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] >= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] >= od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []float32
			if same {
				ss = make([]float32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] >= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] >= od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []float32
			if same {
				ss = make([]float32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] >= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] >= od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = gteDDSameF32(td, od)
			} else {
				ret = gteDDBoolsF32(td, od)
			}
		}

	case reflect.Float64:
		td := t.float64s()
		od := other.float64s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []float64
			if same {
				ss = make([]float64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] >= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] >= od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []float64
			if same {
				ss = make([]float64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] >= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] >= od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []float64
			if same {
				ss = make([]float64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] >= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] >= od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = gteDDSameF64(td, od)
			} else {
				ret = gteDDBoolsF64(td, od)
			}
		}

	case reflect.String:
		td := t.strings()
		od := other.strings()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			bs := make([]bool, t.Shape().TotalSize())
			ret = bs
			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				bs[k] = td[i] >= od[j]
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			bs := make([]bool, t.Shape().TotalSize())
			ret = bs
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				bs[k] = td[i] >= od[j]
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			bs := make([]bool, t.Shape().TotalSize())
			ret = bs
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				bs[k] = td[i] >= od[j]
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			ret = gteDDBoolsStr(td, od)
		}

	default:
		err = errors.Errorf(unsupportedDtype, t.t, "gte")
	}

	if err != nil {
		return
	}
	retVal.fromSlice(ret)
	retVal.fix()
	err = retVal.sanity()

	switch {
	case toReuse:
		copyDense(reuse, retVal)
		ReturnTensor(retVal)
		retVal = reuse
	case !safe:
		copyDense(t, retVal)
		ReturnTensor(retVal)
		retVal = t
	}
	return
}

/* Lt */

func (t *Dense) ltDD(other *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	reuse, safe, same, toReuse, err := prepBinaryDenseCmp(t, other, opts...)
	if err != nil {
		return nil, err
	}

	retVal = recycledDenseNoFix(t.t, t.Shape().Clone())
	var ret interface{} // slice of some sort
	switch t.t.Kind() {

	case reflect.Int:
		td := t.ints()
		od := other.ints()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []int
			if same {
				ss = make([]int, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] < od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] < od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int
			if same {
				ss = make([]int, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] < od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] < od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int
			if same {
				ss = make([]int, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] < od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] < od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = ltDDSameI(td, od)
			} else {
				ret = ltDDBoolsI(td, od)
			}
		}

	case reflect.Int8:
		td := t.int8s()
		od := other.int8s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []int8
			if same {
				ss = make([]int8, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] < od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] < od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int8
			if same {
				ss = make([]int8, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] < od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] < od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int8
			if same {
				ss = make([]int8, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] < od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] < od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = ltDDSameI8(td, od)
			} else {
				ret = ltDDBoolsI8(td, od)
			}
		}

	case reflect.Int16:
		td := t.int16s()
		od := other.int16s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []int16
			if same {
				ss = make([]int16, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] < od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] < od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int16
			if same {
				ss = make([]int16, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] < od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] < od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int16
			if same {
				ss = make([]int16, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] < od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] < od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = ltDDSameI16(td, od)
			} else {
				ret = ltDDBoolsI16(td, od)
			}
		}

	case reflect.Int32:
		td := t.int32s()
		od := other.int32s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []int32
			if same {
				ss = make([]int32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] < od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] < od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int32
			if same {
				ss = make([]int32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] < od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] < od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int32
			if same {
				ss = make([]int32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] < od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] < od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = ltDDSameI32(td, od)
			} else {
				ret = ltDDBoolsI32(td, od)
			}
		}

	case reflect.Int64:
		td := t.int64s()
		od := other.int64s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []int64
			if same {
				ss = make([]int64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] < od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] < od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int64
			if same {
				ss = make([]int64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] < od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] < od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int64
			if same {
				ss = make([]int64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] < od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] < od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = ltDDSameI64(td, od)
			} else {
				ret = ltDDBoolsI64(td, od)
			}
		}

	case reflect.Uint:
		td := t.uints()
		od := other.uints()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []uint
			if same {
				ss = make([]uint, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] < od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] < od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint
			if same {
				ss = make([]uint, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] < od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] < od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint
			if same {
				ss = make([]uint, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] < od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] < od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = ltDDSameU(td, od)
			} else {
				ret = ltDDBoolsU(td, od)
			}
		}

	case reflect.Uint8:
		td := t.uint8s()
		od := other.uint8s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []uint8
			if same {
				ss = make([]uint8, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] < od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] < od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint8
			if same {
				ss = make([]uint8, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] < od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] < od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint8
			if same {
				ss = make([]uint8, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] < od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] < od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = ltDDSameU8(td, od)
			} else {
				ret = ltDDBoolsU8(td, od)
			}
		}

	case reflect.Uint16:
		td := t.uint16s()
		od := other.uint16s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []uint16
			if same {
				ss = make([]uint16, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] < od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] < od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint16
			if same {
				ss = make([]uint16, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] < od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] < od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint16
			if same {
				ss = make([]uint16, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] < od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] < od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = ltDDSameU16(td, od)
			} else {
				ret = ltDDBoolsU16(td, od)
			}
		}

	case reflect.Uint32:
		td := t.uint32s()
		od := other.uint32s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []uint32
			if same {
				ss = make([]uint32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] < od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] < od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint32
			if same {
				ss = make([]uint32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] < od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] < od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint32
			if same {
				ss = make([]uint32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] < od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] < od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = ltDDSameU32(td, od)
			} else {
				ret = ltDDBoolsU32(td, od)
			}
		}

	case reflect.Uint64:
		td := t.uint64s()
		od := other.uint64s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []uint64
			if same {
				ss = make([]uint64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] < od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] < od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint64
			if same {
				ss = make([]uint64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] < od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] < od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint64
			if same {
				ss = make([]uint64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] < od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] < od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = ltDDSameU64(td, od)
			} else {
				ret = ltDDBoolsU64(td, od)
			}
		}

	case reflect.Float32:
		td := t.float32s()
		od := other.float32s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []float32
			if same {
				ss = make([]float32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] < od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] < od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []float32
			if same {
				ss = make([]float32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] < od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] < od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []float32
			if same {
				ss = make([]float32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] < od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] < od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = ltDDSameF32(td, od)
			} else {
				ret = ltDDBoolsF32(td, od)
			}
		}

	case reflect.Float64:
		td := t.float64s()
		od := other.float64s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []float64
			if same {
				ss = make([]float64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] < od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] < od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []float64
			if same {
				ss = make([]float64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] < od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] < od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []float64
			if same {
				ss = make([]float64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] < od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] < od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = ltDDSameF64(td, od)
			} else {
				ret = ltDDBoolsF64(td, od)
			}
		}

	case reflect.String:
		td := t.strings()
		od := other.strings()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			bs := make([]bool, t.Shape().TotalSize())
			ret = bs
			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				bs[k] = td[i] < od[j]
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			bs := make([]bool, t.Shape().TotalSize())
			ret = bs
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				bs[k] = td[i] < od[j]
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			bs := make([]bool, t.Shape().TotalSize())
			ret = bs
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				bs[k] = td[i] < od[j]
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			ret = ltDDBoolsStr(td, od)
		}

	default:
		err = errors.Errorf(unsupportedDtype, t.t, "lt")
	}

	if err != nil {
		return
	}
	retVal.fromSlice(ret)
	retVal.fix()
	err = retVal.sanity()

	switch {
	case toReuse:
		copyDense(reuse, retVal)
		ReturnTensor(retVal)
		retVal = reuse
	case !safe:
		copyDense(t, retVal)
		ReturnTensor(retVal)
		retVal = t
	}
	return
}

/* Lte */

func (t *Dense) lteDD(other *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	reuse, safe, same, toReuse, err := prepBinaryDenseCmp(t, other, opts...)
	if err != nil {
		return nil, err
	}

	retVal = recycledDenseNoFix(t.t, t.Shape().Clone())
	var ret interface{} // slice of some sort
	switch t.t.Kind() {

	case reflect.Int:
		td := t.ints()
		od := other.ints()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []int
			if same {
				ss = make([]int, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] <= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] <= od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int
			if same {
				ss = make([]int, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] <= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] <= od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int
			if same {
				ss = make([]int, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] <= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] <= od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = lteDDSameI(td, od)
			} else {
				ret = lteDDBoolsI(td, od)
			}
		}

	case reflect.Int8:
		td := t.int8s()
		od := other.int8s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []int8
			if same {
				ss = make([]int8, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] <= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] <= od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int8
			if same {
				ss = make([]int8, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] <= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] <= od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int8
			if same {
				ss = make([]int8, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] <= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] <= od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = lteDDSameI8(td, od)
			} else {
				ret = lteDDBoolsI8(td, od)
			}
		}

	case reflect.Int16:
		td := t.int16s()
		od := other.int16s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []int16
			if same {
				ss = make([]int16, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] <= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] <= od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int16
			if same {
				ss = make([]int16, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] <= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] <= od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int16
			if same {
				ss = make([]int16, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] <= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] <= od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = lteDDSameI16(td, od)
			} else {
				ret = lteDDBoolsI16(td, od)
			}
		}

	case reflect.Int32:
		td := t.int32s()
		od := other.int32s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []int32
			if same {
				ss = make([]int32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] <= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] <= od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int32
			if same {
				ss = make([]int32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] <= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] <= od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int32
			if same {
				ss = make([]int32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] <= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] <= od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = lteDDSameI32(td, od)
			} else {
				ret = lteDDBoolsI32(td, od)
			}
		}

	case reflect.Int64:
		td := t.int64s()
		od := other.int64s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []int64
			if same {
				ss = make([]int64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] <= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] <= od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int64
			if same {
				ss = make([]int64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] <= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] <= od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []int64
			if same {
				ss = make([]int64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] <= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] <= od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = lteDDSameI64(td, od)
			} else {
				ret = lteDDBoolsI64(td, od)
			}
		}

	case reflect.Uint:
		td := t.uints()
		od := other.uints()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []uint
			if same {
				ss = make([]uint, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] <= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] <= od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint
			if same {
				ss = make([]uint, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] <= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] <= od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint
			if same {
				ss = make([]uint, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] <= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] <= od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = lteDDSameU(td, od)
			} else {
				ret = lteDDBoolsU(td, od)
			}
		}

	case reflect.Uint8:
		td := t.uint8s()
		od := other.uint8s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []uint8
			if same {
				ss = make([]uint8, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] <= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] <= od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint8
			if same {
				ss = make([]uint8, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] <= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] <= od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint8
			if same {
				ss = make([]uint8, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] <= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] <= od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = lteDDSameU8(td, od)
			} else {
				ret = lteDDBoolsU8(td, od)
			}
		}

	case reflect.Uint16:
		td := t.uint16s()
		od := other.uint16s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []uint16
			if same {
				ss = make([]uint16, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] <= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] <= od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint16
			if same {
				ss = make([]uint16, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] <= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] <= od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint16
			if same {
				ss = make([]uint16, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] <= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] <= od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = lteDDSameU16(td, od)
			} else {
				ret = lteDDBoolsU16(td, od)
			}
		}

	case reflect.Uint32:
		td := t.uint32s()
		od := other.uint32s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []uint32
			if same {
				ss = make([]uint32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] <= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] <= od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint32
			if same {
				ss = make([]uint32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] <= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] <= od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint32
			if same {
				ss = make([]uint32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] <= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] <= od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = lteDDSameU32(td, od)
			} else {
				ret = lteDDBoolsU32(td, od)
			}
		}

	case reflect.Uint64:
		td := t.uint64s()
		od := other.uint64s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []uint64
			if same {
				ss = make([]uint64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] <= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] <= od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint64
			if same {
				ss = make([]uint64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] <= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] <= od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []uint64
			if same {
				ss = make([]uint64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] <= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] <= od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = lteDDSameU64(td, od)
			} else {
				ret = lteDDBoolsU64(td, od)
			}
		}

	case reflect.Float32:
		td := t.float32s()
		od := other.float32s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []float32
			if same {
				ss = make([]float32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] <= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] <= od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []float32
			if same {
				ss = make([]float32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] <= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] <= od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []float32
			if same {
				ss = make([]float32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] <= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] <= od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = lteDDSameF32(td, od)
			} else {
				ret = lteDDBoolsF32(td, od)
			}
		}

	case reflect.Float64:
		td := t.float64s()
		od := other.float64s()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			var bs []bool
			var ss []float64
			if same {
				ss = make([]float64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if same {
					if td[i] <= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] <= od[j]
				}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var bs []bool
			var ss []float64
			if same {
				ss = make([]float64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if same {
					if td[i] <= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] <= od[j]
				}
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			var bs []bool
			var ss []float64
			if same {
				ss = make([]float64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if same {
					if td[i] <= od[j] {
						ss[k] = 1
					}
				} else {
					bs[k] = td[i] <= od[j]
				}
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			if same {
				ret = lteDDSameF64(td, od)
			} else {
				ret = lteDDBoolsF64(td, od)
			}
		}

	case reflect.String:
		td := t.strings()
		od := other.strings()
		var i, j, k int
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			bs := make([]bool, t.Shape().TotalSize())
			ret = bs
			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				bs[k] = td[i] <= od[j]
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			bs := make([]bool, t.Shape().TotalSize())
			ret = bs
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				bs[k] = td[i] <= od[j]
				j++
				k++
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			bs := make([]bool, t.Shape().TotalSize())
			ret = bs
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				bs[k] = td[i] <= od[j]
				i++
				k++
			}
			err = handleNoOp(err)
		default:
			ret = lteDDBoolsStr(td, od)
		}

	default:
		err = errors.Errorf(unsupportedDtype, t.t, "lte")
	}

	if err != nil {
		return
	}
	retVal.fromSlice(ret)
	retVal.fix()
	err = retVal.sanity()

	switch {
	case toReuse:
		copyDense(reuse, retVal)
		ReturnTensor(retVal)
		retVal = reuse
	case !safe:
		copyDense(t, retVal)
		ReturnTensor(retVal)
		retVal = t
	}
	return
}

/* Eq */

func (t *Dense) eqDS(other interface{}, opts ...FuncOpt) (retVal *Dense, err error) {
	reuse, safe, same, toReuse, err := prepUnaryDenseCmp(t, opts...)
	if err != nil {
		return nil, err
	}

	var ret interface{} // slice of some sort
	retVal = recycledDenseNoFix(t.t, t.Shape().Clone())
	switch t.t.Kind() {
	case reflect.Bool:
		data := t.bools()
		b := other.(bool)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			bs := make([]bool, t.Shape().TotalSize())
			ret = bs
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				bs[j] = data[i] == b
				j++
			}
		default:
			ret = eqDSBoolsB(data, b)
		}
	case reflect.Int:
		data := t.ints()
		b := other.(int)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []int
			if same {
				ss = make([]int, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] == b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] == b
				}
				j++
			}
		default:
			if same {
				ret = eqDSSameI(data, b)
			} else {
				ret = eqDSBoolsI(data, b)
			}
		}
	case reflect.Int8:
		data := t.int8s()
		b := other.(int8)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []int8
			if same {
				ss = make([]int8, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] == b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] == b
				}
				j++
			}
		default:
			if same {
				ret = eqDSSameI8(data, b)
			} else {
				ret = eqDSBoolsI8(data, b)
			}
		}
	case reflect.Int16:
		data := t.int16s()
		b := other.(int16)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []int16
			if same {
				ss = make([]int16, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] == b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] == b
				}
				j++
			}
		default:
			if same {
				ret = eqDSSameI16(data, b)
			} else {
				ret = eqDSBoolsI16(data, b)
			}
		}
	case reflect.Int32:
		data := t.int32s()
		b := other.(int32)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []int32
			if same {
				ss = make([]int32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] == b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] == b
				}
				j++
			}
		default:
			if same {
				ret = eqDSSameI32(data, b)
			} else {
				ret = eqDSBoolsI32(data, b)
			}
		}
	case reflect.Int64:
		data := t.int64s()
		b := other.(int64)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []int64
			if same {
				ss = make([]int64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] == b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] == b
				}
				j++
			}
		default:
			if same {
				ret = eqDSSameI64(data, b)
			} else {
				ret = eqDSBoolsI64(data, b)
			}
		}
	case reflect.Uint:
		data := t.uints()
		b := other.(uint)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []uint
			if same {
				ss = make([]uint, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] == b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] == b
				}
				j++
			}
		default:
			if same {
				ret = eqDSSameU(data, b)
			} else {
				ret = eqDSBoolsU(data, b)
			}
		}
	case reflect.Uint8:
		data := t.uint8s()
		b := other.(uint8)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []uint8
			if same {
				ss = make([]uint8, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] == b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] == b
				}
				j++
			}
		default:
			if same {
				ret = eqDSSameU8(data, b)
			} else {
				ret = eqDSBoolsU8(data, b)
			}
		}
	case reflect.Uint16:
		data := t.uint16s()
		b := other.(uint16)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []uint16
			if same {
				ss = make([]uint16, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] == b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] == b
				}
				j++
			}
		default:
			if same {
				ret = eqDSSameU16(data, b)
			} else {
				ret = eqDSBoolsU16(data, b)
			}
		}
	case reflect.Uint32:
		data := t.uint32s()
		b := other.(uint32)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []uint32
			if same {
				ss = make([]uint32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] == b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] == b
				}
				j++
			}
		default:
			if same {
				ret = eqDSSameU32(data, b)
			} else {
				ret = eqDSBoolsU32(data, b)
			}
		}
	case reflect.Uint64:
		data := t.uint64s()
		b := other.(uint64)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []uint64
			if same {
				ss = make([]uint64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] == b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] == b
				}
				j++
			}
		default:
			if same {
				ret = eqDSSameU64(data, b)
			} else {
				ret = eqDSBoolsU64(data, b)
			}
		}
	case reflect.Uintptr:
		data := t.uintptrs()
		b := other.(uintptr)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			bs := make([]bool, t.Shape().TotalSize())
			ret = bs
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				bs[j] = data[i] == b
				j++
			}
		default:
			ret = eqDSBoolsUintptr(data, b)
		}
	case reflect.Float32:
		data := t.float32s()
		b := other.(float32)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []float32
			if same {
				ss = make([]float32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] == b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] == b
				}
				j++
			}
		default:
			if same {
				ret = eqDSSameF32(data, b)
			} else {
				ret = eqDSBoolsF32(data, b)
			}
		}
	case reflect.Float64:
		data := t.float64s()
		b := other.(float64)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []float64
			if same {
				ss = make([]float64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] == b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] == b
				}
				j++
			}
		default:
			if same {
				ret = eqDSSameF64(data, b)
			} else {
				ret = eqDSBoolsF64(data, b)
			}
		}
	case reflect.Complex64:
		data := t.complex64s()
		b := other.(complex64)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []complex64
			if same {
				ss = make([]complex64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] == b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] == b
				}
				j++
			}
		default:
			if same {
				ret = eqDSSameC64(data, b)
			} else {
				ret = eqDSBoolsC64(data, b)
			}
		}
	case reflect.Complex128:
		data := t.complex128s()
		b := other.(complex128)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []complex128
			if same {
				ss = make([]complex128, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] == b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] == b
				}
				j++
			}
		default:
			if same {
				ret = eqDSSameC128(data, b)
			} else {
				ret = eqDSBoolsC128(data, b)
			}
		}
	case reflect.String:
		data := t.strings()
		b := other.(string)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			bs := make([]bool, t.Shape().TotalSize())
			ret = bs
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				bs[j] = data[i] == b
				j++
			}
		default:
			ret = eqDSBoolsStr(data, b)
		}
	case reflect.UnsafePointer:
		data := t.unsafePointers()
		b := other.(unsafe.Pointer)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			bs := make([]bool, t.Shape().TotalSize())
			ret = bs
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				bs[j] = data[i] == b
				j++
			}
		default:
			ret = eqDSBoolsUnsafePointer(data, b)
		}
	default:
		err = errors.Errorf(unsupportedDtype, t.t, "eq")
		return
	}
	retVal.fromSlice(ret)
	retVal.fix()
	err = retVal.sanity()

	switch {
	case toReuse:
		copyDense(reuse, retVal)
		ReturnTensor(retVal)
		retVal = reuse
	case !safe:
		copyDense(t, retVal)
		ReturnTensor(retVal)
		retVal = t
	}
	return
}

/* Ne */

func (t *Dense) neDS(other interface{}, opts ...FuncOpt) (retVal *Dense, err error) {
	reuse, safe, same, toReuse, err := prepUnaryDenseCmp(t, opts...)
	if err != nil {
		return nil, err
	}

	var ret interface{} // slice of some sort
	retVal = recycledDenseNoFix(t.t, t.Shape().Clone())
	switch t.t.Kind() {
	case reflect.Bool:
		data := t.bools()
		b := other.(bool)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			bs := make([]bool, t.Shape().TotalSize())
			ret = bs
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				bs[j] = data[i] != b
				j++
			}
		default:
			ret = neDSBoolsB(data, b)
		}
	case reflect.Int:
		data := t.ints()
		b := other.(int)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []int
			if same {
				ss = make([]int, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] != b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] != b
				}
				j++
			}
		default:
			if same {
				ret = neDSSameI(data, b)
			} else {
				ret = neDSBoolsI(data, b)
			}
		}
	case reflect.Int8:
		data := t.int8s()
		b := other.(int8)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []int8
			if same {
				ss = make([]int8, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] != b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] != b
				}
				j++
			}
		default:
			if same {
				ret = neDSSameI8(data, b)
			} else {
				ret = neDSBoolsI8(data, b)
			}
		}
	case reflect.Int16:
		data := t.int16s()
		b := other.(int16)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []int16
			if same {
				ss = make([]int16, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] != b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] != b
				}
				j++
			}
		default:
			if same {
				ret = neDSSameI16(data, b)
			} else {
				ret = neDSBoolsI16(data, b)
			}
		}
	case reflect.Int32:
		data := t.int32s()
		b := other.(int32)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []int32
			if same {
				ss = make([]int32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] != b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] != b
				}
				j++
			}
		default:
			if same {
				ret = neDSSameI32(data, b)
			} else {
				ret = neDSBoolsI32(data, b)
			}
		}
	case reflect.Int64:
		data := t.int64s()
		b := other.(int64)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []int64
			if same {
				ss = make([]int64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] != b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] != b
				}
				j++
			}
		default:
			if same {
				ret = neDSSameI64(data, b)
			} else {
				ret = neDSBoolsI64(data, b)
			}
		}
	case reflect.Uint:
		data := t.uints()
		b := other.(uint)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []uint
			if same {
				ss = make([]uint, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] != b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] != b
				}
				j++
			}
		default:
			if same {
				ret = neDSSameU(data, b)
			} else {
				ret = neDSBoolsU(data, b)
			}
		}
	case reflect.Uint8:
		data := t.uint8s()
		b := other.(uint8)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []uint8
			if same {
				ss = make([]uint8, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] != b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] != b
				}
				j++
			}
		default:
			if same {
				ret = neDSSameU8(data, b)
			} else {
				ret = neDSBoolsU8(data, b)
			}
		}
	case reflect.Uint16:
		data := t.uint16s()
		b := other.(uint16)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []uint16
			if same {
				ss = make([]uint16, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] != b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] != b
				}
				j++
			}
		default:
			if same {
				ret = neDSSameU16(data, b)
			} else {
				ret = neDSBoolsU16(data, b)
			}
		}
	case reflect.Uint32:
		data := t.uint32s()
		b := other.(uint32)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []uint32
			if same {
				ss = make([]uint32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] != b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] != b
				}
				j++
			}
		default:
			if same {
				ret = neDSSameU32(data, b)
			} else {
				ret = neDSBoolsU32(data, b)
			}
		}
	case reflect.Uint64:
		data := t.uint64s()
		b := other.(uint64)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []uint64
			if same {
				ss = make([]uint64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] != b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] != b
				}
				j++
			}
		default:
			if same {
				ret = neDSSameU64(data, b)
			} else {
				ret = neDSBoolsU64(data, b)
			}
		}
	case reflect.Uintptr:
		data := t.uintptrs()
		b := other.(uintptr)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			bs := make([]bool, t.Shape().TotalSize())
			ret = bs
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				bs[j] = data[i] != b
				j++
			}
		default:
			ret = neDSBoolsUintptr(data, b)
		}
	case reflect.Float32:
		data := t.float32s()
		b := other.(float32)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []float32
			if same {
				ss = make([]float32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] != b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] != b
				}
				j++
			}
		default:
			if same {
				ret = neDSSameF32(data, b)
			} else {
				ret = neDSBoolsF32(data, b)
			}
		}
	case reflect.Float64:
		data := t.float64s()
		b := other.(float64)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []float64
			if same {
				ss = make([]float64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] != b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] != b
				}
				j++
			}
		default:
			if same {
				ret = neDSSameF64(data, b)
			} else {
				ret = neDSBoolsF64(data, b)
			}
		}
	case reflect.Complex64:
		data := t.complex64s()
		b := other.(complex64)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []complex64
			if same {
				ss = make([]complex64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] != b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] != b
				}
				j++
			}
		default:
			if same {
				ret = neDSSameC64(data, b)
			} else {
				ret = neDSBoolsC64(data, b)
			}
		}
	case reflect.Complex128:
		data := t.complex128s()
		b := other.(complex128)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []complex128
			if same {
				ss = make([]complex128, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] != b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] != b
				}
				j++
			}
		default:
			if same {
				ret = neDSSameC128(data, b)
			} else {
				ret = neDSBoolsC128(data, b)
			}
		}
	case reflect.String:
		data := t.strings()
		b := other.(string)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			bs := make([]bool, t.Shape().TotalSize())
			ret = bs
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				bs[j] = data[i] != b
				j++
			}
		default:
			ret = neDSBoolsStr(data, b)
		}
	case reflect.UnsafePointer:
		data := t.unsafePointers()
		b := other.(unsafe.Pointer)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			bs := make([]bool, t.Shape().TotalSize())
			ret = bs
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				bs[j] = data[i] != b
				j++
			}
		default:
			ret = neDSBoolsUnsafePointer(data, b)
		}
	default:
		err = errors.Errorf(unsupportedDtype, t.t, "ne")
		return
	}
	retVal.fromSlice(ret)
	retVal.fix()
	err = retVal.sanity()

	switch {
	case toReuse:
		copyDense(reuse, retVal)
		ReturnTensor(retVal)
		retVal = reuse
	case !safe:
		copyDense(t, retVal)
		ReturnTensor(retVal)
		retVal = t
	}
	return
}

/* Gt */

func (t *Dense) gtDS(other interface{}, opts ...FuncOpt) (retVal *Dense, err error) {
	reuse, safe, same, toReuse, err := prepUnaryDenseCmp(t, opts...)
	if err != nil {
		return nil, err
	}

	var ret interface{} // slice of some sort
	retVal = recycledDenseNoFix(t.t, t.Shape().Clone())
	switch t.t.Kind() {
	case reflect.Int:
		data := t.ints()
		b := other.(int)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []int
			if same {
				ss = make([]int, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] > b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] > b
				}
				j++
			}
		default:
			if same {
				ret = gtDSSameI(data, b)
			} else {
				ret = gtDSBoolsI(data, b)
			}
		}
	case reflect.Int8:
		data := t.int8s()
		b := other.(int8)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []int8
			if same {
				ss = make([]int8, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] > b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] > b
				}
				j++
			}
		default:
			if same {
				ret = gtDSSameI8(data, b)
			} else {
				ret = gtDSBoolsI8(data, b)
			}
		}
	case reflect.Int16:
		data := t.int16s()
		b := other.(int16)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []int16
			if same {
				ss = make([]int16, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] > b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] > b
				}
				j++
			}
		default:
			if same {
				ret = gtDSSameI16(data, b)
			} else {
				ret = gtDSBoolsI16(data, b)
			}
		}
	case reflect.Int32:
		data := t.int32s()
		b := other.(int32)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []int32
			if same {
				ss = make([]int32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] > b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] > b
				}
				j++
			}
		default:
			if same {
				ret = gtDSSameI32(data, b)
			} else {
				ret = gtDSBoolsI32(data, b)
			}
		}
	case reflect.Int64:
		data := t.int64s()
		b := other.(int64)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []int64
			if same {
				ss = make([]int64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] > b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] > b
				}
				j++
			}
		default:
			if same {
				ret = gtDSSameI64(data, b)
			} else {
				ret = gtDSBoolsI64(data, b)
			}
		}
	case reflect.Uint:
		data := t.uints()
		b := other.(uint)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []uint
			if same {
				ss = make([]uint, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] > b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] > b
				}
				j++
			}
		default:
			if same {
				ret = gtDSSameU(data, b)
			} else {
				ret = gtDSBoolsU(data, b)
			}
		}
	case reflect.Uint8:
		data := t.uint8s()
		b := other.(uint8)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []uint8
			if same {
				ss = make([]uint8, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] > b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] > b
				}
				j++
			}
		default:
			if same {
				ret = gtDSSameU8(data, b)
			} else {
				ret = gtDSBoolsU8(data, b)
			}
		}
	case reflect.Uint16:
		data := t.uint16s()
		b := other.(uint16)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []uint16
			if same {
				ss = make([]uint16, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] > b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] > b
				}
				j++
			}
		default:
			if same {
				ret = gtDSSameU16(data, b)
			} else {
				ret = gtDSBoolsU16(data, b)
			}
		}
	case reflect.Uint32:
		data := t.uint32s()
		b := other.(uint32)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []uint32
			if same {
				ss = make([]uint32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] > b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] > b
				}
				j++
			}
		default:
			if same {
				ret = gtDSSameU32(data, b)
			} else {
				ret = gtDSBoolsU32(data, b)
			}
		}
	case reflect.Uint64:
		data := t.uint64s()
		b := other.(uint64)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []uint64
			if same {
				ss = make([]uint64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] > b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] > b
				}
				j++
			}
		default:
			if same {
				ret = gtDSSameU64(data, b)
			} else {
				ret = gtDSBoolsU64(data, b)
			}
		}
	case reflect.Float32:
		data := t.float32s()
		b := other.(float32)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []float32
			if same {
				ss = make([]float32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] > b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] > b
				}
				j++
			}
		default:
			if same {
				ret = gtDSSameF32(data, b)
			} else {
				ret = gtDSBoolsF32(data, b)
			}
		}
	case reflect.Float64:
		data := t.float64s()
		b := other.(float64)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []float64
			if same {
				ss = make([]float64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] > b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] > b
				}
				j++
			}
		default:
			if same {
				ret = gtDSSameF64(data, b)
			} else {
				ret = gtDSBoolsF64(data, b)
			}
		}
	case reflect.String:
		data := t.strings()
		b := other.(string)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			bs := make([]bool, t.Shape().TotalSize())
			ret = bs
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				bs[j] = data[i] > b
				j++
			}
		default:
			ret = gtDSBoolsStr(data, b)
		}
	default:
		err = errors.Errorf(unsupportedDtype, t.t, "gt")
		return
	}
	retVal.fromSlice(ret)
	retVal.fix()
	err = retVal.sanity()

	switch {
	case toReuse:
		copyDense(reuse, retVal)
		ReturnTensor(retVal)
		retVal = reuse
	case !safe:
		copyDense(t, retVal)
		ReturnTensor(retVal)
		retVal = t
	}
	return
}

/* Gte */

func (t *Dense) gteDS(other interface{}, opts ...FuncOpt) (retVal *Dense, err error) {
	reuse, safe, same, toReuse, err := prepUnaryDenseCmp(t, opts...)
	if err != nil {
		return nil, err
	}

	var ret interface{} // slice of some sort
	retVal = recycledDenseNoFix(t.t, t.Shape().Clone())
	switch t.t.Kind() {
	case reflect.Int:
		data := t.ints()
		b := other.(int)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []int
			if same {
				ss = make([]int, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] >= b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] >= b
				}
				j++
			}
		default:
			if same {
				ret = gteDSSameI(data, b)
			} else {
				ret = gteDSBoolsI(data, b)
			}
		}
	case reflect.Int8:
		data := t.int8s()
		b := other.(int8)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []int8
			if same {
				ss = make([]int8, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] >= b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] >= b
				}
				j++
			}
		default:
			if same {
				ret = gteDSSameI8(data, b)
			} else {
				ret = gteDSBoolsI8(data, b)
			}
		}
	case reflect.Int16:
		data := t.int16s()
		b := other.(int16)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []int16
			if same {
				ss = make([]int16, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] >= b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] >= b
				}
				j++
			}
		default:
			if same {
				ret = gteDSSameI16(data, b)
			} else {
				ret = gteDSBoolsI16(data, b)
			}
		}
	case reflect.Int32:
		data := t.int32s()
		b := other.(int32)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []int32
			if same {
				ss = make([]int32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] >= b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] >= b
				}
				j++
			}
		default:
			if same {
				ret = gteDSSameI32(data, b)
			} else {
				ret = gteDSBoolsI32(data, b)
			}
		}
	case reflect.Int64:
		data := t.int64s()
		b := other.(int64)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []int64
			if same {
				ss = make([]int64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] >= b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] >= b
				}
				j++
			}
		default:
			if same {
				ret = gteDSSameI64(data, b)
			} else {
				ret = gteDSBoolsI64(data, b)
			}
		}
	case reflect.Uint:
		data := t.uints()
		b := other.(uint)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []uint
			if same {
				ss = make([]uint, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] >= b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] >= b
				}
				j++
			}
		default:
			if same {
				ret = gteDSSameU(data, b)
			} else {
				ret = gteDSBoolsU(data, b)
			}
		}
	case reflect.Uint8:
		data := t.uint8s()
		b := other.(uint8)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []uint8
			if same {
				ss = make([]uint8, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] >= b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] >= b
				}
				j++
			}
		default:
			if same {
				ret = gteDSSameU8(data, b)
			} else {
				ret = gteDSBoolsU8(data, b)
			}
		}
	case reflect.Uint16:
		data := t.uint16s()
		b := other.(uint16)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []uint16
			if same {
				ss = make([]uint16, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] >= b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] >= b
				}
				j++
			}
		default:
			if same {
				ret = gteDSSameU16(data, b)
			} else {
				ret = gteDSBoolsU16(data, b)
			}
		}
	case reflect.Uint32:
		data := t.uint32s()
		b := other.(uint32)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []uint32
			if same {
				ss = make([]uint32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] >= b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] >= b
				}
				j++
			}
		default:
			if same {
				ret = gteDSSameU32(data, b)
			} else {
				ret = gteDSBoolsU32(data, b)
			}
		}
	case reflect.Uint64:
		data := t.uint64s()
		b := other.(uint64)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []uint64
			if same {
				ss = make([]uint64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] >= b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] >= b
				}
				j++
			}
		default:
			if same {
				ret = gteDSSameU64(data, b)
			} else {
				ret = gteDSBoolsU64(data, b)
			}
		}
	case reflect.Float32:
		data := t.float32s()
		b := other.(float32)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []float32
			if same {
				ss = make([]float32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] >= b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] >= b
				}
				j++
			}
		default:
			if same {
				ret = gteDSSameF32(data, b)
			} else {
				ret = gteDSBoolsF32(data, b)
			}
		}
	case reflect.Float64:
		data := t.float64s()
		b := other.(float64)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []float64
			if same {
				ss = make([]float64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] >= b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] >= b
				}
				j++
			}
		default:
			if same {
				ret = gteDSSameF64(data, b)
			} else {
				ret = gteDSBoolsF64(data, b)
			}
		}
	case reflect.String:
		data := t.strings()
		b := other.(string)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			bs := make([]bool, t.Shape().TotalSize())
			ret = bs
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				bs[j] = data[i] >= b
				j++
			}
		default:
			ret = gteDSBoolsStr(data, b)
		}
	default:
		err = errors.Errorf(unsupportedDtype, t.t, "gte")
		return
	}
	retVal.fromSlice(ret)
	retVal.fix()
	err = retVal.sanity()

	switch {
	case toReuse:
		copyDense(reuse, retVal)
		ReturnTensor(retVal)
		retVal = reuse
	case !safe:
		copyDense(t, retVal)
		ReturnTensor(retVal)
		retVal = t
	}
	return
}

/* Lt */

func (t *Dense) ltDS(other interface{}, opts ...FuncOpt) (retVal *Dense, err error) {
	reuse, safe, same, toReuse, err := prepUnaryDenseCmp(t, opts...)
	if err != nil {
		return nil, err
	}

	var ret interface{} // slice of some sort
	retVal = recycledDenseNoFix(t.t, t.Shape().Clone())
	switch t.t.Kind() {
	case reflect.Int:
		data := t.ints()
		b := other.(int)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []int
			if same {
				ss = make([]int, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] < b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] < b
				}
				j++
			}
		default:
			if same {
				ret = ltDSSameI(data, b)
			} else {
				ret = ltDSBoolsI(data, b)
			}
		}
	case reflect.Int8:
		data := t.int8s()
		b := other.(int8)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []int8
			if same {
				ss = make([]int8, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] < b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] < b
				}
				j++
			}
		default:
			if same {
				ret = ltDSSameI8(data, b)
			} else {
				ret = ltDSBoolsI8(data, b)
			}
		}
	case reflect.Int16:
		data := t.int16s()
		b := other.(int16)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []int16
			if same {
				ss = make([]int16, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] < b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] < b
				}
				j++
			}
		default:
			if same {
				ret = ltDSSameI16(data, b)
			} else {
				ret = ltDSBoolsI16(data, b)
			}
		}
	case reflect.Int32:
		data := t.int32s()
		b := other.(int32)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []int32
			if same {
				ss = make([]int32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] < b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] < b
				}
				j++
			}
		default:
			if same {
				ret = ltDSSameI32(data, b)
			} else {
				ret = ltDSBoolsI32(data, b)
			}
		}
	case reflect.Int64:
		data := t.int64s()
		b := other.(int64)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []int64
			if same {
				ss = make([]int64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] < b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] < b
				}
				j++
			}
		default:
			if same {
				ret = ltDSSameI64(data, b)
			} else {
				ret = ltDSBoolsI64(data, b)
			}
		}
	case reflect.Uint:
		data := t.uints()
		b := other.(uint)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []uint
			if same {
				ss = make([]uint, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] < b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] < b
				}
				j++
			}
		default:
			if same {
				ret = ltDSSameU(data, b)
			} else {
				ret = ltDSBoolsU(data, b)
			}
		}
	case reflect.Uint8:
		data := t.uint8s()
		b := other.(uint8)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []uint8
			if same {
				ss = make([]uint8, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] < b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] < b
				}
				j++
			}
		default:
			if same {
				ret = ltDSSameU8(data, b)
			} else {
				ret = ltDSBoolsU8(data, b)
			}
		}
	case reflect.Uint16:
		data := t.uint16s()
		b := other.(uint16)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []uint16
			if same {
				ss = make([]uint16, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] < b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] < b
				}
				j++
			}
		default:
			if same {
				ret = ltDSSameU16(data, b)
			} else {
				ret = ltDSBoolsU16(data, b)
			}
		}
	case reflect.Uint32:
		data := t.uint32s()
		b := other.(uint32)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []uint32
			if same {
				ss = make([]uint32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] < b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] < b
				}
				j++
			}
		default:
			if same {
				ret = ltDSSameU32(data, b)
			} else {
				ret = ltDSBoolsU32(data, b)
			}
		}
	case reflect.Uint64:
		data := t.uint64s()
		b := other.(uint64)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []uint64
			if same {
				ss = make([]uint64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] < b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] < b
				}
				j++
			}
		default:
			if same {
				ret = ltDSSameU64(data, b)
			} else {
				ret = ltDSBoolsU64(data, b)
			}
		}
	case reflect.Float32:
		data := t.float32s()
		b := other.(float32)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []float32
			if same {
				ss = make([]float32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] < b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] < b
				}
				j++
			}
		default:
			if same {
				ret = ltDSSameF32(data, b)
			} else {
				ret = ltDSBoolsF32(data, b)
			}
		}
	case reflect.Float64:
		data := t.float64s()
		b := other.(float64)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []float64
			if same {
				ss = make([]float64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] < b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] < b
				}
				j++
			}
		default:
			if same {
				ret = ltDSSameF64(data, b)
			} else {
				ret = ltDSBoolsF64(data, b)
			}
		}
	case reflect.String:
		data := t.strings()
		b := other.(string)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			bs := make([]bool, t.Shape().TotalSize())
			ret = bs
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				bs[j] = data[i] < b
				j++
			}
		default:
			ret = ltDSBoolsStr(data, b)
		}
	default:
		err = errors.Errorf(unsupportedDtype, t.t, "lt")
		return
	}
	retVal.fromSlice(ret)
	retVal.fix()
	err = retVal.sanity()

	switch {
	case toReuse:
		copyDense(reuse, retVal)
		ReturnTensor(retVal)
		retVal = reuse
	case !safe:
		copyDense(t, retVal)
		ReturnTensor(retVal)
		retVal = t
	}
	return
}

/* Lte */

func (t *Dense) lteDS(other interface{}, opts ...FuncOpt) (retVal *Dense, err error) {
	reuse, safe, same, toReuse, err := prepUnaryDenseCmp(t, opts...)
	if err != nil {
		return nil, err
	}

	var ret interface{} // slice of some sort
	retVal = recycledDenseNoFix(t.t, t.Shape().Clone())
	switch t.t.Kind() {
	case reflect.Int:
		data := t.ints()
		b := other.(int)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []int
			if same {
				ss = make([]int, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] <= b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] <= b
				}
				j++
			}
		default:
			if same {
				ret = lteDSSameI(data, b)
			} else {
				ret = lteDSBoolsI(data, b)
			}
		}
	case reflect.Int8:
		data := t.int8s()
		b := other.(int8)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []int8
			if same {
				ss = make([]int8, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] <= b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] <= b
				}
				j++
			}
		default:
			if same {
				ret = lteDSSameI8(data, b)
			} else {
				ret = lteDSBoolsI8(data, b)
			}
		}
	case reflect.Int16:
		data := t.int16s()
		b := other.(int16)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []int16
			if same {
				ss = make([]int16, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] <= b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] <= b
				}
				j++
			}
		default:
			if same {
				ret = lteDSSameI16(data, b)
			} else {
				ret = lteDSBoolsI16(data, b)
			}
		}
	case reflect.Int32:
		data := t.int32s()
		b := other.(int32)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []int32
			if same {
				ss = make([]int32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] <= b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] <= b
				}
				j++
			}
		default:
			if same {
				ret = lteDSSameI32(data, b)
			} else {
				ret = lteDSBoolsI32(data, b)
			}
		}
	case reflect.Int64:
		data := t.int64s()
		b := other.(int64)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []int64
			if same {
				ss = make([]int64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] <= b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] <= b
				}
				j++
			}
		default:
			if same {
				ret = lteDSSameI64(data, b)
			} else {
				ret = lteDSBoolsI64(data, b)
			}
		}
	case reflect.Uint:
		data := t.uints()
		b := other.(uint)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []uint
			if same {
				ss = make([]uint, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] <= b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] <= b
				}
				j++
			}
		default:
			if same {
				ret = lteDSSameU(data, b)
			} else {
				ret = lteDSBoolsU(data, b)
			}
		}
	case reflect.Uint8:
		data := t.uint8s()
		b := other.(uint8)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []uint8
			if same {
				ss = make([]uint8, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] <= b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] <= b
				}
				j++
			}
		default:
			if same {
				ret = lteDSSameU8(data, b)
			} else {
				ret = lteDSBoolsU8(data, b)
			}
		}
	case reflect.Uint16:
		data := t.uint16s()
		b := other.(uint16)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []uint16
			if same {
				ss = make([]uint16, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] <= b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] <= b
				}
				j++
			}
		default:
			if same {
				ret = lteDSSameU16(data, b)
			} else {
				ret = lteDSBoolsU16(data, b)
			}
		}
	case reflect.Uint32:
		data := t.uint32s()
		b := other.(uint32)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []uint32
			if same {
				ss = make([]uint32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] <= b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] <= b
				}
				j++
			}
		default:
			if same {
				ret = lteDSSameU32(data, b)
			} else {
				ret = lteDSBoolsU32(data, b)
			}
		}
	case reflect.Uint64:
		data := t.uint64s()
		b := other.(uint64)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []uint64
			if same {
				ss = make([]uint64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] <= b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] <= b
				}
				j++
			}
		default:
			if same {
				ret = lteDSSameU64(data, b)
			} else {
				ret = lteDSBoolsU64(data, b)
			}
		}
	case reflect.Float32:
		data := t.float32s()
		b := other.(float32)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []float32
			if same {
				ss = make([]float32, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] <= b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] <= b
				}
				j++
			}
		default:
			if same {
				ret = lteDSSameF32(data, b)
			} else {
				ret = lteDSBoolsF32(data, b)
			}
		}
	case reflect.Float64:
		data := t.float64s()
		b := other.(float64)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			var bs []bool
			var ss []float64
			if same {
				ss = make([]float64, t.Shape().TotalSize())
				ret = ss
			} else {
				bs = make([]bool, t.Shape().TotalSize())
				ret = bs
			}

			for i, err = it.Next(); err == nil; i, err = it.Next() {

				if same {
					if data[i] <= b {
						ss[j] = 1
					}
				} else {
					bs[j] = data[i] <= b
				}
				j++
			}
		default:
			if same {
				ret = lteDSSameF64(data, b)
			} else {
				ret = lteDSBoolsF64(data, b)
			}
		}
	case reflect.String:
		data := t.strings()
		b := other.(string)
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int
			bs := make([]bool, t.Shape().TotalSize())
			ret = bs
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				bs[j] = data[i] <= b
				j++
			}
		default:
			ret = lteDSBoolsStr(data, b)
		}
	default:
		err = errors.Errorf(unsupportedDtype, t.t, "lte")
		return
	}
	retVal.fromSlice(ret)
	retVal.fix()
	err = retVal.sanity()

	switch {
	case toReuse:
		copyDense(reuse, retVal)
		ReturnTensor(retVal)
		retVal = reuse
	case !safe:
		copyDense(t, retVal)
		ReturnTensor(retVal)
		retVal = t
	}
	return
}
