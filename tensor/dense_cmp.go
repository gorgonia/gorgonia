package tensor

import (
	"reflect"

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
	fo := parseFuncOpts(opts...)
	reuseT, _ := fo.incrReuse()
	safe = fo.safe()
	same = fo.same
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
	fo := parseFuncOpts(opts...)
	reuseT, _ := fo.incrReuse()
	safe = fo.safe()
	same = fo.same
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
	switch t.t.Kind() {
	case reflect.Bool:
		td := t.bools()
		od := other.bools()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Int:
		td := t.ints()
		od := other.ints()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Int8:
		td := t.int8s()
		od := other.int8s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Int16:
		td := t.int16s()
		od := other.int16s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Int32:
		td := t.int32s()
		od := other.int32s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Int64:
		td := t.int64s()
		od := other.int64s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Uint:
		td := t.uints()
		od := other.uints()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Uint8:
		td := t.uint8s()
		od := other.uint8s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Uint16:
		td := t.uint16s()
		od := other.uint16s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Uint32:
		td := t.uint32s()
		od := other.uint32s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Uint64:
		td := t.uint64s()
		od := other.uint64s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Uintptr:
		td := t.uintptrs()
		od := other.uintptrs()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Float32:
		td := t.float32s()
		od := other.float32s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Float64:
		td := t.float64s()
		od := other.float64s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Complex64:
		td := t.complex64s()
		od := other.complex64s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Complex128:
		td := t.complex128s()
		od := other.complex128s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.String:
		td := t.strings()
		od := other.strings()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.UnsafePointer:
		td := t.unsafePointers()
		od := other.unsafePointers()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	default:
		err = errors.Errorf(unsupportedDtype, t.t, "eq")
		return
	}

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
	switch t.t.Kind() {
	case reflect.Int:
		td := t.ints()
		od := other.ints()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Int8:
		td := t.int8s()
		od := other.int8s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Int16:
		td := t.int16s()
		od := other.int16s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Int32:
		td := t.int32s()
		od := other.int32s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Int64:
		td := t.int64s()
		od := other.int64s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Uint:
		td := t.uints()
		od := other.uints()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Uint8:
		td := t.uint8s()
		od := other.uint8s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Uint16:
		td := t.uint16s()
		od := other.uint16s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Uint32:
		td := t.uint32s()
		od := other.uint32s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Uint64:
		td := t.uint64s()
		od := other.uint64s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Uintptr:
		td := t.uintptrs()
		od := other.uintptrs()
		var i, j, k int
		var ret interface{} // slice of some sort
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
			ret = gtDDBoolsUintptr(td, od)
		}
		retVal.fromSlice(ret)

	case reflect.Float32:
		td := t.float32s()
		od := other.float32s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Float64:
		td := t.float64s()
		od := other.float64s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.String:
		td := t.strings()
		od := other.strings()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	default:
		err = errors.Errorf(unsupportedDtype, t.t, "gt")
		return
	}

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
	switch t.t.Kind() {
	case reflect.Int:
		td := t.ints()
		od := other.ints()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Int8:
		td := t.int8s()
		od := other.int8s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Int16:
		td := t.int16s()
		od := other.int16s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Int32:
		td := t.int32s()
		od := other.int32s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Int64:
		td := t.int64s()
		od := other.int64s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Uint:
		td := t.uints()
		od := other.uints()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Uint8:
		td := t.uint8s()
		od := other.uint8s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Uint16:
		td := t.uint16s()
		od := other.uint16s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Uint32:
		td := t.uint32s()
		od := other.uint32s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Uint64:
		td := t.uint64s()
		od := other.uint64s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Uintptr:
		td := t.uintptrs()
		od := other.uintptrs()
		var i, j, k int
		var ret interface{} // slice of some sort
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
			ret = gteDDBoolsUintptr(td, od)
		}
		retVal.fromSlice(ret)

	case reflect.Float32:
		td := t.float32s()
		od := other.float32s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Float64:
		td := t.float64s()
		od := other.float64s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.String:
		td := t.strings()
		od := other.strings()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	default:
		err = errors.Errorf(unsupportedDtype, t.t, "gte")
		return
	}

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
	switch t.t.Kind() {
	case reflect.Int:
		td := t.ints()
		od := other.ints()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Int8:
		td := t.int8s()
		od := other.int8s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Int16:
		td := t.int16s()
		od := other.int16s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Int32:
		td := t.int32s()
		od := other.int32s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Int64:
		td := t.int64s()
		od := other.int64s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Uint:
		td := t.uints()
		od := other.uints()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Uint8:
		td := t.uint8s()
		od := other.uint8s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Uint16:
		td := t.uint16s()
		od := other.uint16s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Uint32:
		td := t.uint32s()
		od := other.uint32s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Uint64:
		td := t.uint64s()
		od := other.uint64s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Uintptr:
		td := t.uintptrs()
		od := other.uintptrs()
		var i, j, k int
		var ret interface{} // slice of some sort
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
			ret = ltDDBoolsUintptr(td, od)
		}
		retVal.fromSlice(ret)

	case reflect.Float32:
		td := t.float32s()
		od := other.float32s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Float64:
		td := t.float64s()
		od := other.float64s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.String:
		td := t.strings()
		od := other.strings()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	default:
		err = errors.Errorf(unsupportedDtype, t.t, "lt")
		return
	}

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
	switch t.t.Kind() {
	case reflect.Int:
		td := t.ints()
		od := other.ints()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Int8:
		td := t.int8s()
		od := other.int8s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Int16:
		td := t.int16s()
		od := other.int16s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Int32:
		td := t.int32s()
		od := other.int32s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Int64:
		td := t.int64s()
		od := other.int64s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Uint:
		td := t.uints()
		od := other.uints()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Uint8:
		td := t.uint8s()
		od := other.uint8s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Uint16:
		td := t.uint16s()
		od := other.uint16s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Uint32:
		td := t.uint32s()
		od := other.uint32s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Uint64:
		td := t.uint64s()
		od := other.uint64s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Uintptr:
		td := t.uintptrs()
		od := other.uintptrs()
		var i, j, k int
		var ret interface{} // slice of some sort
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
			ret = lteDDBoolsUintptr(td, od)
		}
		retVal.fromSlice(ret)

	case reflect.Float32:
		td := t.float32s()
		od := other.float32s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.Float64:
		td := t.float64s()
		od := other.float64s()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	case reflect.String:
		td := t.strings()
		od := other.strings()
		var i, j, k int
		var ret interface{} // slice of some sort
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
		retVal.fromSlice(ret)

	default:
		err = errors.Errorf(unsupportedDtype, t.t, "lte")
		return
	}

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
