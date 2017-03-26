package tensor

import (
	"math"
	"reflect"
	"runtime"

	"github.com/pkg/errors"
)

/*
GENERATED FILE. DO NOT EDIT
*/

/* MaskedEqual */

// MaskedEqual sets the mask to true where the corresponding data is equal to  val
// Any values must be the same type as the tensor
func (t *Dense) MaskedEqual(val1 interface{}) (err error) {

	if !t.IsMasked() {
		t.SetMaskStrides(t.strides)
		t.fix()
	}
	it := MultIteratorFromDense(t)
	runtime.SetFinalizer(it, destroyMultIterator)

	switch t.t.Kind() {

	case reflect.Int:
		data := t.ints()
		mask := t.mask
		x := val1.(int)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a == x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a == x)
			}
		}
		it.Reset()

	case reflect.Int8:
		data := t.int8s()
		mask := t.mask
		x := val1.(int8)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a == x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a == x)
			}
		}
		it.Reset()

	case reflect.Int16:
		data := t.int16s()
		mask := t.mask
		x := val1.(int16)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a == x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a == x)
			}
		}
		it.Reset()

	case reflect.Int32:
		data := t.int32s()
		mask := t.mask
		x := val1.(int32)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a == x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a == x)
			}
		}
		it.Reset()

	case reflect.Int64:
		data := t.int64s()
		mask := t.mask
		x := val1.(int64)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a == x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a == x)
			}
		}
		it.Reset()

	case reflect.Uint:
		data := t.uints()
		mask := t.mask
		x := val1.(uint)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a == x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a == x)
			}
		}
		it.Reset()

	case reflect.Uint8:
		data := t.uint8s()
		mask := t.mask
		x := val1.(uint8)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a == x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a == x)
			}
		}
		it.Reset()

	case reflect.Uint16:
		data := t.uint16s()
		mask := t.mask
		x := val1.(uint16)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a == x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a == x)
			}
		}
		it.Reset()

	case reflect.Uint32:
		data := t.uint32s()
		mask := t.mask
		x := val1.(uint32)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a == x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a == x)
			}
		}
		it.Reset()

	case reflect.Uint64:
		data := t.uint64s()
		mask := t.mask
		x := val1.(uint64)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a == x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a == x)
			}
		}
		it.Reset()

	case reflect.Float32:
		data := t.float32s()
		mask := t.mask
		x := val1.(float32)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a == x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a == x)
			}
		}
		it.Reset()

	case reflect.Float64:
		data := t.float64s()
		mask := t.mask
		x := val1.(float64)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a == x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a == x)
			}
		}
		it.Reset()

	case reflect.String:
		data := t.strings()
		mask := t.mask
		x := val1.(string)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a == x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a == x)
			}
		}
		it.Reset()

	}
	return nil
}

/* MaskedNotEqual */

// MaskedNotEqual sets the mask to true where the corresponding data is not equal to  val
// Any values must be the same type as the tensor
func (t *Dense) MaskedNotEqual(val1 interface{}) (err error) {

	if !t.IsMasked() {
		t.SetMaskStrides(t.strides)
		t.fix()
	}
	it := MultIteratorFromDense(t)
	runtime.SetFinalizer(it, destroyMultIterator)

	switch t.t.Kind() {

	case reflect.Int:
		data := t.ints()
		mask := t.mask
		x := val1.(int)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a != x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a != x)
			}
		}
		it.Reset()

	case reflect.Int8:
		data := t.int8s()
		mask := t.mask
		x := val1.(int8)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a != x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a != x)
			}
		}
		it.Reset()

	case reflect.Int16:
		data := t.int16s()
		mask := t.mask
		x := val1.(int16)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a != x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a != x)
			}
		}
		it.Reset()

	case reflect.Int32:
		data := t.int32s()
		mask := t.mask
		x := val1.(int32)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a != x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a != x)
			}
		}
		it.Reset()

	case reflect.Int64:
		data := t.int64s()
		mask := t.mask
		x := val1.(int64)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a != x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a != x)
			}
		}
		it.Reset()

	case reflect.Uint:
		data := t.uints()
		mask := t.mask
		x := val1.(uint)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a != x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a != x)
			}
		}
		it.Reset()

	case reflect.Uint8:
		data := t.uint8s()
		mask := t.mask
		x := val1.(uint8)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a != x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a != x)
			}
		}
		it.Reset()

	case reflect.Uint16:
		data := t.uint16s()
		mask := t.mask
		x := val1.(uint16)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a != x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a != x)
			}
		}
		it.Reset()

	case reflect.Uint32:
		data := t.uint32s()
		mask := t.mask
		x := val1.(uint32)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a != x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a != x)
			}
		}
		it.Reset()

	case reflect.Uint64:
		data := t.uint64s()
		mask := t.mask
		x := val1.(uint64)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a != x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a != x)
			}
		}
		it.Reset()

	case reflect.Float32:
		data := t.float32s()
		mask := t.mask
		x := val1.(float32)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a != x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a != x)
			}
		}
		it.Reset()

	case reflect.Float64:
		data := t.float64s()
		mask := t.mask
		x := val1.(float64)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a != x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a != x)
			}
		}
		it.Reset()

	case reflect.String:
		data := t.strings()
		mask := t.mask
		x := val1.(string)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a != x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a != x)
			}
		}
		it.Reset()

	}
	return nil
}

/* MaskedValues */

// MaskedValues sets the mask to true where the corresponding data is  equal to  val
// Any values must be the same type as the tensor
func (t *Dense) MaskedValues(val1 interface{}, val2 interface{}, val3 ...interface{}) (err error) {

	if !isFloat(t.t) {
		err = errors.Errorf("Can only do MaskedValues with floating point types")
		return
	}

	if !t.IsMasked() {
		t.SetMaskStrides(t.strides)
		t.fix()
	}
	it := MultIteratorFromDense(t)
	runtime.SetFinalizer(it, destroyMultIterator)

	switch t.t.Kind() {

	case reflect.Float32:
		data := t.float32s()
		mask := t.mask
		x := val1.(float32)
		y := val2.(float32)

		delta := float64(1.0e-8)
		if len(val3) > 0 {
			delta = float64(val3[0].(float32)) + float64(y)*math.Abs(float64(x))
		}

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (math.Abs(float64(a-x)) <= delta)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (math.Abs(float64(a-x)) <= delta)
			}
		}
		it.Reset()

	case reflect.Float64:
		data := t.float64s()
		mask := t.mask
		x := val1.(float64)
		y := val2.(float64)

		delta := float64(1.0e-8)
		if len(val3) > 0 {
			delta = float64(val3[0].(float64)) + float64(y)*math.Abs(float64(x))
		}

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (math.Abs(float64(a-x)) <= delta)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (math.Abs(float64(a-x)) <= delta)
			}
		}
		it.Reset()

	}
	return nil
}

/* MaskedGreater */

// MaskedGreater sets the mask to true where the corresponding data is  greater than  val
// Any values must be the same type as the tensor
func (t *Dense) MaskedGreater(val1 interface{}) (err error) {

	if !t.IsMasked() {
		t.SetMaskStrides(t.strides)
		t.fix()
	}
	it := MultIteratorFromDense(t)
	runtime.SetFinalizer(it, destroyMultIterator)

	switch t.t.Kind() {

	case reflect.Int:
		data := t.ints()
		mask := t.mask
		x := val1.(int)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a > x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a > x)
			}
		}
		it.Reset()

	case reflect.Int8:
		data := t.int8s()
		mask := t.mask
		x := val1.(int8)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a > x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a > x)
			}
		}
		it.Reset()

	case reflect.Int16:
		data := t.int16s()
		mask := t.mask
		x := val1.(int16)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a > x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a > x)
			}
		}
		it.Reset()

	case reflect.Int32:
		data := t.int32s()
		mask := t.mask
		x := val1.(int32)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a > x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a > x)
			}
		}
		it.Reset()

	case reflect.Int64:
		data := t.int64s()
		mask := t.mask
		x := val1.(int64)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a > x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a > x)
			}
		}
		it.Reset()

	case reflect.Uint:
		data := t.uints()
		mask := t.mask
		x := val1.(uint)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a > x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a > x)
			}
		}
		it.Reset()

	case reflect.Uint8:
		data := t.uint8s()
		mask := t.mask
		x := val1.(uint8)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a > x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a > x)
			}
		}
		it.Reset()

	case reflect.Uint16:
		data := t.uint16s()
		mask := t.mask
		x := val1.(uint16)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a > x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a > x)
			}
		}
		it.Reset()

	case reflect.Uint32:
		data := t.uint32s()
		mask := t.mask
		x := val1.(uint32)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a > x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a > x)
			}
		}
		it.Reset()

	case reflect.Uint64:
		data := t.uint64s()
		mask := t.mask
		x := val1.(uint64)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a > x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a > x)
			}
		}
		it.Reset()

	case reflect.Float32:
		data := t.float32s()
		mask := t.mask
		x := val1.(float32)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a > x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a > x)
			}
		}
		it.Reset()

	case reflect.Float64:
		data := t.float64s()
		mask := t.mask
		x := val1.(float64)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a > x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a > x)
			}
		}
		it.Reset()

	case reflect.String:
		data := t.strings()
		mask := t.mask
		x := val1.(string)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a > x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a > x)
			}
		}
		it.Reset()

	}
	return nil
}

/* MaskedGreaterEqual */

// MaskedGreaterEqual sets the mask to true where the corresponding data is  greater than or equal to  val
// Any values must be the same type as the tensor
func (t *Dense) MaskedGreaterEqual(val1 interface{}) (err error) {

	if !t.IsMasked() {
		t.SetMaskStrides(t.strides)
		t.fix()
	}
	it := MultIteratorFromDense(t)
	runtime.SetFinalizer(it, destroyMultIterator)

	switch t.t.Kind() {

	case reflect.Int:
		data := t.ints()
		mask := t.mask
		x := val1.(int)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a >= x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a >= x)
			}
		}
		it.Reset()

	case reflect.Int8:
		data := t.int8s()
		mask := t.mask
		x := val1.(int8)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a >= x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a >= x)
			}
		}
		it.Reset()

	case reflect.Int16:
		data := t.int16s()
		mask := t.mask
		x := val1.(int16)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a >= x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a >= x)
			}
		}
		it.Reset()

	case reflect.Int32:
		data := t.int32s()
		mask := t.mask
		x := val1.(int32)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a >= x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a >= x)
			}
		}
		it.Reset()

	case reflect.Int64:
		data := t.int64s()
		mask := t.mask
		x := val1.(int64)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a >= x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a >= x)
			}
		}
		it.Reset()

	case reflect.Uint:
		data := t.uints()
		mask := t.mask
		x := val1.(uint)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a >= x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a >= x)
			}
		}
		it.Reset()

	case reflect.Uint8:
		data := t.uint8s()
		mask := t.mask
		x := val1.(uint8)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a >= x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a >= x)
			}
		}
		it.Reset()

	case reflect.Uint16:
		data := t.uint16s()
		mask := t.mask
		x := val1.(uint16)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a >= x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a >= x)
			}
		}
		it.Reset()

	case reflect.Uint32:
		data := t.uint32s()
		mask := t.mask
		x := val1.(uint32)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a >= x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a >= x)
			}
		}
		it.Reset()

	case reflect.Uint64:
		data := t.uint64s()
		mask := t.mask
		x := val1.(uint64)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a >= x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a >= x)
			}
		}
		it.Reset()

	case reflect.Float32:
		data := t.float32s()
		mask := t.mask
		x := val1.(float32)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a >= x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a >= x)
			}
		}
		it.Reset()

	case reflect.Float64:
		data := t.float64s()
		mask := t.mask
		x := val1.(float64)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a >= x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a >= x)
			}
		}
		it.Reset()

	case reflect.String:
		data := t.strings()
		mask := t.mask
		x := val1.(string)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a >= x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a >= x)
			}
		}
		it.Reset()

	}
	return nil
}

/* MaskedLess */

// MaskedLess sets the mask to true where the corresponding data is  less than  val
// Any values must be the same type as the tensor
func (t *Dense) MaskedLess(val1 interface{}) (err error) {

	if !t.IsMasked() {
		t.SetMaskStrides(t.strides)
		t.fix()
	}
	it := MultIteratorFromDense(t)
	runtime.SetFinalizer(it, destroyMultIterator)

	switch t.t.Kind() {

	case reflect.Int:
		data := t.ints()
		mask := t.mask
		x := val1.(int)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a < x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a < x)
			}
		}
		it.Reset()

	case reflect.Int8:
		data := t.int8s()
		mask := t.mask
		x := val1.(int8)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a < x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a < x)
			}
		}
		it.Reset()

	case reflect.Int16:
		data := t.int16s()
		mask := t.mask
		x := val1.(int16)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a < x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a < x)
			}
		}
		it.Reset()

	case reflect.Int32:
		data := t.int32s()
		mask := t.mask
		x := val1.(int32)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a < x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a < x)
			}
		}
		it.Reset()

	case reflect.Int64:
		data := t.int64s()
		mask := t.mask
		x := val1.(int64)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a < x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a < x)
			}
		}
		it.Reset()

	case reflect.Uint:
		data := t.uints()
		mask := t.mask
		x := val1.(uint)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a < x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a < x)
			}
		}
		it.Reset()

	case reflect.Uint8:
		data := t.uint8s()
		mask := t.mask
		x := val1.(uint8)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a < x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a < x)
			}
		}
		it.Reset()

	case reflect.Uint16:
		data := t.uint16s()
		mask := t.mask
		x := val1.(uint16)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a < x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a < x)
			}
		}
		it.Reset()

	case reflect.Uint32:
		data := t.uint32s()
		mask := t.mask
		x := val1.(uint32)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a < x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a < x)
			}
		}
		it.Reset()

	case reflect.Uint64:
		data := t.uint64s()
		mask := t.mask
		x := val1.(uint64)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a < x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a < x)
			}
		}
		it.Reset()

	case reflect.Float32:
		data := t.float32s()
		mask := t.mask
		x := val1.(float32)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a < x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a < x)
			}
		}
		it.Reset()

	case reflect.Float64:
		data := t.float64s()
		mask := t.mask
		x := val1.(float64)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a < x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a < x)
			}
		}
		it.Reset()

	case reflect.String:
		data := t.strings()
		mask := t.mask
		x := val1.(string)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a < x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a < x)
			}
		}
		it.Reset()

	}
	return nil
}

/* MaskedLessEqual */

// MaskedLessEqual sets the mask to true where the corresponding data is  less than or equal to  val
// Any values must be the same type as the tensor
func (t *Dense) MaskedLessEqual(val1 interface{}) (err error) {

	if !t.IsMasked() {
		t.SetMaskStrides(t.strides)
		t.fix()
	}
	it := MultIteratorFromDense(t)
	runtime.SetFinalizer(it, destroyMultIterator)

	switch t.t.Kind() {

	case reflect.Int:
		data := t.ints()
		mask := t.mask
		x := val1.(int)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a <= x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a <= x)
			}
		}
		it.Reset()

	case reflect.Int8:
		data := t.int8s()
		mask := t.mask
		x := val1.(int8)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a <= x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a <= x)
			}
		}
		it.Reset()

	case reflect.Int16:
		data := t.int16s()
		mask := t.mask
		x := val1.(int16)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a <= x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a <= x)
			}
		}
		it.Reset()

	case reflect.Int32:
		data := t.int32s()
		mask := t.mask
		x := val1.(int32)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a <= x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a <= x)
			}
		}
		it.Reset()

	case reflect.Int64:
		data := t.int64s()
		mask := t.mask
		x := val1.(int64)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a <= x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a <= x)
			}
		}
		it.Reset()

	case reflect.Uint:
		data := t.uints()
		mask := t.mask
		x := val1.(uint)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a <= x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a <= x)
			}
		}
		it.Reset()

	case reflect.Uint8:
		data := t.uint8s()
		mask := t.mask
		x := val1.(uint8)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a <= x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a <= x)
			}
		}
		it.Reset()

	case reflect.Uint16:
		data := t.uint16s()
		mask := t.mask
		x := val1.(uint16)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a <= x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a <= x)
			}
		}
		it.Reset()

	case reflect.Uint32:
		data := t.uint32s()
		mask := t.mask
		x := val1.(uint32)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a <= x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a <= x)
			}
		}
		it.Reset()

	case reflect.Uint64:
		data := t.uint64s()
		mask := t.mask
		x := val1.(uint64)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a <= x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a <= x)
			}
		}
		it.Reset()

	case reflect.Float32:
		data := t.float32s()
		mask := t.mask
		x := val1.(float32)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a <= x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a <= x)
			}
		}
		it.Reset()

	case reflect.Float64:
		data := t.float64s()
		mask := t.mask
		x := val1.(float64)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a <= x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a <= x)
			}
		}
		it.Reset()

	case reflect.String:
		data := t.strings()
		mask := t.mask
		x := val1.(string)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = (a <= x)
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || (a <= x)
			}
		}
		it.Reset()

	}
	return nil
}

/* MaskedInside */

// MaskedInside sets the mask to true where the corresponding data is  inside range of  val
// Any values must be the same type as the tensor
func (t *Dense) MaskedInside(val1 interface{}, val2 interface{}) (err error) {

	if !t.IsMasked() {
		t.SetMaskStrides(t.strides)
		t.fix()
	}
	it := MultIteratorFromDense(t)
	runtime.SetFinalizer(it, destroyMultIterator)

	switch t.t.Kind() {

	case reflect.Int:
		data := t.ints()
		mask := t.mask
		x := val1.(int)
		y := val2.(int)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = ((a >= x) && (a <= y))
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || ((a >= x) && (a <= y))
			}
		}
		it.Reset()

	case reflect.Int8:
		data := t.int8s()
		mask := t.mask
		x := val1.(int8)
		y := val2.(int8)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = ((a >= x) && (a <= y))
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || ((a >= x) && (a <= y))
			}
		}
		it.Reset()

	case reflect.Int16:
		data := t.int16s()
		mask := t.mask
		x := val1.(int16)
		y := val2.(int16)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = ((a >= x) && (a <= y))
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || ((a >= x) && (a <= y))
			}
		}
		it.Reset()

	case reflect.Int32:
		data := t.int32s()
		mask := t.mask
		x := val1.(int32)
		y := val2.(int32)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = ((a >= x) && (a <= y))
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || ((a >= x) && (a <= y))
			}
		}
		it.Reset()

	case reflect.Int64:
		data := t.int64s()
		mask := t.mask
		x := val1.(int64)
		y := val2.(int64)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = ((a >= x) && (a <= y))
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || ((a >= x) && (a <= y))
			}
		}
		it.Reset()

	case reflect.Uint:
		data := t.uints()
		mask := t.mask
		x := val1.(uint)
		y := val2.(uint)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = ((a >= x) && (a <= y))
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || ((a >= x) && (a <= y))
			}
		}
		it.Reset()

	case reflect.Uint8:
		data := t.uint8s()
		mask := t.mask
		x := val1.(uint8)
		y := val2.(uint8)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = ((a >= x) && (a <= y))
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || ((a >= x) && (a <= y))
			}
		}
		it.Reset()

	case reflect.Uint16:
		data := t.uint16s()
		mask := t.mask
		x := val1.(uint16)
		y := val2.(uint16)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = ((a >= x) && (a <= y))
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || ((a >= x) && (a <= y))
			}
		}
		it.Reset()

	case reflect.Uint32:
		data := t.uint32s()
		mask := t.mask
		x := val1.(uint32)
		y := val2.(uint32)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = ((a >= x) && (a <= y))
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || ((a >= x) && (a <= y))
			}
		}
		it.Reset()

	case reflect.Uint64:
		data := t.uint64s()
		mask := t.mask
		x := val1.(uint64)
		y := val2.(uint64)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = ((a >= x) && (a <= y))
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || ((a >= x) && (a <= y))
			}
		}
		it.Reset()

	case reflect.Float32:
		data := t.float32s()
		mask := t.mask
		x := val1.(float32)
		y := val2.(float32)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = ((a >= x) && (a <= y))
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || ((a >= x) && (a <= y))
			}
		}
		it.Reset()

	case reflect.Float64:
		data := t.float64s()
		mask := t.mask
		x := val1.(float64)
		y := val2.(float64)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = ((a >= x) && (a <= y))
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || ((a >= x) && (a <= y))
			}
		}
		it.Reset()

	case reflect.String:
		data := t.strings()
		mask := t.mask
		x := val1.(string)
		y := val2.(string)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = ((a >= x) && (a <= y))
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || ((a >= x) && (a <= y))
			}
		}
		it.Reset()

	}
	return nil
}

/* MaskedOutside */

// MaskedOutside sets the mask to true where the corresponding data is  outside range of  val
// Any values must be the same type as the tensor
func (t *Dense) MaskedOutside(val1 interface{}, val2 interface{}) (err error) {

	if !t.IsMasked() {
		t.SetMaskStrides(t.strides)
		t.fix()
	}
	it := MultIteratorFromDense(t)
	runtime.SetFinalizer(it, destroyMultIterator)

	switch t.t.Kind() {

	case reflect.Int:
		data := t.ints()
		mask := t.mask
		x := val1.(int)
		y := val2.(int)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = ((a < x) || (a > y))
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || ((a < x) || (a > y))
			}
		}
		it.Reset()

	case reflect.Int8:
		data := t.int8s()
		mask := t.mask
		x := val1.(int8)
		y := val2.(int8)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = ((a < x) || (a > y))
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || ((a < x) || (a > y))
			}
		}
		it.Reset()

	case reflect.Int16:
		data := t.int16s()
		mask := t.mask
		x := val1.(int16)
		y := val2.(int16)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = ((a < x) || (a > y))
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || ((a < x) || (a > y))
			}
		}
		it.Reset()

	case reflect.Int32:
		data := t.int32s()
		mask := t.mask
		x := val1.(int32)
		y := val2.(int32)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = ((a < x) || (a > y))
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || ((a < x) || (a > y))
			}
		}
		it.Reset()

	case reflect.Int64:
		data := t.int64s()
		mask := t.mask
		x := val1.(int64)
		y := val2.(int64)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = ((a < x) || (a > y))
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || ((a < x) || (a > y))
			}
		}
		it.Reset()

	case reflect.Uint:
		data := t.uints()
		mask := t.mask
		x := val1.(uint)
		y := val2.(uint)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = ((a < x) || (a > y))
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || ((a < x) || (a > y))
			}
		}
		it.Reset()

	case reflect.Uint8:
		data := t.uint8s()
		mask := t.mask
		x := val1.(uint8)
		y := val2.(uint8)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = ((a < x) || (a > y))
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || ((a < x) || (a > y))
			}
		}
		it.Reset()

	case reflect.Uint16:
		data := t.uint16s()
		mask := t.mask
		x := val1.(uint16)
		y := val2.(uint16)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = ((a < x) || (a > y))
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || ((a < x) || (a > y))
			}
		}
		it.Reset()

	case reflect.Uint32:
		data := t.uint32s()
		mask := t.mask
		x := val1.(uint32)
		y := val2.(uint32)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = ((a < x) || (a > y))
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || ((a < x) || (a > y))
			}
		}
		it.Reset()

	case reflect.Uint64:
		data := t.uint64s()
		mask := t.mask
		x := val1.(uint64)
		y := val2.(uint64)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = ((a < x) || (a > y))
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || ((a < x) || (a > y))
			}
		}
		it.Reset()

	case reflect.Float32:
		data := t.float32s()
		mask := t.mask
		x := val1.(float32)
		y := val2.(float32)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = ((a < x) || (a > y))
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || ((a < x) || (a > y))
			}
		}
		it.Reset()

	case reflect.Float64:
		data := t.float64s()
		mask := t.mask
		x := val1.(float64)
		y := val2.(float64)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = ((a < x) || (a > y))
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || ((a < x) || (a > y))
			}
		}
		it.Reset()

	case reflect.String:
		data := t.strings()
		mask := t.mask
		x := val1.(string)
		y := val2.(string)

		if t.softmask {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = ((a < x) || (a > y))
			}
		} else {
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || ((a < x) || (a > y))
			}
		}
		it.Reset()

	}
	return nil
}
