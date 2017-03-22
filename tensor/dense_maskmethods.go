package tensor

import (
	"math"
	"reflect"
)

type maskCmpFn func(float64, float64, float64) bool

func checkMaskNotEqual(x, y float64) bool {
	return x != y
}

func maskedCmp(t *Dense, fn maskCmpFn, arg1, arg2 interface{}) (err error) {
	if !t.IsMasked() {
		t.SetMaskStrides(t.strides)
		t.fix()
	}
	it := MultIteratorFromDense(t)
	switch t.t.Kind() {
	case reflect.Float64:
		data := t.Data().([]float64)
		mask := t.mask
		x := float64(arg1.(float64))
		y := float64(arg2.(float64))
		for i, err := it.Next(); err == nil; i, err = it.Next() {
			j := it.LastMaskIndex(0)
			mask[j] = mask[j] || fn(float64(data[i]), x, y)
		}
		it.Reset()
		DestroyMultIterator(it)
	}
	return nil
}

//ResetMask fills the mask with either false, or the provided boolean value
func (t *Dense) ResetMask(val ...bool) error {
	if !t.IsMasked() {
		t.SetMaskStrides(t.strides)
		t.fix()
	}
	var fillValue = false
	if len(val) > 0 {
		fillValue = val[0]
	}

	for i := range t.mask {
		t.mask[i] = fillValue
	}
	return nil
}

//MaskedEqual masks an array where equal to a given value
func (t *Dense) MaskedEqual(val interface{}) error {
	fn := func(x, y, z float64) bool { return x == y }
	return maskedCmp(t, fn, val, 0.0)
}

//MaskedNotEqual masks an array where not equal to a given value
func (t *Dense) MaskedNotEqual(val interface{}) error {
	fn := func(x, y, z float64) bool { return x != y }
	return maskedCmp(t, fn, val, 0.0)
}

//MaskedValues masks an array using floating point equality
// value, rtol=1e-05, atol=1e-08
// (abs(x-value) <= atol+rtol*abs(value))
func (t *Dense) MaskedValues(val interface{}, tols ...interface{}) error {
	rtol := 1.0e-5
	if len(tols) > 0 {
		rtol = tols[0].(float64)
	}
	fn := func(x, y, z float64) bool {
		atol := 1.0e-8
		delta := atol + rtol*math.Abs(y)
		return math.Abs(x-y) <= delta
	}
	return maskedCmp(t, fn, val, rtol)
}

//MaskedGreater masks an array where greater than a given value
func (t *Dense) MaskedGreater(val interface{}) error {
	fn := func(x, y, z float64) bool { return x > y }
	return maskedCmp(t, fn, val, 0.0)
}

//MaskedGreaterEqual masks an array where greater than or equal a given value
func (t *Dense) MaskedGreaterEqual(val interface{}) error {
	fn := func(x, y, z float64) bool { return x >= y }
	return maskedCmp(t, fn, val, 0.0)
}

//MaskedLess masks an array where greater than a given value
func (t *Dense) MaskedLess(val interface{}) error {
	fn := func(x, y, z float64) bool { return x < y }
	return maskedCmp(t, fn, val, 0.0)
}

//MaskedLessEqual masks an array where greater than or equal a given value
func (t *Dense) MaskedLessEqual(val interface{}) error {
	fn := func(x, y, z float64) bool { return x <= y }
	return maskedCmp(t, fn, val, 0.0)
}

//MaskedInside masks an array inside a given interval.
// boundaries can be given in any order
func (t *Dense) MaskedInside(val1, val2 interface{}) error {
	if val1.(float64) > val2.(float64) {
		t := val1.(float64)
		val1 = val2
		val2 = t
	}
	fn := func(x, y, z float64) bool { return (x >= y) && (x <= z) }
	return maskedCmp(t, fn, val1, val2)
}

//MaskedOutside masks an array outside a given interval.
// boundaries can be given in any order
func (t *Dense) MaskedOutside(val1, val2 interface{}) error {
	if val1.(float64) > val2.(float64) {
		t := val1.(float64)
		val1 = val2
		val2 = t
	}
	fn := func(x, y, z float64) bool { return (x < y) || (x > z) }
	return maskedCmp(t, fn, val1, val2)
}
