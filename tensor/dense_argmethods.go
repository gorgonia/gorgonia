package tensor

import (
	"reflect"
	"runtime"

	"github.com/pkg/errors"
)

/*
GENERATED FILE. DO NOT EDIT
*/

/* Argmax */

// Argmax finds the index of the max value along the axis provided
func (t *Dense) Argmax(axis int) (retVal *Dense, err error) {
	if axis == AllAxes {
		return t.argmax(nil)
	}

	if axis >= len(t.Shape()) {
		err = errors.Errorf(dimMismatch, len(t.Shape()), axis)
		return
	}

	axes := make([]int, len(t.Shape()))
	for i := range t.Shape() {
		switch {
		case i < axis:
			axes[i] = i
		case i == axis:
			axes[len(axes)-1] = i
		case i > axis:
			axes[i-1] = i
		}
	}

	// be a good citizen - borrow and return, since we're only using this AP to figure out the moves
	newAP, _, err := t.AP.T(axes...)
	if _, ok := err.(NoOpError); !ok && err != nil {
		return
	} else if ok {
		newAP = t.AP.Clone()
	}
	defer ReturnAP(newAP)

	it := IteratorFromDense(t)
	runtime.SetFinalizer(it, destroyIterator)
	iteratorLoadAP(it, newAP)
	return t.argmax(it)
}

func (t *Dense) argmax(it Iterator) (retVal *Dense, err error) {
	var lastSize, next int
	var newShape Shape
	var indices []int
	var mask []bool
	if it != nil {
		lastSize = it.Shape()[len(it.Shape())-1]
		newShape = it.Shape().Clone()
		newShape = newShape[:len(newShape)-1]
		defer ReturnInts(newShape)
	}

	switch t.t.Kind() {

	case reflect.Int:
		var isMasked = t.IsMasked()
		if it == nil {
			retVal = New(FromScalar(argmaxI(t.ints(), t.mask)))
			return
		}
		data := t.ints()
		tmp := make([]int, 0, lastSize)
		mask = make([]bool, 0, lastSize)
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			tmp = append(tmp, data[next])
			if isMasked {
				mask = append(mask, t.mask[next])
			}
			if len(tmp) == lastSize {
				am := argmaxI(tmp, mask)
				indices = append(indices, am)

				// reset
				tmp = tmp[:0]
			}
		}
		if _, ok := err.(NoOpError); !ok && err != nil {
			return
		}
		err = nil
		retVal = New(WithShape(newShape...), WithBacking(indices))
		return

	case reflect.Int8:
		var isMasked = t.IsMasked()
		if it == nil {
			retVal = New(FromScalar(argmaxI8(t.int8s(), t.mask)))
			return
		}
		data := t.int8s()
		tmp := make([]int8, 0, lastSize)
		mask = make([]bool, 0, lastSize)
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			tmp = append(tmp, data[next])
			if isMasked {
				mask = append(mask, t.mask[next])
			}
			if len(tmp) == lastSize {
				am := argmaxI8(tmp, mask)
				indices = append(indices, am)

				// reset
				tmp = tmp[:0]
			}
		}
		if _, ok := err.(NoOpError); !ok && err != nil {
			return
		}
		err = nil
		retVal = New(WithShape(newShape...), WithBacking(indices))
		return

	case reflect.Int16:
		var isMasked = t.IsMasked()
		if it == nil {
			retVal = New(FromScalar(argmaxI16(t.int16s(), t.mask)))
			return
		}
		data := t.int16s()
		tmp := make([]int16, 0, lastSize)
		mask = make([]bool, 0, lastSize)
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			tmp = append(tmp, data[next])
			if isMasked {
				mask = append(mask, t.mask[next])
			}
			if len(tmp) == lastSize {
				am := argmaxI16(tmp, mask)
				indices = append(indices, am)

				// reset
				tmp = tmp[:0]
			}
		}
		if _, ok := err.(NoOpError); !ok && err != nil {
			return
		}
		err = nil
		retVal = New(WithShape(newShape...), WithBacking(indices))
		return

	case reflect.Int32:
		var isMasked = t.IsMasked()
		if it == nil {
			retVal = New(FromScalar(argmaxI32(t.int32s(), t.mask)))
			return
		}
		data := t.int32s()
		tmp := make([]int32, 0, lastSize)
		mask = make([]bool, 0, lastSize)
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			tmp = append(tmp, data[next])
			if isMasked {
				mask = append(mask, t.mask[next])
			}
			if len(tmp) == lastSize {
				am := argmaxI32(tmp, mask)
				indices = append(indices, am)

				// reset
				tmp = tmp[:0]
			}
		}
		if _, ok := err.(NoOpError); !ok && err != nil {
			return
		}
		err = nil
		retVal = New(WithShape(newShape...), WithBacking(indices))
		return

	case reflect.Int64:
		var isMasked = t.IsMasked()
		if it == nil {
			retVal = New(FromScalar(argmaxI64(t.int64s(), t.mask)))
			return
		}
		data := t.int64s()
		tmp := make([]int64, 0, lastSize)
		mask = make([]bool, 0, lastSize)
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			tmp = append(tmp, data[next])
			if isMasked {
				mask = append(mask, t.mask[next])
			}
			if len(tmp) == lastSize {
				am := argmaxI64(tmp, mask)
				indices = append(indices, am)

				// reset
				tmp = tmp[:0]
			}
		}
		if _, ok := err.(NoOpError); !ok && err != nil {
			return
		}
		err = nil
		retVal = New(WithShape(newShape...), WithBacking(indices))
		return

	case reflect.Uint:
		var isMasked = t.IsMasked()
		if it == nil {
			retVal = New(FromScalar(argmaxU(t.uints(), t.mask)))
			return
		}
		data := t.uints()
		tmp := make([]uint, 0, lastSize)
		mask = make([]bool, 0, lastSize)
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			tmp = append(tmp, data[next])
			if isMasked {
				mask = append(mask, t.mask[next])
			}
			if len(tmp) == lastSize {
				am := argmaxU(tmp, mask)
				indices = append(indices, am)

				// reset
				tmp = tmp[:0]
			}
		}
		if _, ok := err.(NoOpError); !ok && err != nil {
			return
		}
		err = nil
		retVal = New(WithShape(newShape...), WithBacking(indices))
		return

	case reflect.Uint8:
		var isMasked = t.IsMasked()
		if it == nil {
			retVal = New(FromScalar(argmaxU8(t.uint8s(), t.mask)))
			return
		}
		data := t.uint8s()
		tmp := make([]uint8, 0, lastSize)
		mask = make([]bool, 0, lastSize)
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			tmp = append(tmp, data[next])
			if isMasked {
				mask = append(mask, t.mask[next])
			}
			if len(tmp) == lastSize {
				am := argmaxU8(tmp, mask)
				indices = append(indices, am)

				// reset
				tmp = tmp[:0]
			}
		}
		if _, ok := err.(NoOpError); !ok && err != nil {
			return
		}
		err = nil
		retVal = New(WithShape(newShape...), WithBacking(indices))
		return

	case reflect.Uint16:
		var isMasked = t.IsMasked()
		if it == nil {
			retVal = New(FromScalar(argmaxU16(t.uint16s(), t.mask)))
			return
		}
		data := t.uint16s()
		tmp := make([]uint16, 0, lastSize)
		mask = make([]bool, 0, lastSize)
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			tmp = append(tmp, data[next])
			if isMasked {
				mask = append(mask, t.mask[next])
			}
			if len(tmp) == lastSize {
				am := argmaxU16(tmp, mask)
				indices = append(indices, am)

				// reset
				tmp = tmp[:0]
			}
		}
		if _, ok := err.(NoOpError); !ok && err != nil {
			return
		}
		err = nil
		retVal = New(WithShape(newShape...), WithBacking(indices))
		return

	case reflect.Uint32:
		var isMasked = t.IsMasked()
		if it == nil {
			retVal = New(FromScalar(argmaxU32(t.uint32s(), t.mask)))
			return
		}
		data := t.uint32s()
		tmp := make([]uint32, 0, lastSize)
		mask = make([]bool, 0, lastSize)
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			tmp = append(tmp, data[next])
			if isMasked {
				mask = append(mask, t.mask[next])
			}
			if len(tmp) == lastSize {
				am := argmaxU32(tmp, mask)
				indices = append(indices, am)

				// reset
				tmp = tmp[:0]
			}
		}
		if _, ok := err.(NoOpError); !ok && err != nil {
			return
		}
		err = nil
		retVal = New(WithShape(newShape...), WithBacking(indices))
		return

	case reflect.Uint64:
		var isMasked = t.IsMasked()
		if it == nil {
			retVal = New(FromScalar(argmaxU64(t.uint64s(), t.mask)))
			return
		}
		data := t.uint64s()
		tmp := make([]uint64, 0, lastSize)
		mask = make([]bool, 0, lastSize)
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			tmp = append(tmp, data[next])
			if isMasked {
				mask = append(mask, t.mask[next])
			}
			if len(tmp) == lastSize {
				am := argmaxU64(tmp, mask)
				indices = append(indices, am)

				// reset
				tmp = tmp[:0]
			}
		}
		if _, ok := err.(NoOpError); !ok && err != nil {
			return
		}
		err = nil
		retVal = New(WithShape(newShape...), WithBacking(indices))
		return

	case reflect.Float32:
		var isMasked = t.IsMasked()
		if it == nil {
			retVal = New(FromScalar(argmaxF32(t.float32s(), t.mask)))
			return
		}
		data := t.float32s()
		tmp := make([]float32, 0, lastSize)
		mask = make([]bool, 0, lastSize)
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			tmp = append(tmp, data[next])
			if isMasked {
				mask = append(mask, t.mask[next])
			}
			if len(tmp) == lastSize {
				am := argmaxF32(tmp, mask)
				indices = append(indices, am)

				// reset
				tmp = tmp[:0]
			}
		}
		if _, ok := err.(NoOpError); !ok && err != nil {
			return
		}
		err = nil
		retVal = New(WithShape(newShape...), WithBacking(indices))
		return

	case reflect.Float64:
		var isMasked = t.IsMasked()
		if it == nil {
			retVal = New(FromScalar(argmaxF64(t.float64s(), t.mask)))
			return
		}
		data := t.float64s()
		tmp := make([]float64, 0, lastSize)
		mask = make([]bool, 0, lastSize)
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			tmp = append(tmp, data[next])
			if isMasked {
				mask = append(mask, t.mask[next])
			}
			if len(tmp) == lastSize {
				am := argmaxF64(tmp, mask)
				indices = append(indices, am)

				// reset
				tmp = tmp[:0]
			}
		}
		if _, ok := err.(NoOpError); !ok && err != nil {
			return
		}
		err = nil
		retVal = New(WithShape(newShape...), WithBacking(indices))
		return

	}
	panic("Unreachable")
}

/* Argmin */

// Argmin finds the index of the min value along the axis provided
func (t *Dense) Argmin(axis int) (retVal *Dense, err error) {
	if axis == AllAxes {
		return t.argmin(nil)
	}

	if axis >= len(t.Shape()) {
		err = errors.Errorf(dimMismatch, len(t.Shape()), axis)
		return
	}

	axes := make([]int, len(t.Shape()))
	for i := range t.Shape() {
		switch {
		case i < axis:
			axes[i] = i
		case i == axis:
			axes[len(axes)-1] = i
		case i > axis:
			axes[i-1] = i
		}
	}

	// be a good citizen - borrow and return, since we're only using this AP to figure out the moves
	newAP, _, err := t.AP.T(axes...)
	if _, ok := err.(NoOpError); !ok && err != nil {
		return
	} else if ok {
		newAP = t.AP.Clone()
	}
	defer ReturnAP(newAP)

	it := IteratorFromDense(t)
	runtime.SetFinalizer(it, destroyIterator)
	iteratorLoadAP(it, newAP)
	return t.argmin(it)
}

func (t *Dense) argmin(it Iterator) (retVal *Dense, err error) {
	var lastSize, next int
	var newShape Shape
	var indices []int
	var mask []bool
	if it != nil {
		lastSize = it.Shape()[len(it.Shape())-1]
		newShape = it.Shape().Clone()
		newShape = newShape[:len(newShape)-1]
		defer ReturnInts(newShape)
	}

	switch t.t.Kind() {

	case reflect.Int:
		var isMasked = t.IsMasked()
		if it == nil {
			retVal = New(FromScalar(argminI(t.ints(), t.mask)))
			return
		}
		data := t.ints()
		tmp := make([]int, 0, lastSize)
		mask = make([]bool, 0, lastSize)
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			tmp = append(tmp, data[next])
			if isMasked {
				mask = append(mask, t.mask[next])
			}
			if len(tmp) == lastSize {
				am := argminI(tmp, mask)
				indices = append(indices, am)

				// reset
				tmp = tmp[:0]
			}
		}
		if _, ok := err.(NoOpError); !ok && err != nil {
			return
		}
		err = nil
		retVal = New(WithShape(newShape...), WithBacking(indices))
		return

	case reflect.Int8:
		var isMasked = t.IsMasked()
		if it == nil {
			retVal = New(FromScalar(argminI8(t.int8s(), t.mask)))
			return
		}
		data := t.int8s()
		tmp := make([]int8, 0, lastSize)
		mask = make([]bool, 0, lastSize)
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			tmp = append(tmp, data[next])
			if isMasked {
				mask = append(mask, t.mask[next])
			}
			if len(tmp) == lastSize {
				am := argminI8(tmp, mask)
				indices = append(indices, am)

				// reset
				tmp = tmp[:0]
			}
		}
		if _, ok := err.(NoOpError); !ok && err != nil {
			return
		}
		err = nil
		retVal = New(WithShape(newShape...), WithBacking(indices))
		return

	case reflect.Int16:
		var isMasked = t.IsMasked()
		if it == nil {
			retVal = New(FromScalar(argminI16(t.int16s(), t.mask)))
			return
		}
		data := t.int16s()
		tmp := make([]int16, 0, lastSize)
		mask = make([]bool, 0, lastSize)
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			tmp = append(tmp, data[next])
			if isMasked {
				mask = append(mask, t.mask[next])
			}
			if len(tmp) == lastSize {
				am := argminI16(tmp, mask)
				indices = append(indices, am)

				// reset
				tmp = tmp[:0]
			}
		}
		if _, ok := err.(NoOpError); !ok && err != nil {
			return
		}
		err = nil
		retVal = New(WithShape(newShape...), WithBacking(indices))
		return

	case reflect.Int32:
		var isMasked = t.IsMasked()
		if it == nil {
			retVal = New(FromScalar(argminI32(t.int32s(), t.mask)))
			return
		}
		data := t.int32s()
		tmp := make([]int32, 0, lastSize)
		mask = make([]bool, 0, lastSize)
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			tmp = append(tmp, data[next])
			if isMasked {
				mask = append(mask, t.mask[next])
			}
			if len(tmp) == lastSize {
				am := argminI32(tmp, mask)
				indices = append(indices, am)

				// reset
				tmp = tmp[:0]
			}
		}
		if _, ok := err.(NoOpError); !ok && err != nil {
			return
		}
		err = nil
		retVal = New(WithShape(newShape...), WithBacking(indices))
		return

	case reflect.Int64:
		var isMasked = t.IsMasked()
		if it == nil {
			retVal = New(FromScalar(argminI64(t.int64s(), t.mask)))
			return
		}
		data := t.int64s()
		tmp := make([]int64, 0, lastSize)
		mask = make([]bool, 0, lastSize)
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			tmp = append(tmp, data[next])
			if isMasked {
				mask = append(mask, t.mask[next])
			}
			if len(tmp) == lastSize {
				am := argminI64(tmp, mask)
				indices = append(indices, am)

				// reset
				tmp = tmp[:0]
			}
		}
		if _, ok := err.(NoOpError); !ok && err != nil {
			return
		}
		err = nil
		retVal = New(WithShape(newShape...), WithBacking(indices))
		return

	case reflect.Uint:
		var isMasked = t.IsMasked()
		if it == nil {
			retVal = New(FromScalar(argminU(t.uints(), t.mask)))
			return
		}
		data := t.uints()
		tmp := make([]uint, 0, lastSize)
		mask = make([]bool, 0, lastSize)
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			tmp = append(tmp, data[next])
			if isMasked {
				mask = append(mask, t.mask[next])
			}
			if len(tmp) == lastSize {
				am := argminU(tmp, mask)
				indices = append(indices, am)

				// reset
				tmp = tmp[:0]
			}
		}
		if _, ok := err.(NoOpError); !ok && err != nil {
			return
		}
		err = nil
		retVal = New(WithShape(newShape...), WithBacking(indices))
		return

	case reflect.Uint8:
		var isMasked = t.IsMasked()
		if it == nil {
			retVal = New(FromScalar(argminU8(t.uint8s(), t.mask)))
			return
		}
		data := t.uint8s()
		tmp := make([]uint8, 0, lastSize)
		mask = make([]bool, 0, lastSize)
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			tmp = append(tmp, data[next])
			if isMasked {
				mask = append(mask, t.mask[next])
			}
			if len(tmp) == lastSize {
				am := argminU8(tmp, mask)
				indices = append(indices, am)

				// reset
				tmp = tmp[:0]
			}
		}
		if _, ok := err.(NoOpError); !ok && err != nil {
			return
		}
		err = nil
		retVal = New(WithShape(newShape...), WithBacking(indices))
		return

	case reflect.Uint16:
		var isMasked = t.IsMasked()
		if it == nil {
			retVal = New(FromScalar(argminU16(t.uint16s(), t.mask)))
			return
		}
		data := t.uint16s()
		tmp := make([]uint16, 0, lastSize)
		mask = make([]bool, 0, lastSize)
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			tmp = append(tmp, data[next])
			if isMasked {
				mask = append(mask, t.mask[next])
			}
			if len(tmp) == lastSize {
				am := argminU16(tmp, mask)
				indices = append(indices, am)

				// reset
				tmp = tmp[:0]
			}
		}
		if _, ok := err.(NoOpError); !ok && err != nil {
			return
		}
		err = nil
		retVal = New(WithShape(newShape...), WithBacking(indices))
		return

	case reflect.Uint32:
		var isMasked = t.IsMasked()
		if it == nil {
			retVal = New(FromScalar(argminU32(t.uint32s(), t.mask)))
			return
		}
		data := t.uint32s()
		tmp := make([]uint32, 0, lastSize)
		mask = make([]bool, 0, lastSize)
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			tmp = append(tmp, data[next])
			if isMasked {
				mask = append(mask, t.mask[next])
			}
			if len(tmp) == lastSize {
				am := argminU32(tmp, mask)
				indices = append(indices, am)

				// reset
				tmp = tmp[:0]
			}
		}
		if _, ok := err.(NoOpError); !ok && err != nil {
			return
		}
		err = nil
		retVal = New(WithShape(newShape...), WithBacking(indices))
		return

	case reflect.Uint64:
		var isMasked = t.IsMasked()
		if it == nil {
			retVal = New(FromScalar(argminU64(t.uint64s(), t.mask)))
			return
		}
		data := t.uint64s()
		tmp := make([]uint64, 0, lastSize)
		mask = make([]bool, 0, lastSize)
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			tmp = append(tmp, data[next])
			if isMasked {
				mask = append(mask, t.mask[next])
			}
			if len(tmp) == lastSize {
				am := argminU64(tmp, mask)
				indices = append(indices, am)

				// reset
				tmp = tmp[:0]
			}
		}
		if _, ok := err.(NoOpError); !ok && err != nil {
			return
		}
		err = nil
		retVal = New(WithShape(newShape...), WithBacking(indices))
		return

	case reflect.Float32:
		var isMasked = t.IsMasked()
		if it == nil {
			retVal = New(FromScalar(argminF32(t.float32s(), t.mask)))
			return
		}
		data := t.float32s()
		tmp := make([]float32, 0, lastSize)
		mask = make([]bool, 0, lastSize)
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			tmp = append(tmp, data[next])
			if isMasked {
				mask = append(mask, t.mask[next])
			}
			if len(tmp) == lastSize {
				am := argminF32(tmp, mask)
				indices = append(indices, am)

				// reset
				tmp = tmp[:0]
			}
		}
		if _, ok := err.(NoOpError); !ok && err != nil {
			return
		}
		err = nil
		retVal = New(WithShape(newShape...), WithBacking(indices))
		return

	case reflect.Float64:
		var isMasked = t.IsMasked()
		if it == nil {
			retVal = New(FromScalar(argminF64(t.float64s(), t.mask)))
			return
		}
		data := t.float64s()
		tmp := make([]float64, 0, lastSize)
		mask = make([]bool, 0, lastSize)
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			tmp = append(tmp, data[next])
			if isMasked {
				mask = append(mask, t.mask[next])
			}
			if len(tmp) == lastSize {
				am := argminF64(tmp, mask)
				indices = append(indices, am)

				// reset
				tmp = tmp[:0]
			}
		}
		if _, ok := err.(NoOpError); !ok && err != nil {
			return
		}
		err = nil
		retVal = New(WithShape(newShape...), WithBacking(indices))
		return

	}
	panic("Unreachable")
}
