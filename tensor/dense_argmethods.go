package tensor

import (
	"reflect"

	"github.com/pkg/errors"
)

/*
GENERATED FILE. DO NOT EDIT
*/

/* Argmax */

// Argmax finds the index of the max value along the axis provided
func (t *Dense) Argmax(axis int) (retVal *Dense, err error) {
	e := t.e
	if e == nil {
		e = StdEng{}
	}
	switch am := e.(type) {
	case denseArgmaxer:
		return am.argmaxDenseTensor(t, axis)
	case Argmaxer:
		var ret Tensor
		var ok bool
		if ret, err = am.Argmax(t, axis); err != nil {
			return nil, errors.Wrapf(err, opFail, "Argmax")
		}
		if retVal, ok = ret.(*Dense); !ok {
			return nil, errors.Errorf(extractionFail, "*Dense", ret)
		}
		return
	}
	return nil, errors.New("Engine does not suport Argmax")
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
			retVal = New(FromScalar(argminI(t.Ints(), t.mask)))
			return
		}
		data := t.Ints()
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
			retVal = New(FromScalar(argminI8(t.Int8s(), t.mask)))
			return
		}
		data := t.Int8s()
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
			retVal = New(FromScalar(argminI16(t.Int16s(), t.mask)))
			return
		}
		data := t.Int16s()
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
			retVal = New(FromScalar(argminI32(t.Int32s(), t.mask)))
			return
		}
		data := t.Int32s()
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
			retVal = New(FromScalar(argminI64(t.Int64s(), t.mask)))
			return
		}
		data := t.Int64s()
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
			retVal = New(FromScalar(argminU(t.Uints(), t.mask)))
			return
		}
		data := t.Uints()
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
			retVal = New(FromScalar(argminU8(t.Uint8s(), t.mask)))
			return
		}
		data := t.Uint8s()
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
			retVal = New(FromScalar(argminU16(t.Uint16s(), t.mask)))
			return
		}
		data := t.Uint16s()
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
			retVal = New(FromScalar(argminU32(t.Uint32s(), t.mask)))
			return
		}
		data := t.Uint32s()
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
			retVal = New(FromScalar(argminU64(t.Uint64s(), t.mask)))
			return
		}
		data := t.Uint64s()
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
			retVal = New(FromScalar(argminF32(t.Float32s(), t.mask)))
			return
		}
		data := t.Float32s()
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
			retVal = New(FromScalar(argminF64(t.Float64s(), t.mask)))
			return
		}
		data := t.Float64s()
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
