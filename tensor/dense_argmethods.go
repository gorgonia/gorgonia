package tensor

import (
	"reflect"

	"github.com/pkg/errors"
)

/*
GENERATED FILE. DO NOT EDIT
*/

/* Argmax */

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

	it := NewFlatIterator(newAP)
	return t.argmax(it)
}

func (t *Dense) argmax(it *FlatIterator) (retVal *Dense, err error) {
	var lastSize, next int
	var newShape Shape
	var indices []int
	if it != nil {
		lastSize = it.Shape()[len(it.Shape())-1]
		newShape = it.Shape().Clone()
		newShape = newShape[:len(newShape)-1]
		defer ReturnInts(newShape)
	}

	switch t.t.Kind() {

	case reflect.Int:
		if it == nil {
			retVal = New(FromScalar(argmaxI(t.ints())))
			return
		}
		data := t.ints()
		tmp := make([]int, 0, lastSize)
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			tmp = append(tmp, data[next])

			if len(tmp) == lastSize {
				am := argmaxI(tmp)
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
		if it == nil {
			retVal = New(FromScalar(argmaxI8(t.int8s())))
			return
		}
		data := t.int8s()
		tmp := make([]int8, 0, lastSize)
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			tmp = append(tmp, data[next])

			if len(tmp) == lastSize {
				am := argmaxI8(tmp)
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
		if it == nil {
			retVal = New(FromScalar(argmaxI16(t.int16s())))
			return
		}
		data := t.int16s()
		tmp := make([]int16, 0, lastSize)
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			tmp = append(tmp, data[next])

			if len(tmp) == lastSize {
				am := argmaxI16(tmp)
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
		if it == nil {
			retVal = New(FromScalar(argmaxI32(t.int32s())))
			return
		}
		data := t.int32s()
		tmp := make([]int32, 0, lastSize)
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			tmp = append(tmp, data[next])

			if len(tmp) == lastSize {
				am := argmaxI32(tmp)
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
		if it == nil {
			retVal = New(FromScalar(argmaxI64(t.int64s())))
			return
		}
		data := t.int64s()
		tmp := make([]int64, 0, lastSize)
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			tmp = append(tmp, data[next])

			if len(tmp) == lastSize {
				am := argmaxI64(tmp)
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
		if it == nil {
			retVal = New(FromScalar(argmaxU(t.uints())))
			return
		}
		data := t.uints()
		tmp := make([]uint, 0, lastSize)
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			tmp = append(tmp, data[next])

			if len(tmp) == lastSize {
				am := argmaxU(tmp)
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
		if it == nil {
			retVal = New(FromScalar(argmaxU8(t.uint8s())))
			return
		}
		data := t.uint8s()
		tmp := make([]uint8, 0, lastSize)
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			tmp = append(tmp, data[next])

			if len(tmp) == lastSize {
				am := argmaxU8(tmp)
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
		if it == nil {
			retVal = New(FromScalar(argmaxU16(t.uint16s())))
			return
		}
		data := t.uint16s()
		tmp := make([]uint16, 0, lastSize)
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			tmp = append(tmp, data[next])

			if len(tmp) == lastSize {
				am := argmaxU16(tmp)
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
		if it == nil {
			retVal = New(FromScalar(argmaxU32(t.uint32s())))
			return
		}
		data := t.uint32s()
		tmp := make([]uint32, 0, lastSize)
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			tmp = append(tmp, data[next])

			if len(tmp) == lastSize {
				am := argmaxU32(tmp)
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
		if it == nil {
			retVal = New(FromScalar(argmaxU64(t.uint64s())))
			return
		}
		data := t.uint64s()
		tmp := make([]uint64, 0, lastSize)
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			tmp = append(tmp, data[next])

			if len(tmp) == lastSize {
				am := argmaxU64(tmp)
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
		if it == nil {
			retVal = New(FromScalar(argmaxF32(t.float32s())))
			return
		}
		data := t.float32s()
		tmp := make([]float32, 0, lastSize)
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			tmp = append(tmp, data[next])

			if len(tmp) == lastSize {
				am := argmaxF32(tmp)
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
		if it == nil {
			retVal = New(FromScalar(argmaxF64(t.float64s())))
			return
		}
		data := t.float64s()
		tmp := make([]float64, 0, lastSize)
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			tmp = append(tmp, data[next])

			if len(tmp) == lastSize {
				am := argmaxF64(tmp)
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

	it := NewFlatIterator(newAP)
	return t.argmin(it)
}

func (t *Dense) argmin(it *FlatIterator) (retVal *Dense, err error) {
	var lastSize, next int
	var newShape Shape
	var indices []int
	if it != nil {
		lastSize = it.Shape()[len(it.Shape())-1]
		newShape = it.Shape().Clone()
		newShape = newShape[:len(newShape)-1]
		defer ReturnInts(newShape)
	}

	switch t.t.Kind() {

	case reflect.Int:
		if it == nil {
			retVal = New(FromScalar(argminI(t.ints())))
			return
		}
		data := t.ints()
		tmp := make([]int, 0, lastSize)
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			tmp = append(tmp, data[next])

			if len(tmp) == lastSize {
				am := argminI(tmp)
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
		if it == nil {
			retVal = New(FromScalar(argminI8(t.int8s())))
			return
		}
		data := t.int8s()
		tmp := make([]int8, 0, lastSize)
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			tmp = append(tmp, data[next])

			if len(tmp) == lastSize {
				am := argminI8(tmp)
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
		if it == nil {
			retVal = New(FromScalar(argminI16(t.int16s())))
			return
		}
		data := t.int16s()
		tmp := make([]int16, 0, lastSize)
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			tmp = append(tmp, data[next])

			if len(tmp) == lastSize {
				am := argminI16(tmp)
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
		if it == nil {
			retVal = New(FromScalar(argminI32(t.int32s())))
			return
		}
		data := t.int32s()
		tmp := make([]int32, 0, lastSize)
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			tmp = append(tmp, data[next])

			if len(tmp) == lastSize {
				am := argminI32(tmp)
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
		if it == nil {
			retVal = New(FromScalar(argminI64(t.int64s())))
			return
		}
		data := t.int64s()
		tmp := make([]int64, 0, lastSize)
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			tmp = append(tmp, data[next])

			if len(tmp) == lastSize {
				am := argminI64(tmp)
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
		if it == nil {
			retVal = New(FromScalar(argminU(t.uints())))
			return
		}
		data := t.uints()
		tmp := make([]uint, 0, lastSize)
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			tmp = append(tmp, data[next])

			if len(tmp) == lastSize {
				am := argminU(tmp)
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
		if it == nil {
			retVal = New(FromScalar(argminU8(t.uint8s())))
			return
		}
		data := t.uint8s()
		tmp := make([]uint8, 0, lastSize)
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			tmp = append(tmp, data[next])

			if len(tmp) == lastSize {
				am := argminU8(tmp)
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
		if it == nil {
			retVal = New(FromScalar(argminU16(t.uint16s())))
			return
		}
		data := t.uint16s()
		tmp := make([]uint16, 0, lastSize)
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			tmp = append(tmp, data[next])

			if len(tmp) == lastSize {
				am := argminU16(tmp)
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
		if it == nil {
			retVal = New(FromScalar(argminU32(t.uint32s())))
			return
		}
		data := t.uint32s()
		tmp := make([]uint32, 0, lastSize)
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			tmp = append(tmp, data[next])

			if len(tmp) == lastSize {
				am := argminU32(tmp)
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
		if it == nil {
			retVal = New(FromScalar(argminU64(t.uint64s())))
			return
		}
		data := t.uint64s()
		tmp := make([]uint64, 0, lastSize)
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			tmp = append(tmp, data[next])

			if len(tmp) == lastSize {
				am := argminU64(tmp)
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
		if it == nil {
			retVal = New(FromScalar(argminF32(t.float32s())))
			return
		}
		data := t.float32s()
		tmp := make([]float32, 0, lastSize)
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			tmp = append(tmp, data[next])

			if len(tmp) == lastSize {
				am := argminF32(tmp)
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
		if it == nil {
			retVal = New(FromScalar(argminF64(t.float64s())))
			return
		}
		data := t.float64s()
		tmp := make([]float64, 0, lastSize)
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			tmp = append(tmp, data[next])

			if len(tmp) == lastSize {
				am := argminF64(tmp)
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
