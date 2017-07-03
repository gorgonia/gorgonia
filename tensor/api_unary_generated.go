package tensor

import (
	"math"
	"reflect"

	"github.com/chewxy/math32"
	"github.com/pkg/errors"
)

/*
GENERATED FILE. DO NOT EDIT
*/

// Clamp clamps the values of the Tensor to the min and max provided. The min and max provided must be the same type as the Tensor type.
// Incr is not supported (it doesn't make sense anyway)
func Clamp(a Tensor, minVal, maxVal interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	switch t := a.(type) {
	case *Dense:
		if t.IsMaterializable() {
			var f interface{}
			switch t.t.Kind() {
			case reflect.Int:
				var min, max int
				var ok bool
				if min, ok = minVal.(int); !ok {
					err = errors.Wrapf(errors.Errorf(typeMismatch, min, minVal), "Clamp() min")
					return
				}
				if max, ok = maxVal.(int); !ok {
					err = errors.Wrapf(errors.Errorf(typeMismatch, max, maxVal), "Clamp() max")
					return
				}
				f = func(x int) int {
					if x < min {
						return min
					}
					if x > max {
						return max
					}
					return x
				}
			case reflect.Int8:
				var min, max int8
				var ok bool
				if min, ok = minVal.(int8); !ok {
					err = errors.Wrapf(errors.Errorf(typeMismatch, min, minVal), "Clamp() min")
					return
				}
				if max, ok = maxVal.(int8); !ok {
					err = errors.Wrapf(errors.Errorf(typeMismatch, max, maxVal), "Clamp() max")
					return
				}
				f = func(x int8) int8 {
					if x < min {
						return min
					}
					if x > max {
						return max
					}
					return x
				}
			case reflect.Int16:
				var min, max int16
				var ok bool
				if min, ok = minVal.(int16); !ok {
					err = errors.Wrapf(errors.Errorf(typeMismatch, min, minVal), "Clamp() min")
					return
				}
				if max, ok = maxVal.(int16); !ok {
					err = errors.Wrapf(errors.Errorf(typeMismatch, max, maxVal), "Clamp() max")
					return
				}
				f = func(x int16) int16 {
					if x < min {
						return min
					}
					if x > max {
						return max
					}
					return x
				}
			case reflect.Int32:
				var min, max int32
				var ok bool
				if min, ok = minVal.(int32); !ok {
					err = errors.Wrapf(errors.Errorf(typeMismatch, min, minVal), "Clamp() min")
					return
				}
				if max, ok = maxVal.(int32); !ok {
					err = errors.Wrapf(errors.Errorf(typeMismatch, max, maxVal), "Clamp() max")
					return
				}
				f = func(x int32) int32 {
					if x < min {
						return min
					}
					if x > max {
						return max
					}
					return x
				}
			case reflect.Int64:
				var min, max int64
				var ok bool
				if min, ok = minVal.(int64); !ok {
					err = errors.Wrapf(errors.Errorf(typeMismatch, min, minVal), "Clamp() min")
					return
				}
				if max, ok = maxVal.(int64); !ok {
					err = errors.Wrapf(errors.Errorf(typeMismatch, max, maxVal), "Clamp() max")
					return
				}
				f = func(x int64) int64 {
					if x < min {
						return min
					}
					if x > max {
						return max
					}
					return x
				}
			case reflect.Uint:
				var min, max uint
				var ok bool
				if min, ok = minVal.(uint); !ok {
					err = errors.Wrapf(errors.Errorf(typeMismatch, min, minVal), "Clamp() min")
					return
				}
				if max, ok = maxVal.(uint); !ok {
					err = errors.Wrapf(errors.Errorf(typeMismatch, max, maxVal), "Clamp() max")
					return
				}
				f = func(x uint) uint {
					if x < min {
						return min
					}
					if x > max {
						return max
					}
					return x
				}
			case reflect.Uint8:
				var min, max uint8
				var ok bool
				if min, ok = minVal.(uint8); !ok {
					err = errors.Wrapf(errors.Errorf(typeMismatch, min, minVal), "Clamp() min")
					return
				}
				if max, ok = maxVal.(uint8); !ok {
					err = errors.Wrapf(errors.Errorf(typeMismatch, max, maxVal), "Clamp() max")
					return
				}
				f = func(x uint8) uint8 {
					if x < min {
						return min
					}
					if x > max {
						return max
					}
					return x
				}
			case reflect.Uint16:
				var min, max uint16
				var ok bool
				if min, ok = minVal.(uint16); !ok {
					err = errors.Wrapf(errors.Errorf(typeMismatch, min, minVal), "Clamp() min")
					return
				}
				if max, ok = maxVal.(uint16); !ok {
					err = errors.Wrapf(errors.Errorf(typeMismatch, max, maxVal), "Clamp() max")
					return
				}
				f = func(x uint16) uint16 {
					if x < min {
						return min
					}
					if x > max {
						return max
					}
					return x
				}
			case reflect.Uint32:
				var min, max uint32
				var ok bool
				if min, ok = minVal.(uint32); !ok {
					err = errors.Wrapf(errors.Errorf(typeMismatch, min, minVal), "Clamp() min")
					return
				}
				if max, ok = maxVal.(uint32); !ok {
					err = errors.Wrapf(errors.Errorf(typeMismatch, max, maxVal), "Clamp() max")
					return
				}
				f = func(x uint32) uint32 {
					if x < min {
						return min
					}
					if x > max {
						return max
					}
					return x
				}
			case reflect.Uint64:
				var min, max uint64
				var ok bool
				if min, ok = minVal.(uint64); !ok {
					err = errors.Wrapf(errors.Errorf(typeMismatch, min, minVal), "Clamp() min")
					return
				}
				if max, ok = maxVal.(uint64); !ok {
					err = errors.Wrapf(errors.Errorf(typeMismatch, max, maxVal), "Clamp() max")
					return
				}
				f = func(x uint64) uint64 {
					if x < min {
						return min
					}
					if x > max {
						return max
					}
					return x
				}
			case reflect.Float32:
				var min, max float32
				var ok bool
				if min, ok = minVal.(float32); !ok {
					err = errors.Wrapf(errors.Errorf(typeMismatch, min, minVal), "Clamp() min")
					return
				}
				if max, ok = maxVal.(float32); !ok {
					err = errors.Wrapf(errors.Errorf(typeMismatch, max, maxVal), "Clamp() max")
					return
				}
				f = func(x float32) float32 {
					if x < min || math32.IsInf(x, -1) {
						return min
					}
					if x > max || math32.IsInf(x, 1) {
						return max
					}
					return x
				}
			case reflect.Float64:
				var min, max float64
				var ok bool
				if min, ok = minVal.(float64); !ok {
					err = errors.Wrapf(errors.Errorf(typeMismatch, min, minVal), "Clamp() min")
					return
				}
				if max, ok = maxVal.(float64); !ok {
					err = errors.Wrapf(errors.Errorf(typeMismatch, max, maxVal), "Clamp() max")
					return
				}
				f = func(x float64) float64 {
					if x < min || math.IsInf(x, -1) {
						return min
					}
					if x > max || math.IsInf(x, 1) {
						return max
					}
					return x
				}
			}
			return t.Apply(f, opts...)
		}

		if !isNumber(t.t) {
			err = errors.Errorf("Clamp only works on numbers")
			return
		}

		// otherwise, we have optimizations for this (basically remove the repeated function calls)
		var reuse *Dense
		var safe, toReuse, incr bool
		if reuse, safe, toReuse, incr, err = prepUnaryDense(t, opts...); err != nil {
			err = errors.Wrapf(err, opFail, "Clamp")
			return
		}

		var ret *Dense
		switch {
		case incr:
			fallthrough
		case toReuse:
			copyDense(reuse, t)
			ret = reuse
		case safe:
			ret = t.Clone().(*Dense)
		case !safe:
			ret = t
		}

		switch t.t.Kind() {
		case reflect.Int:
			var min, max int
			var ok bool
			if min, ok = minVal.(int); !ok {
				err = errors.Wrapf(errors.Errorf(typeMismatch, min, minVal), "Clamp() min")
				return
			}
			if max, ok = maxVal.(int); !ok {
				err = errors.Wrapf(errors.Errorf(typeMismatch, max, maxVal), "Clamp() max")
				return
			}
			data := ret.ints()

			if !ret.IsMasked() {
				for i, v := range data {
					if v < min {
						data[i] = min
						continue
					}
					if v > max {
						data[i] = max
					}
				}
			} else {
				for i, v := range data {
					if !ret.mask[i] {
						if v < min {
							data[i] = min
							continue
						}
						if v > max {
							data[i] = max
						}
					}
				}
			}
		case reflect.Int8:
			var min, max int8
			var ok bool
			if min, ok = minVal.(int8); !ok {
				err = errors.Wrapf(errors.Errorf(typeMismatch, min, minVal), "Clamp() min")
				return
			}
			if max, ok = maxVal.(int8); !ok {
				err = errors.Wrapf(errors.Errorf(typeMismatch, max, maxVal), "Clamp() max")
				return
			}
			data := ret.int8s()

			if !ret.IsMasked() {
				for i, v := range data {
					if v < min {
						data[i] = min
						continue
					}
					if v > max {
						data[i] = max
					}
				}
			} else {
				for i, v := range data {
					if !ret.mask[i] {
						if v < min {
							data[i] = min
							continue
						}
						if v > max {
							data[i] = max
						}
					}
				}
			}
		case reflect.Int16:
			var min, max int16
			var ok bool
			if min, ok = minVal.(int16); !ok {
				err = errors.Wrapf(errors.Errorf(typeMismatch, min, minVal), "Clamp() min")
				return
			}
			if max, ok = maxVal.(int16); !ok {
				err = errors.Wrapf(errors.Errorf(typeMismatch, max, maxVal), "Clamp() max")
				return
			}
			data := ret.int16s()

			if !ret.IsMasked() {
				for i, v := range data {
					if v < min {
						data[i] = min
						continue
					}
					if v > max {
						data[i] = max
					}
				}
			} else {
				for i, v := range data {
					if !ret.mask[i] {
						if v < min {
							data[i] = min
							continue
						}
						if v > max {
							data[i] = max
						}
					}
				}
			}
		case reflect.Int32:
			var min, max int32
			var ok bool
			if min, ok = minVal.(int32); !ok {
				err = errors.Wrapf(errors.Errorf(typeMismatch, min, minVal), "Clamp() min")
				return
			}
			if max, ok = maxVal.(int32); !ok {
				err = errors.Wrapf(errors.Errorf(typeMismatch, max, maxVal), "Clamp() max")
				return
			}
			data := ret.int32s()

			if !ret.IsMasked() {
				for i, v := range data {
					if v < min {
						data[i] = min
						continue
					}
					if v > max {
						data[i] = max
					}
				}
			} else {
				for i, v := range data {
					if !ret.mask[i] {
						if v < min {
							data[i] = min
							continue
						}
						if v > max {
							data[i] = max
						}
					}
				}
			}
		case reflect.Int64:
			var min, max int64
			var ok bool
			if min, ok = minVal.(int64); !ok {
				err = errors.Wrapf(errors.Errorf(typeMismatch, min, minVal), "Clamp() min")
				return
			}
			if max, ok = maxVal.(int64); !ok {
				err = errors.Wrapf(errors.Errorf(typeMismatch, max, maxVal), "Clamp() max")
				return
			}
			data := ret.int64s()

			if !ret.IsMasked() {
				for i, v := range data {
					if v < min {
						data[i] = min
						continue
					}
					if v > max {
						data[i] = max
					}
				}
			} else {
				for i, v := range data {
					if !ret.mask[i] {
						if v < min {
							data[i] = min
							continue
						}
						if v > max {
							data[i] = max
						}
					}
				}
			}
		case reflect.Uint:
			var min, max uint
			var ok bool
			if min, ok = minVal.(uint); !ok {
				err = errors.Wrapf(errors.Errorf(typeMismatch, min, minVal), "Clamp() min")
				return
			}
			if max, ok = maxVal.(uint); !ok {
				err = errors.Wrapf(errors.Errorf(typeMismatch, max, maxVal), "Clamp() max")
				return
			}
			data := ret.uints()

			if !ret.IsMasked() {
				for i, v := range data {
					if v < min {
						data[i] = min
						continue
					}
					if v > max {
						data[i] = max
					}
				}
			} else {
				for i, v := range data {
					if !ret.mask[i] {
						if v < min {
							data[i] = min
							continue
						}
						if v > max {
							data[i] = max
						}
					}
				}
			}
		case reflect.Uint8:
			var min, max uint8
			var ok bool
			if min, ok = minVal.(uint8); !ok {
				err = errors.Wrapf(errors.Errorf(typeMismatch, min, minVal), "Clamp() min")
				return
			}
			if max, ok = maxVal.(uint8); !ok {
				err = errors.Wrapf(errors.Errorf(typeMismatch, max, maxVal), "Clamp() max")
				return
			}
			data := ret.uint8s()

			if !ret.IsMasked() {
				for i, v := range data {
					if v < min {
						data[i] = min
						continue
					}
					if v > max {
						data[i] = max
					}
				}
			} else {
				for i, v := range data {
					if !ret.mask[i] {
						if v < min {
							data[i] = min
							continue
						}
						if v > max {
							data[i] = max
						}
					}
				}
			}
		case reflect.Uint16:
			var min, max uint16
			var ok bool
			if min, ok = minVal.(uint16); !ok {
				err = errors.Wrapf(errors.Errorf(typeMismatch, min, minVal), "Clamp() min")
				return
			}
			if max, ok = maxVal.(uint16); !ok {
				err = errors.Wrapf(errors.Errorf(typeMismatch, max, maxVal), "Clamp() max")
				return
			}
			data := ret.uint16s()

			if !ret.IsMasked() {
				for i, v := range data {
					if v < min {
						data[i] = min
						continue
					}
					if v > max {
						data[i] = max
					}
				}
			} else {
				for i, v := range data {
					if !ret.mask[i] {
						if v < min {
							data[i] = min
							continue
						}
						if v > max {
							data[i] = max
						}
					}
				}
			}
		case reflect.Uint32:
			var min, max uint32
			var ok bool
			if min, ok = minVal.(uint32); !ok {
				err = errors.Wrapf(errors.Errorf(typeMismatch, min, minVal), "Clamp() min")
				return
			}
			if max, ok = maxVal.(uint32); !ok {
				err = errors.Wrapf(errors.Errorf(typeMismatch, max, maxVal), "Clamp() max")
				return
			}
			data := ret.uint32s()

			if !ret.IsMasked() {
				for i, v := range data {
					if v < min {
						data[i] = min
						continue
					}
					if v > max {
						data[i] = max
					}
				}
			} else {
				for i, v := range data {
					if !ret.mask[i] {
						if v < min {
							data[i] = min
							continue
						}
						if v > max {
							data[i] = max
						}
					}
				}
			}
		case reflect.Uint64:
			var min, max uint64
			var ok bool
			if min, ok = minVal.(uint64); !ok {
				err = errors.Wrapf(errors.Errorf(typeMismatch, min, minVal), "Clamp() min")
				return
			}
			if max, ok = maxVal.(uint64); !ok {
				err = errors.Wrapf(errors.Errorf(typeMismatch, max, maxVal), "Clamp() max")
				return
			}
			data := ret.uint64s()

			if !ret.IsMasked() {
				for i, v := range data {
					if v < min {
						data[i] = min
						continue
					}
					if v > max {
						data[i] = max
					}
				}
			} else {
				for i, v := range data {
					if !ret.mask[i] {
						if v < min {
							data[i] = min
							continue
						}
						if v > max {
							data[i] = max
						}
					}
				}
			}
		case reflect.Float32:
			var min, max float32
			var ok bool
			if min, ok = minVal.(float32); !ok {
				err = errors.Wrapf(errors.Errorf(typeMismatch, min, minVal), "Clamp() min")
				return
			}
			if max, ok = maxVal.(float32); !ok {
				err = errors.Wrapf(errors.Errorf(typeMismatch, max, maxVal), "Clamp() max")
				return
			}
			data := ret.float32s()

			if !ret.IsMasked() {
				for i, v := range data {
					if v < min || math32.IsInf(v, -1) {
						data[i] = min
						continue
					}
					if v > max || math32.IsInf(v, 1) {
						data[i] = max
					}
				}
			} else {
				for i, v := range data {
					if !ret.mask[i] {
						if v < min || math32.IsInf(v, -1) {
							data[i] = min
							continue
						}
						if v > max || math32.IsInf(v, 1) {
							data[i] = max
						}
					}
				}
			}
		case reflect.Float64:
			var min, max float64
			var ok bool
			if min, ok = minVal.(float64); !ok {
				err = errors.Wrapf(errors.Errorf(typeMismatch, min, minVal), "Clamp() min")
				return
			}
			if max, ok = maxVal.(float64); !ok {
				err = errors.Wrapf(errors.Errorf(typeMismatch, max, maxVal), "Clamp() max")
				return
			}
			data := ret.float64s()

			if !ret.IsMasked() {
				for i, v := range data {
					if v < min || math.IsInf(v, -1) {
						data[i] = min
						continue
					}
					if v > max || math.IsInf(v, 1) {
						data[i] = max
					}
				}
			} else {
				for i, v := range data {
					if !ret.mask[i] {
						if v < min || math.IsInf(v, -1) {
							data[i] = min
							continue
						}
						if v > max || math.IsInf(v, 1) {
							data[i] = max
						}
					}
				}
			}
		}
		retVal = ret
		return
	default:
		return nil, errors.Errorf(typeNYI, "Clamp", a)
	}
}

// Sign returns the sign function as applied to each element in the ndarray. It does not yet support the incr option.
// Incr is not supported (it doesn't make sense anyway)
func Sign(a Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	switch t := a.(type) {
	case *Dense:
		if !isNumber(t.t) {
			err = errors.Errorf("Sign only works on numbers")
			return
		}

		if t.IsMaterializable() {
			var f interface{}
			switch t.t.Kind() {
			case reflect.Int:
				f = func(x int) int {
					if x < 0 {
						return -1
					}
					if x > 0 {
						return 1
					}
					return 0
				}
			case reflect.Int8:
				f = func(x int8) int8 {
					if x < 0 {
						return -1
					}
					if x > 0 {
						return 1
					}
					return 0
				}
			case reflect.Int16:
				f = func(x int16) int16 {
					if x < 0 {
						return -1
					}
					if x > 0 {
						return 1
					}
					return 0
				}
			case reflect.Int32:
				f = func(x int32) int32 {
					if x < 0 {
						return -1
					}
					if x > 0 {
						return 1
					}
					return 0
				}
			case reflect.Int64:
				f = func(x int64) int64 {
					if x < 0 {
						return -1
					}
					if x > 0 {
						return 1
					}
					return 0
				}
			case reflect.Uint:
				f = func(x uint) uint {
					if x > 0 {
						return 1
					}
					return 0
				}
			case reflect.Uint8:
				f = func(x uint8) uint8 {
					if x > 0 {
						return 1
					}
					return 0
				}
			case reflect.Uint16:
				f = func(x uint16) uint16 {
					if x > 0 {
						return 1
					}
					return 0
				}
			case reflect.Uint32:
				f = func(x uint32) uint32 {
					if x > 0 {
						return 1
					}
					return 0
				}
			case reflect.Uint64:
				f = func(x uint64) uint64 {
					if x > 0 {
						return 1
					}
					return 0
				}
			case reflect.Float32:
				f = func(x float32) float32 {
					if x < 0 {
						return -1
					}
					if x > 0 {
						return 1
					}
					return 0
				}
			case reflect.Float64:
				f = func(x float64) float64 {
					if x < 0 {
						return -1
					}
					if x > 0 {
						return 1
					}
					return 0
				}
			}
			return t.Apply(f, opts...)
		}

		// otherwise, we have optimizations for this (basically remove the repeated function calls)
		var reuse *Dense
		var safe, toReuse, incr bool
		if reuse, safe, toReuse, incr, err = prepUnaryDense(t, opts...); err != nil {
			err = errors.Wrapf(err, opFail, "Sign")
			return
		}

		var ret *Dense
		switch {
		case incr:
			fallthrough
		case toReuse:
			copyDense(reuse, t)
			ret = reuse
		case safe:
			ret = t.Clone().(*Dense)
		case !safe:
			ret = t
		}

		switch t.t.Kind() {
		case reflect.Int:
			data := ret.ints()
			if !ret.IsMasked() {
				for i, v := range data {
					if v < 0 {
						data[i] = -1
						continue
					}
					if v > 0 {
						data[i] = 1
						continue
					}
					data[i] = 0
				}
			} else {
				for i, v := range data {
					if !ret.mask[i] {
						if v < 0 {
							data[i] = -1
							continue
						}
						if v > 0 {
							data[i] = 1
							continue
						}
						data[i] = 0
					}
				}
			}
		case reflect.Int8:
			data := ret.int8s()
			if !ret.IsMasked() {
				for i, v := range data {
					if v < 0 {
						data[i] = -1
						continue
					}
					if v > 0 {
						data[i] = 1
						continue
					}
					data[i] = 0
				}
			} else {
				for i, v := range data {
					if !ret.mask[i] {
						if v < 0 {
							data[i] = -1
							continue
						}
						if v > 0 {
							data[i] = 1
							continue
						}
						data[i] = 0
					}
				}
			}
		case reflect.Int16:
			data := ret.int16s()
			if !ret.IsMasked() {
				for i, v := range data {
					if v < 0 {
						data[i] = -1
						continue
					}
					if v > 0 {
						data[i] = 1
						continue
					}
					data[i] = 0
				}
			} else {
				for i, v := range data {
					if !ret.mask[i] {
						if v < 0 {
							data[i] = -1
							continue
						}
						if v > 0 {
							data[i] = 1
							continue
						}
						data[i] = 0
					}
				}
			}
		case reflect.Int32:
			data := ret.int32s()
			if !ret.IsMasked() {
				for i, v := range data {
					if v < 0 {
						data[i] = -1
						continue
					}
					if v > 0 {
						data[i] = 1
						continue
					}
					data[i] = 0
				}
			} else {
				for i, v := range data {
					if !ret.mask[i] {
						if v < 0 {
							data[i] = -1
							continue
						}
						if v > 0 {
							data[i] = 1
							continue
						}
						data[i] = 0
					}
				}
			}
		case reflect.Int64:
			data := ret.int64s()
			if !ret.IsMasked() {
				for i, v := range data {
					if v < 0 {
						data[i] = -1
						continue
					}
					if v > 0 {
						data[i] = 1
						continue
					}
					data[i] = 0
				}
			} else {
				for i, v := range data {
					if !ret.mask[i] {
						if v < 0 {
							data[i] = -1
							continue
						}
						if v > 0 {
							data[i] = 1
							continue
						}
						data[i] = 0
					}
				}
			}
		case reflect.Uint:
			data := ret.uints()
			if !ret.IsMasked() {
				for i := range data {
					if data[i] > 0 {
						data[i] = 1
					}
				}
			} else {
				for i := range data {
					if !ret.mask[i] {
						if data[i] > 0 {
							data[i] = 1
						}
					}
				}
			}
		case reflect.Uint8:
			data := ret.uint8s()
			if !ret.IsMasked() {
				for i := range data {
					if data[i] > 0 {
						data[i] = 1
					}
				}
			} else {
				for i := range data {
					if !ret.mask[i] {
						if data[i] > 0 {
							data[i] = 1
						}
					}
				}
			}
		case reflect.Uint16:
			data := ret.uint16s()
			if !ret.IsMasked() {
				for i := range data {
					if data[i] > 0 {
						data[i] = 1
					}
				}
			} else {
				for i := range data {
					if !ret.mask[i] {
						if data[i] > 0 {
							data[i] = 1
						}
					}
				}
			}
		case reflect.Uint32:
			data := ret.uint32s()
			if !ret.IsMasked() {
				for i := range data {
					if data[i] > 0 {
						data[i] = 1
					}
				}
			} else {
				for i := range data {
					if !ret.mask[i] {
						if data[i] > 0 {
							data[i] = 1
						}
					}
				}
			}
		case reflect.Uint64:
			data := ret.uint64s()
			if !ret.IsMasked() {
				for i := range data {
					if data[i] > 0 {
						data[i] = 1
					}
				}
			} else {
				for i := range data {
					if !ret.mask[i] {
						if data[i] > 0 {
							data[i] = 1
						}
					}
				}
			}
		case reflect.Float32:
			data := ret.float32s()
			if !ret.IsMasked() {
				for i, v := range data {
					if v < 0 {
						data[i] = -1
						continue
					}
					if v > 0 {
						data[i] = 1
						continue
					}
					data[i] = 0
				}
			} else {
				for i, v := range data {
					if !ret.mask[i] {
						if v < 0 {
							data[i] = -1
							continue
						}
						if v > 0 {
							data[i] = 1
							continue
						}
						data[i] = 0
					}
				}
			}
		case reflect.Float64:
			data := ret.float64s()
			if !ret.IsMasked() {
				for i, v := range data {
					if v < 0 {
						data[i] = -1
						continue
					}
					if v > 0 {
						data[i] = 1
						continue
					}
					data[i] = 0
				}
			} else {
				for i, v := range data {
					if !ret.mask[i] {
						if v < 0 {
							data[i] = -1
							continue
						}
						if v > 0 {
							data[i] = 1
							continue
						}
						data[i] = 0
					}
				}
			}
		}
		retVal = ret
		return
	default:
		return nil, errors.Errorf(typeNYI, "Sign", a)
	}
}

// Neg returns the sign function as applied to each element in the ndarray.
// Incr is not supported (it doesn't make sense anyway - you'd just call Sub())
func Neg(a Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	switch t := a.(type) {
	case *Dense:
		if !isNumber(t.t) || isUnsigned(t.t) {
			err = errors.Errorf("Neg only works on signed numbers")
			return
		}

		if t.IsMaterializable() {
			var f interface{}
			switch t.t.Kind() {

			case reflect.Int:
				f = func(x int) int {
					return -x
				}

			case reflect.Int8:
				f = func(x int8) int8 {
					return -x
				}

			case reflect.Int16:
				f = func(x int16) int16 {
					return -x
				}

			case reflect.Int32:
				f = func(x int32) int32 {
					return -x
				}

			case reflect.Int64:
				f = func(x int64) int64 {
					return -x
				}

			case reflect.Float32:
				f = func(x float32) float32 {
					return -x
				}

			case reflect.Float64:
				f = func(x float64) float64 {
					return -x
				}

			case reflect.Complex64:
				f = func(x complex64) complex64 {
					return -x
				}

			case reflect.Complex128:
				f = func(x complex128) complex128 {
					return -x
				}

			}
			return t.Apply(f, opts...)
		}

		// otherwise, we have optimizations for this (basically remove the repeated function calls)
		var reuse *Dense
		var safe, toReuse, incr bool
		if reuse, safe, toReuse, incr, err = prepUnaryDense(t, opts...); err != nil {
			err = errors.Wrapf(err, opFail, "Neg")
			return
		}

		var ret *Dense
		switch {
		case incr:
			fallthrough
		case toReuse:
			copyDense(reuse, t)
			ret = reuse
		case safe:
			ret = t.Clone().(*Dense)
		case !safe:
			ret = t
		}

		switch t.t.Kind() {
		case reflect.Int:
			data := ret.ints()
			if !ret.IsMasked() {
				for i := range data {
					data[i] = -data[i]
				}
			} else {
				for i := range data {
					if !ret.mask[i] {
						data[i] = -data[i]
					}
				}
			}

		case reflect.Int8:
			data := ret.int8s()
			if !ret.IsMasked() {
				for i := range data {
					data[i] = -data[i]
				}
			} else {
				for i := range data {
					if !ret.mask[i] {
						data[i] = -data[i]
					}
				}
			}

		case reflect.Int16:
			data := ret.int16s()
			if !ret.IsMasked() {
				for i := range data {
					data[i] = -data[i]
				}
			} else {
				for i := range data {
					if !ret.mask[i] {
						data[i] = -data[i]
					}
				}
			}

		case reflect.Int32:
			data := ret.int32s()
			if !ret.IsMasked() {
				for i := range data {
					data[i] = -data[i]
				}
			} else {
				for i := range data {
					if !ret.mask[i] {
						data[i] = -data[i]
					}
				}
			}

		case reflect.Int64:
			data := ret.int64s()
			if !ret.IsMasked() {
				for i := range data {
					data[i] = -data[i]
				}
			} else {
				for i := range data {
					if !ret.mask[i] {
						data[i] = -data[i]
					}
				}
			}

		case reflect.Float32:
			data := ret.float32s()
			if !ret.IsMasked() {
				for i := range data {
					data[i] = -data[i]
				}
			} else {
				for i := range data {
					if !ret.mask[i] {
						data[i] = -data[i]
					}
				}
			}

		case reflect.Float64:
			data := ret.float64s()
			if !ret.IsMasked() {
				for i := range data {
					data[i] = -data[i]
				}
			} else {
				for i := range data {
					if !ret.mask[i] {
						data[i] = -data[i]
					}
				}
			}

		case reflect.Complex64:
			data := ret.complex64s()
			if !ret.IsMasked() {
				for i := range data {
					data[i] = -data[i]
				}
			} else {
				for i := range data {
					if !ret.mask[i] {
						data[i] = -data[i]
					}
				}
			}

		case reflect.Complex128:
			data := ret.complex128s()
			if !ret.IsMasked() {
				for i := range data {
					data[i] = -data[i]
				}
			} else {
				for i := range data {
					if !ret.mask[i] {
						data[i] = -data[i]
					}
				}
			}

		}
		retVal = ret
		return
	default:
		return nil, errors.Errorf(typeNYI, "Neg", a)
	}
}
