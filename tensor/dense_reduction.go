package tensor

import (
	"reflect"
	"unsafe"

	"github.com/pkg/errors"
)

/*
GENERATED FILE. DO NOT EDIT
*/

/*
This file contains code that deals with the reduction of a Tensor by axis.


All of the code in this file is structured in such a way that they're embarassingly parallel.
This message will serve as a reminder until all the code in this file which are embarassingly parallel
has been parallelized

List of functions parallalized:
	<crickets>

A visual explanation for the main reduction algorithm:
			Say you have a (2,3,2,3)-shaped tensor. It looks something like that:

				0  1  2		18 19 20
				3  4  5		21 22 23

				6  7  8		24 25 26
				9 10 11		27 28 29

				12 13 14	30 31 32
				15 16 17	33 34 35

			We'll consider only the first layer (0 - 17), since the same actions can be repeated upon the second layer

			Let's say we want to reduce axis 2. The resulting shape would be (2,3,3) (it's as simple as removing the second axis from the shape).
			This is how the matrix is laid out in the strided slice:

			t.data:
				0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16  17
				+   +   +   +   +   +   +   +   +   +    +   +  +   +   +    +   +   +
				|   |   |   |   |   |   |   |   |   |    |   |  |   |   |    |   |   |
				|   |   |   |   |   |   |   |   |   |    |   |  |   |   |    |   |   |
				+---------------------+-+-----------------------+   |   |    |   |   |
				    |   |   |   |   | |     |   |   |    |   |      |   |    |   |   |
				    +--------------------+--+-----------------------+   |    |   |   |
				        |   |   |   | |  |      |   |    |   |          |    |   |   |
				        +-----------------------------------------------+    |   |   |
				            |   |   | |  |      |   |    |   |               |   |   |
				            |   |   | +  +      +   |    |   |               |   |   |
			res.data index  |   |   | 0  1      2   |    |   |               |   |   |
				            |   |   |               |    |   |               |   |   |
				            +----------------------------+-+-----------------+   |   |
				                |   |               |      | |                   |   |
				                +------------------------------------------------+---+
				                    |               |      | |                       |
				                    +------------------------+-----+-----------------+
				                                    |      |       |
				                                    |      |       |
				                                    +      +       +
			res.data indes                          3      4       5

			It's a little difficult to see, but elements (0, 6, 12) from t.data will be written to index 0 of the reduced strided array. This is the listing:
				reduce (t[0], t[6], t[12]) -> res[0]
				reduce (t[1], t[7], t[13]) -> res[1]
				reduce (t[2], t[8], t[14]) -> res[2]
				...

			These are the basic rules:
				size of axis to be reduced  = number of elements to be reduced
				stride of axis to be reduced = how many to skip innerStart
				newStride[0] = expected number of groups within a layer

			The main idea is then this - we loop through the resulting array, and for each index, we find the elements of the original array that is supposed to fit in
			there, and then we reduce it. It is quite self explanatory.
*/

func reductionFnType(x interface{}, expectedType reflect.Type) (v reflect.Value, t reflect.Type, err error) {
	v = reflect.ValueOf(x)
	if v.Kind() != reflect.Func {
		err = errors.Errorf(extractionFail, "func(a, a) a", x)
		return
	}
	t = v.Type()
	if t.NumOut() != 1 {
		err = errors.Errorf("Expected one return value in reduction function")
		return
	}
	if t.Out(0) != expectedType {
		err = errors.Errorf("Expected return type of reduction function to be %v. Got %v instead", expectedType, t.Out(0))
		return
	}
	return
}

// Reduce recursively applies a function f on the data along the provided axis. A default value has to be provided.
// The provided function must have this signature:
// 		func(T, T) T
// where T is the same as the Dtype of *Dense
func (t *Dense) Reduce(f interface{}, defaultValue interface{}, axis int) (retVal *Dense, err error) {
	if axis >= t.Dims() {
		err = errors.Errorf(dimMismatch, axis, t.Dims())
		return
	}

	var newShape Shape
	for i, s := range t.Shape() {
		if i == axis {
			continue
		}
		newShape = append(newShape, s)
	}
	lastAxis := t.Dims() - 1
	retVal = New(Of(t.t), WithShape(newShape...))

	switch axis {
	case 0:
		err = t.reduce0(retVal, f)
	case lastAxis:
		err = t.reduceLast(retVal, axis, f, defaultValue)
	default:
		err = t.reduceDefault(retVal, axis, f)
	}
	return
}

func (t *Dense) reduce0(retVal *Dense, fn interface{}) (err error) {
	size := t.Shape()[0]
	split := t.len() / size
	copySliced(retVal, 0, split, t, 0, split)
	start := split

	var ok bool
	switch t.t.Kind() {
	case reflect.Bool:
		var f func(a, b bool) bool
		if f, ok = fn.(func(a, b bool) bool); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b bool) bool. Got %v instead", fn)
		}

		data := retVal.Bools()
		for i := 0; i < size-1; i++ {
			for j := 0; j < split; j++ {
				data[j] = f(data[j], t.GetB(j+start))
			}
			start += split
		}
	case reflect.Int:
		var f func(a, b int) int
		if f, ok = fn.(func(a, b int) int); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b int) int. Got %v instead", fn)
		}

		data := retVal.Ints()
		for i := 0; i < size-1; i++ {
			for j := 0; j < split; j++ {
				data[j] = f(data[j], t.GetI(j+start))
			}
			start += split
		}
	case reflect.Int8:
		var f func(a, b int8) int8
		if f, ok = fn.(func(a, b int8) int8); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b int8) int8. Got %v instead", fn)
		}

		data := retVal.Int8s()
		for i := 0; i < size-1; i++ {
			for j := 0; j < split; j++ {
				data[j] = f(data[j], t.GetI8(j+start))
			}
			start += split
		}
	case reflect.Int16:
		var f func(a, b int16) int16
		if f, ok = fn.(func(a, b int16) int16); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b int16) int16. Got %v instead", fn)
		}

		data := retVal.Int16s()
		for i := 0; i < size-1; i++ {
			for j := 0; j < split; j++ {
				data[j] = f(data[j], t.GetI16(j+start))
			}
			start += split
		}
	case reflect.Int32:
		var f func(a, b int32) int32
		if f, ok = fn.(func(a, b int32) int32); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b int32) int32. Got %v instead", fn)
		}

		data := retVal.Int32s()
		for i := 0; i < size-1; i++ {
			for j := 0; j < split; j++ {
				data[j] = f(data[j], t.GetI32(j+start))
			}
			start += split
		}
	case reflect.Int64:
		var f func(a, b int64) int64
		if f, ok = fn.(func(a, b int64) int64); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b int64) int64. Got %v instead", fn)
		}

		data := retVal.Int64s()
		for i := 0; i < size-1; i++ {
			for j := 0; j < split; j++ {
				data[j] = f(data[j], t.GetI64(j+start))
			}
			start += split
		}
	case reflect.Uint:
		var f func(a, b uint) uint
		if f, ok = fn.(func(a, b uint) uint); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b uint) uint. Got %v instead", fn)
		}

		data := retVal.Uints()
		for i := 0; i < size-1; i++ {
			for j := 0; j < split; j++ {
				data[j] = f(data[j], t.GetU(j+start))
			}
			start += split
		}
	case reflect.Uint8:
		var f func(a, b uint8) uint8
		if f, ok = fn.(func(a, b uint8) uint8); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b uint8) uint8. Got %v instead", fn)
		}

		data := retVal.Uint8s()
		for i := 0; i < size-1; i++ {
			for j := 0; j < split; j++ {
				data[j] = f(data[j], t.GetU8(j+start))
			}
			start += split
		}
	case reflect.Uint16:
		var f func(a, b uint16) uint16
		if f, ok = fn.(func(a, b uint16) uint16); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b uint16) uint16. Got %v instead", fn)
		}

		data := retVal.Uint16s()
		for i := 0; i < size-1; i++ {
			for j := 0; j < split; j++ {
				data[j] = f(data[j], t.GetU16(j+start))
			}
			start += split
		}
	case reflect.Uint32:
		var f func(a, b uint32) uint32
		if f, ok = fn.(func(a, b uint32) uint32); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b uint32) uint32. Got %v instead", fn)
		}

		data := retVal.Uint32s()
		for i := 0; i < size-1; i++ {
			for j := 0; j < split; j++ {
				data[j] = f(data[j], t.GetU32(j+start))
			}
			start += split
		}
	case reflect.Uint64:
		var f func(a, b uint64) uint64
		if f, ok = fn.(func(a, b uint64) uint64); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b uint64) uint64. Got %v instead", fn)
		}

		data := retVal.Uint64s()
		for i := 0; i < size-1; i++ {
			for j := 0; j < split; j++ {
				data[j] = f(data[j], t.GetU64(j+start))
			}
			start += split
		}
	case reflect.Uintptr:
		var f func(a, b uintptr) uintptr
		if f, ok = fn.(func(a, b uintptr) uintptr); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b uintptr) uintptr. Got %v instead", fn)
		}

		data := retVal.Uintptrs()
		for i := 0; i < size-1; i++ {
			for j := 0; j < split; j++ {
				data[j] = f(data[j], t.GetUintptr(j+start))
			}
			start += split
		}
	case reflect.Float32:
		var f func(a, b float32) float32
		if f, ok = fn.(func(a, b float32) float32); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b float32) float32. Got %v instead", fn)
		}

		data := retVal.Float32s()
		for i := 0; i < size-1; i++ {
			for j := 0; j < split; j++ {
				data[j] = f(data[j], t.GetF32(j+start))
			}
			start += split
		}
	case reflect.Float64:
		var f func(a, b float64) float64
		if f, ok = fn.(func(a, b float64) float64); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b float64) float64. Got %v instead", fn)
		}

		data := retVal.Float64s()
		for i := 0; i < size-1; i++ {
			for j := 0; j < split; j++ {
				data[j] = f(data[j], t.GetF64(j+start))
			}
			start += split
		}
	case reflect.Complex64:
		var f func(a, b complex64) complex64
		if f, ok = fn.(func(a, b complex64) complex64); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b complex64) complex64. Got %v instead", fn)
		}

		data := retVal.Complex64s()
		for i := 0; i < size-1; i++ {
			for j := 0; j < split; j++ {
				data[j] = f(data[j], t.GetC64(j+start))
			}
			start += split
		}
	case reflect.Complex128:
		var f func(a, b complex128) complex128
		if f, ok = fn.(func(a, b complex128) complex128); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b complex128) complex128. Got %v instead", fn)
		}

		data := retVal.Complex128s()
		for i := 0; i < size-1; i++ {
			for j := 0; j < split; j++ {
				data[j] = f(data[j], t.GetC128(j+start))
			}
			start += split
		}
	case reflect.String:
		var f func(a, b string) string
		if f, ok = fn.(func(a, b string) string); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b string) string. Got %v instead", fn)
		}

		data := retVal.Strings()
		for i := 0; i < size-1; i++ {
			for j := 0; j < split; j++ {
				data[j] = f(data[j], t.GetStr(j+start))
			}
			start += split
		}
	case reflect.UnsafePointer:
		var f func(a, b unsafe.Pointer) unsafe.Pointer
		if f, ok = fn.(func(a, b unsafe.Pointer) unsafe.Pointer); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b unsafe.Pointer) unsafe.Pointer. Got %v instead", fn)
		}

		data := retVal.UnsafePointers()
		for i := 0; i < size-1; i++ {
			for j := 0; j < split; j++ {
				data[j] = f(data[j], t.GetUnsafePointer(j+start))
			}
			start += split
		}
	default:
		var f reflect.Value
		var fnT reflect.Type
		if f, fnT, err = reductionFnType(fn, t.t.Type); err != nil {
			return
		}

		args := make([]reflect.Value, 0, fnT.NumIn())
		for i := 0; i < size-1; i++ {
			for j := 0; j < split; j++ {
				args = append(args, reflect.ValueOf(retVal.Get(j)))
				args = append(args, reflect.ValueOf(t.Get(start+j)))
				v := f.Call(args)[0].Interface()
				retVal.Set(j, v)
				args = args[:0]
			}
			start += split
		}
	}
	return nil
}

func (t *Dense) reduceLast(retVal *Dense, axis int, fn interface{}, defaultValue interface{}) error {
	size := t.Shape()[axis]
	var at int
	var ok bool
	switch t.t.Kind() {
	case reflect.Bool:
		var f func(a, b bool) bool
		if f, ok = fn.(func(a, b bool) bool); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b bool) bool. Got %v instead", fn)
		}
		var def bool
		if def, ok = defaultValue.(bool); !ok {
			return errors.Errorf("Expected default value to be bool. Got %v of %T instead", defaultValue, defaultValue)
		}
		for start := 0; start <= t.len()-size; start += size {
			r := reduceB(f, def, t.Bools()[start:start+size]...)
			retVal.SetB(at, r)
			at++
		}

	case reflect.Int:
		var f func(a, b int) int
		if f, ok = fn.(func(a, b int) int); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b int) int. Got %v instead", fn)
		}
		var def int
		if def, ok = defaultValue.(int); !ok {
			return errors.Errorf("Expected default value to be int. Got %v of %T instead", defaultValue, defaultValue)
		}
		for start := 0; start <= t.len()-size; start += size {
			r := reduceI(f, def, t.Ints()[start:start+size]...)
			retVal.SetI(at, r)
			at++
		}

	case reflect.Int8:
		var f func(a, b int8) int8
		if f, ok = fn.(func(a, b int8) int8); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b int8) int8. Got %v instead", fn)
		}
		var def int8
		if def, ok = defaultValue.(int8); !ok {
			return errors.Errorf("Expected default value to be int8. Got %v of %T instead", defaultValue, defaultValue)
		}
		for start := 0; start <= t.len()-size; start += size {
			r := reduceI8(f, def, t.Int8s()[start:start+size]...)
			retVal.SetI8(at, r)
			at++
		}

	case reflect.Int16:
		var f func(a, b int16) int16
		if f, ok = fn.(func(a, b int16) int16); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b int16) int16. Got %v instead", fn)
		}
		var def int16
		if def, ok = defaultValue.(int16); !ok {
			return errors.Errorf("Expected default value to be int16. Got %v of %T instead", defaultValue, defaultValue)
		}
		for start := 0; start <= t.len()-size; start += size {
			r := reduceI16(f, def, t.Int16s()[start:start+size]...)
			retVal.SetI16(at, r)
			at++
		}

	case reflect.Int32:
		var f func(a, b int32) int32
		if f, ok = fn.(func(a, b int32) int32); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b int32) int32. Got %v instead", fn)
		}
		var def int32
		if def, ok = defaultValue.(int32); !ok {
			return errors.Errorf("Expected default value to be int32. Got %v of %T instead", defaultValue, defaultValue)
		}
		for start := 0; start <= t.len()-size; start += size {
			r := reduceI32(f, def, t.Int32s()[start:start+size]...)
			retVal.SetI32(at, r)
			at++
		}

	case reflect.Int64:
		var f func(a, b int64) int64
		if f, ok = fn.(func(a, b int64) int64); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b int64) int64. Got %v instead", fn)
		}
		var def int64
		if def, ok = defaultValue.(int64); !ok {
			return errors.Errorf("Expected default value to be int64. Got %v of %T instead", defaultValue, defaultValue)
		}
		for start := 0; start <= t.len()-size; start += size {
			r := reduceI64(f, def, t.Int64s()[start:start+size]...)
			retVal.SetI64(at, r)
			at++
		}

	case reflect.Uint:
		var f func(a, b uint) uint
		if f, ok = fn.(func(a, b uint) uint); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b uint) uint. Got %v instead", fn)
		}
		var def uint
		if def, ok = defaultValue.(uint); !ok {
			return errors.Errorf("Expected default value to be uint. Got %v of %T instead", defaultValue, defaultValue)
		}
		for start := 0; start <= t.len()-size; start += size {
			r := reduceU(f, def, t.Uints()[start:start+size]...)
			retVal.SetU(at, r)
			at++
		}

	case reflect.Uint8:
		var f func(a, b uint8) uint8
		if f, ok = fn.(func(a, b uint8) uint8); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b uint8) uint8. Got %v instead", fn)
		}
		var def uint8
		if def, ok = defaultValue.(uint8); !ok {
			return errors.Errorf("Expected default value to be uint8. Got %v of %T instead", defaultValue, defaultValue)
		}
		for start := 0; start <= t.len()-size; start += size {
			r := reduceU8(f, def, t.Uint8s()[start:start+size]...)
			retVal.SetU8(at, r)
			at++
		}

	case reflect.Uint16:
		var f func(a, b uint16) uint16
		if f, ok = fn.(func(a, b uint16) uint16); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b uint16) uint16. Got %v instead", fn)
		}
		var def uint16
		if def, ok = defaultValue.(uint16); !ok {
			return errors.Errorf("Expected default value to be uint16. Got %v of %T instead", defaultValue, defaultValue)
		}
		for start := 0; start <= t.len()-size; start += size {
			r := reduceU16(f, def, t.Uint16s()[start:start+size]...)
			retVal.SetU16(at, r)
			at++
		}

	case reflect.Uint32:
		var f func(a, b uint32) uint32
		if f, ok = fn.(func(a, b uint32) uint32); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b uint32) uint32. Got %v instead", fn)
		}
		var def uint32
		if def, ok = defaultValue.(uint32); !ok {
			return errors.Errorf("Expected default value to be uint32. Got %v of %T instead", defaultValue, defaultValue)
		}
		for start := 0; start <= t.len()-size; start += size {
			r := reduceU32(f, def, t.Uint32s()[start:start+size]...)
			retVal.SetU32(at, r)
			at++
		}

	case reflect.Uint64:
		var f func(a, b uint64) uint64
		if f, ok = fn.(func(a, b uint64) uint64); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b uint64) uint64. Got %v instead", fn)
		}
		var def uint64
		if def, ok = defaultValue.(uint64); !ok {
			return errors.Errorf("Expected default value to be uint64. Got %v of %T instead", defaultValue, defaultValue)
		}
		for start := 0; start <= t.len()-size; start += size {
			r := reduceU64(f, def, t.Uint64s()[start:start+size]...)
			retVal.SetU64(at, r)
			at++
		}

	case reflect.Uintptr:
		var f func(a, b uintptr) uintptr
		if f, ok = fn.(func(a, b uintptr) uintptr); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b uintptr) uintptr. Got %v instead", fn)
		}
		var def uintptr
		if def, ok = defaultValue.(uintptr); !ok {
			return errors.Errorf("Expected default value to be uintptr. Got %v of %T instead", defaultValue, defaultValue)
		}
		for start := 0; start <= t.len()-size; start += size {
			r := reduceUintptr(f, def, t.Uintptrs()[start:start+size]...)
			retVal.SetUintptr(at, r)
			at++
		}

	case reflect.Float32:
		var f func(a, b float32) float32
		if f, ok = fn.(func(a, b float32) float32); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b float32) float32. Got %v instead", fn)
		}
		var def float32
		if def, ok = defaultValue.(float32); !ok {
			return errors.Errorf("Expected default value to be float32. Got %v of %T instead", defaultValue, defaultValue)
		}
		for start := 0; start <= t.len()-size; start += size {
			r := reduceF32(f, def, t.Float32s()[start:start+size]...)
			retVal.SetF32(at, r)
			at++
		}

	case reflect.Float64:
		var f func(a, b float64) float64
		if f, ok = fn.(func(a, b float64) float64); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b float64) float64. Got %v instead", fn)
		}
		var def float64
		if def, ok = defaultValue.(float64); !ok {
			return errors.Errorf("Expected default value to be float64. Got %v of %T instead", defaultValue, defaultValue)
		}
		for start := 0; start <= t.len()-size; start += size {
			r := reduceF64(f, def, t.Float64s()[start:start+size]...)
			retVal.SetF64(at, r)
			at++
		}

	case reflect.Complex64:
		var f func(a, b complex64) complex64
		if f, ok = fn.(func(a, b complex64) complex64); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b complex64) complex64. Got %v instead", fn)
		}
		var def complex64
		if def, ok = defaultValue.(complex64); !ok {
			return errors.Errorf("Expected default value to be complex64. Got %v of %T instead", defaultValue, defaultValue)
		}
		for start := 0; start <= t.len()-size; start += size {
			r := reduceC64(f, def, t.Complex64s()[start:start+size]...)
			retVal.SetC64(at, r)
			at++
		}

	case reflect.Complex128:
		var f func(a, b complex128) complex128
		if f, ok = fn.(func(a, b complex128) complex128); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b complex128) complex128. Got %v instead", fn)
		}
		var def complex128
		if def, ok = defaultValue.(complex128); !ok {
			return errors.Errorf("Expected default value to be complex128. Got %v of %T instead", defaultValue, defaultValue)
		}
		for start := 0; start <= t.len()-size; start += size {
			r := reduceC128(f, def, t.Complex128s()[start:start+size]...)
			retVal.SetC128(at, r)
			at++
		}

	case reflect.String:
		var f func(a, b string) string
		if f, ok = fn.(func(a, b string) string); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b string) string. Got %v instead", fn)
		}
		var def string
		if def, ok = defaultValue.(string); !ok {
			return errors.Errorf("Expected default value to be string. Got %v of %T instead", defaultValue, defaultValue)
		}
		for start := 0; start <= t.len()-size; start += size {
			r := reduceStr(f, def, t.Strings()[start:start+size]...)
			retVal.SetStr(at, r)
			at++
		}

	case reflect.UnsafePointer:
		var f func(a, b unsafe.Pointer) unsafe.Pointer
		if f, ok = fn.(func(a, b unsafe.Pointer) unsafe.Pointer); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b unsafe.Pointer) unsafe.Pointer. Got %v instead", fn)
		}
		var def unsafe.Pointer
		if def, ok = defaultValue.(unsafe.Pointer); !ok {
			return errors.Errorf("Expected default value to be unsafe.Pointer. Got %v of %T instead", defaultValue, defaultValue)
		}
		for start := 0; start <= t.len()-size; start += size {
			r := reduceUnsafePointer(f, def, t.UnsafePointers()[start:start+size]...)
			retVal.SetUnsafePointer(at, r)
			at++
		}

	default:
		f, fnT, err := reductionFnType(fn, t.t.Type)
		if err != nil {
			return err
		}
		def := reflect.ValueOf(defaultValue)
		for start := 0; start < t.len()-size; start += size {
			sliced := t.shallowClone()
			sliced.slice(start, start+size)
			r := reduceRef(f, fnT, def, sliced)
			retVal.Set(at, r)
			at++
		}
	}
	return nil
}

func (t *Dense) reduceDefault(retVal *Dense, axis int, fn interface{}) error {
	size := t.Shape()[axis]
	oStride := t.Strides()[0]
	stride := t.Strides()[axis]
	expected := retVal.Strides()[0]

	var ok bool
	switch t.t.Kind() {
	case reflect.Bool:
		var f func(a, b bool) bool
		if f, ok = fn.(func(a, b bool) bool); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b bool) bool. Got %v instead", fn)
		}
		for i := 0; i < t.Shape()[0]; i++ {
			// this loop can be parallelized!
			start := i * oStride
			tdata := t.Bools()[start : start+oStride]
			rdata := retVal.Bools()
			var innerStart, strideTrack int
			for j := 0; j < expected; j++ {
				for k := 0; k < size; k++ {
					readFrom := innerStart + k*stride
					writeTo := i*expected + j
					rdata[writeTo] = f(rdata[writeTo], tdata[readFrom])
				}
				strideTrack++
				if strideTrack >= stride {
					strideTrack = 0
					innerStart += stride
				}
				innerStart++
			}
		}
	case reflect.Int:
		var f func(a, b int) int
		if f, ok = fn.(func(a, b int) int); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b int) int. Got %v instead", fn)
		}
		for i := 0; i < t.Shape()[0]; i++ {
			// this loop can be parallelized!
			start := i * oStride
			tdata := t.Ints()[start : start+oStride]
			rdata := retVal.Ints()
			var innerStart, strideTrack int
			for j := 0; j < expected; j++ {
				for k := 0; k < size; k++ {
					readFrom := innerStart + k*stride
					writeTo := i*expected + j
					rdata[writeTo] = f(rdata[writeTo], tdata[readFrom])
				}
				strideTrack++
				if strideTrack >= stride {
					strideTrack = 0
					innerStart += stride
				}
				innerStart++
			}
		}
	case reflect.Int8:
		var f func(a, b int8) int8
		if f, ok = fn.(func(a, b int8) int8); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b int8) int8. Got %v instead", fn)
		}
		for i := 0; i < t.Shape()[0]; i++ {
			// this loop can be parallelized!
			start := i * oStride
			tdata := t.Int8s()[start : start+oStride]
			rdata := retVal.Int8s()
			var innerStart, strideTrack int
			for j := 0; j < expected; j++ {
				for k := 0; k < size; k++ {
					readFrom := innerStart + k*stride
					writeTo := i*expected + j
					rdata[writeTo] = f(rdata[writeTo], tdata[readFrom])
				}
				strideTrack++
				if strideTrack >= stride {
					strideTrack = 0
					innerStart += stride
				}
				innerStart++
			}
		}
	case reflect.Int16:
		var f func(a, b int16) int16
		if f, ok = fn.(func(a, b int16) int16); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b int16) int16. Got %v instead", fn)
		}
		for i := 0; i < t.Shape()[0]; i++ {
			// this loop can be parallelized!
			start := i * oStride
			tdata := t.Int16s()[start : start+oStride]
			rdata := retVal.Int16s()
			var innerStart, strideTrack int
			for j := 0; j < expected; j++ {
				for k := 0; k < size; k++ {
					readFrom := innerStart + k*stride
					writeTo := i*expected + j
					rdata[writeTo] = f(rdata[writeTo], tdata[readFrom])
				}
				strideTrack++
				if strideTrack >= stride {
					strideTrack = 0
					innerStart += stride
				}
				innerStart++
			}
		}
	case reflect.Int32:
		var f func(a, b int32) int32
		if f, ok = fn.(func(a, b int32) int32); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b int32) int32. Got %v instead", fn)
		}
		for i := 0; i < t.Shape()[0]; i++ {
			// this loop can be parallelized!
			start := i * oStride
			tdata := t.Int32s()[start : start+oStride]
			rdata := retVal.Int32s()
			var innerStart, strideTrack int
			for j := 0; j < expected; j++ {
				for k := 0; k < size; k++ {
					readFrom := innerStart + k*stride
					writeTo := i*expected + j
					rdata[writeTo] = f(rdata[writeTo], tdata[readFrom])
				}
				strideTrack++
				if strideTrack >= stride {
					strideTrack = 0
					innerStart += stride
				}
				innerStart++
			}
		}
	case reflect.Int64:
		var f func(a, b int64) int64
		if f, ok = fn.(func(a, b int64) int64); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b int64) int64. Got %v instead", fn)
		}
		for i := 0; i < t.Shape()[0]; i++ {
			// this loop can be parallelized!
			start := i * oStride
			tdata := t.Int64s()[start : start+oStride]
			rdata := retVal.Int64s()
			var innerStart, strideTrack int
			for j := 0; j < expected; j++ {
				for k := 0; k < size; k++ {
					readFrom := innerStart + k*stride
					writeTo := i*expected + j
					rdata[writeTo] = f(rdata[writeTo], tdata[readFrom])
				}
				strideTrack++
				if strideTrack >= stride {
					strideTrack = 0
					innerStart += stride
				}
				innerStart++
			}
		}
	case reflect.Uint:
		var f func(a, b uint) uint
		if f, ok = fn.(func(a, b uint) uint); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b uint) uint. Got %v instead", fn)
		}
		for i := 0; i < t.Shape()[0]; i++ {
			// this loop can be parallelized!
			start := i * oStride
			tdata := t.Uints()[start : start+oStride]
			rdata := retVal.Uints()
			var innerStart, strideTrack int
			for j := 0; j < expected; j++ {
				for k := 0; k < size; k++ {
					readFrom := innerStart + k*stride
					writeTo := i*expected + j
					rdata[writeTo] = f(rdata[writeTo], tdata[readFrom])
				}
				strideTrack++
				if strideTrack >= stride {
					strideTrack = 0
					innerStart += stride
				}
				innerStart++
			}
		}
	case reflect.Uint8:
		var f func(a, b uint8) uint8
		if f, ok = fn.(func(a, b uint8) uint8); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b uint8) uint8. Got %v instead", fn)
		}
		for i := 0; i < t.Shape()[0]; i++ {
			// this loop can be parallelized!
			start := i * oStride
			tdata := t.Uint8s()[start : start+oStride]
			rdata := retVal.Uint8s()
			var innerStart, strideTrack int
			for j := 0; j < expected; j++ {
				for k := 0; k < size; k++ {
					readFrom := innerStart + k*stride
					writeTo := i*expected + j
					rdata[writeTo] = f(rdata[writeTo], tdata[readFrom])
				}
				strideTrack++
				if strideTrack >= stride {
					strideTrack = 0
					innerStart += stride
				}
				innerStart++
			}
		}
	case reflect.Uint16:
		var f func(a, b uint16) uint16
		if f, ok = fn.(func(a, b uint16) uint16); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b uint16) uint16. Got %v instead", fn)
		}
		for i := 0; i < t.Shape()[0]; i++ {
			// this loop can be parallelized!
			start := i * oStride
			tdata := t.Uint16s()[start : start+oStride]
			rdata := retVal.Uint16s()
			var innerStart, strideTrack int
			for j := 0; j < expected; j++ {
				for k := 0; k < size; k++ {
					readFrom := innerStart + k*stride
					writeTo := i*expected + j
					rdata[writeTo] = f(rdata[writeTo], tdata[readFrom])
				}
				strideTrack++
				if strideTrack >= stride {
					strideTrack = 0
					innerStart += stride
				}
				innerStart++
			}
		}
	case reflect.Uint32:
		var f func(a, b uint32) uint32
		if f, ok = fn.(func(a, b uint32) uint32); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b uint32) uint32. Got %v instead", fn)
		}
		for i := 0; i < t.Shape()[0]; i++ {
			// this loop can be parallelized!
			start := i * oStride
			tdata := t.Uint32s()[start : start+oStride]
			rdata := retVal.Uint32s()
			var innerStart, strideTrack int
			for j := 0; j < expected; j++ {
				for k := 0; k < size; k++ {
					readFrom := innerStart + k*stride
					writeTo := i*expected + j
					rdata[writeTo] = f(rdata[writeTo], tdata[readFrom])
				}
				strideTrack++
				if strideTrack >= stride {
					strideTrack = 0
					innerStart += stride
				}
				innerStart++
			}
		}
	case reflect.Uint64:
		var f func(a, b uint64) uint64
		if f, ok = fn.(func(a, b uint64) uint64); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b uint64) uint64. Got %v instead", fn)
		}
		for i := 0; i < t.Shape()[0]; i++ {
			// this loop can be parallelized!
			start := i * oStride
			tdata := t.Uint64s()[start : start+oStride]
			rdata := retVal.Uint64s()
			var innerStart, strideTrack int
			for j := 0; j < expected; j++ {
				for k := 0; k < size; k++ {
					readFrom := innerStart + k*stride
					writeTo := i*expected + j
					rdata[writeTo] = f(rdata[writeTo], tdata[readFrom])
				}
				strideTrack++
				if strideTrack >= stride {
					strideTrack = 0
					innerStart += stride
				}
				innerStart++
			}
		}
	case reflect.Uintptr:
		var f func(a, b uintptr) uintptr
		if f, ok = fn.(func(a, b uintptr) uintptr); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b uintptr) uintptr. Got %v instead", fn)
		}
		for i := 0; i < t.Shape()[0]; i++ {
			// this loop can be parallelized!
			start := i * oStride
			tdata := t.Uintptrs()[start : start+oStride]
			rdata := retVal.Uintptrs()
			var innerStart, strideTrack int
			for j := 0; j < expected; j++ {
				for k := 0; k < size; k++ {
					readFrom := innerStart + k*stride
					writeTo := i*expected + j
					rdata[writeTo] = f(rdata[writeTo], tdata[readFrom])
				}
				strideTrack++
				if strideTrack >= stride {
					strideTrack = 0
					innerStart += stride
				}
				innerStart++
			}
		}
	case reflect.Float32:
		var f func(a, b float32) float32
		if f, ok = fn.(func(a, b float32) float32); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b float32) float32. Got %v instead", fn)
		}
		for i := 0; i < t.Shape()[0]; i++ {
			// this loop can be parallelized!
			start := i * oStride
			tdata := t.Float32s()[start : start+oStride]
			rdata := retVal.Float32s()
			var innerStart, strideTrack int
			for j := 0; j < expected; j++ {
				for k := 0; k < size; k++ {
					readFrom := innerStart + k*stride
					writeTo := i*expected + j
					rdata[writeTo] = f(rdata[writeTo], tdata[readFrom])
				}
				strideTrack++
				if strideTrack >= stride {
					strideTrack = 0
					innerStart += stride
				}
				innerStart++
			}
		}
	case reflect.Float64:
		var f func(a, b float64) float64
		if f, ok = fn.(func(a, b float64) float64); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b float64) float64. Got %v instead", fn)
		}
		for i := 0; i < t.Shape()[0]; i++ {
			// this loop can be parallelized!
			start := i * oStride
			tdata := t.Float64s()[start : start+oStride]
			rdata := retVal.Float64s()
			var innerStart, strideTrack int
			for j := 0; j < expected; j++ {
				for k := 0; k < size; k++ {
					readFrom := innerStart + k*stride
					writeTo := i*expected + j
					rdata[writeTo] = f(rdata[writeTo], tdata[readFrom])
				}
				strideTrack++
				if strideTrack >= stride {
					strideTrack = 0
					innerStart += stride
				}
				innerStart++
			}
		}
	case reflect.Complex64:
		var f func(a, b complex64) complex64
		if f, ok = fn.(func(a, b complex64) complex64); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b complex64) complex64. Got %v instead", fn)
		}
		for i := 0; i < t.Shape()[0]; i++ {
			// this loop can be parallelized!
			start := i * oStride
			tdata := t.Complex64s()[start : start+oStride]
			rdata := retVal.Complex64s()
			var innerStart, strideTrack int
			for j := 0; j < expected; j++ {
				for k := 0; k < size; k++ {
					readFrom := innerStart + k*stride
					writeTo := i*expected + j
					rdata[writeTo] = f(rdata[writeTo], tdata[readFrom])
				}
				strideTrack++
				if strideTrack >= stride {
					strideTrack = 0
					innerStart += stride
				}
				innerStart++
			}
		}
	case reflect.Complex128:
		var f func(a, b complex128) complex128
		if f, ok = fn.(func(a, b complex128) complex128); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b complex128) complex128. Got %v instead", fn)
		}
		for i := 0; i < t.Shape()[0]; i++ {
			// this loop can be parallelized!
			start := i * oStride
			tdata := t.Complex128s()[start : start+oStride]
			rdata := retVal.Complex128s()
			var innerStart, strideTrack int
			for j := 0; j < expected; j++ {
				for k := 0; k < size; k++ {
					readFrom := innerStart + k*stride
					writeTo := i*expected + j
					rdata[writeTo] = f(rdata[writeTo], tdata[readFrom])
				}
				strideTrack++
				if strideTrack >= stride {
					strideTrack = 0
					innerStart += stride
				}
				innerStart++
			}
		}
	case reflect.String:
		var f func(a, b string) string
		if f, ok = fn.(func(a, b string) string); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b string) string. Got %v instead", fn)
		}
		for i := 0; i < t.Shape()[0]; i++ {
			// this loop can be parallelized!
			start := i * oStride
			tdata := t.Strings()[start : start+oStride]
			rdata := retVal.Strings()
			var innerStart, strideTrack int
			for j := 0; j < expected; j++ {
				for k := 0; k < size; k++ {
					readFrom := innerStart + k*stride
					writeTo := i*expected + j
					rdata[writeTo] = f(rdata[writeTo], tdata[readFrom])
				}
				strideTrack++
				if strideTrack >= stride {
					strideTrack = 0
					innerStart += stride
				}
				innerStart++
			}
		}
	case reflect.UnsafePointer:
		var f func(a, b unsafe.Pointer) unsafe.Pointer
		if f, ok = fn.(func(a, b unsafe.Pointer) unsafe.Pointer); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b unsafe.Pointer) unsafe.Pointer. Got %v instead", fn)
		}
		for i := 0; i < t.Shape()[0]; i++ {
			// this loop can be parallelized!
			start := i * oStride
			tdata := t.UnsafePointers()[start : start+oStride]
			rdata := retVal.UnsafePointers()
			var innerStart, strideTrack int
			for j := 0; j < expected; j++ {
				for k := 0; k < size; k++ {
					readFrom := innerStart + k*stride
					writeTo := i*expected + j
					rdata[writeTo] = f(rdata[writeTo], tdata[readFrom])
				}
				strideTrack++
				if strideTrack >= stride {
					strideTrack = 0
					innerStart += stride
				}
				innerStart++
			}
		}
	default:
		f, fnT, err := reductionFnType(fn, t.t.Type)
		if err != nil {
			return err
		}
		args := make([]reflect.Value, 0, fnT.NumIn())
		for i := 0; i < t.Shape()[0]; i++ {
			// this loop can be parallelized!
			start := i * oStride
			sliced := t.shallowClone()
			sliced.slice(start, start+oStride)

			var innerStart, strideTrack int
			for j := 0; j < expected; j++ {
				for k := 0; k < size; k++ {
					readFrom := innerStart + k*stride
					writeTo := i*expected + j
					args = append(args, reflect.ValueOf(retVal.Get(writeTo)))
					args = append(args, reflect.ValueOf(sliced.Get(readFrom)))
					v := f.Call(args)[0].Interface()
					retVal.Set(writeTo, v)
				}
				strideTrack++
				if strideTrack >= stride {
					strideTrack = 0
					innerStart += stride
				}
				innerStart++
			}
		}
	}
	return nil
}

// sReduceI is a specialization for int reductions, used in methods such as Sum, Prod, Max etc
func (t *Dense) sReduceI(axis int, zeroFn func(a, b []int) error, lastFn func([]int) int, defFn func(a, b int) int) (retVal *Dense) {
	if t.IsScalar() {
		return t
	}

	var newShape Shape
	for i, s := range t.Shape() {
		if i == axis {
			continue
		}
		newShape = append(newShape, s)
	}
	retVal = New(Of(t.t), WithShape(newShape...))
	size := t.Shape()[axis]
	lastAxis := t.Dims() - 1
	switch axis {
	case 0:
		// most efficient
		split := t.len() / size
		copy(retVal.Ints()[0:split], t.Ints()[0:split])

		start := split
		for i := 0; i < size-1; i++ {
			if err := zeroFn(retVal.Ints(), t.Ints()[start:start+split]); err != nil {
				panic(err)
			}
			start += split
		}
	case lastAxis:
		// second most efficient
		var at int
		for start := 0; start <= t.len()-size; start += size {
			retVal.SetI(at, lastFn(t.Ints()[start:start+size]))
			at++
		}
	default:
		outerSize := t.Shape()[0]
		outerStride := t.Strides()[0]
		stride := t.Strides()[axis]
		expected := retVal.Strides()[0]

		for i := 0; i < outerSize; i++ {
			start := i * outerStride
			tdata := t.Ints()[start : start+outerStride]
			rdata := retVal.Ints()
			var innerStart, strideTrack int
			for j := 0; j < expected; j++ {
				for k := 0; k < size; k++ {
					readFrom := innerStart + k*stride
					writeTo := i*expected + j
					a := rdata[writeTo]
					b := tdata[readFrom]
					if k == 0 {
						rdata[writeTo] = b
					} else {
						rdata[writeTo] = defFn(a, b)
					}
				}
				strideTrack++
				if strideTrack >= stride {
					strideTrack = 0
					innerStart += stride
				}
				innerStart++
			}
		}
	}
	return retVal
}

// sReduceI8 is a specialization for int8 reductions, used in methods such as Sum, Prod, Max etc
func (t *Dense) sReduceI8(axis int, zeroFn func(a, b []int8) error, lastFn func([]int8) int8, defFn func(a, b int8) int8) (retVal *Dense) {
	if t.IsScalar() {
		return t
	}

	var newShape Shape
	for i, s := range t.Shape() {
		if i == axis {
			continue
		}
		newShape = append(newShape, s)
	}
	retVal = New(Of(t.t), WithShape(newShape...))
	size := t.Shape()[axis]
	lastAxis := t.Dims() - 1
	switch axis {
	case 0:
		// most efficient
		split := t.len() / size
		copy(retVal.Int8s()[0:split], t.Int8s()[0:split])

		start := split
		for i := 0; i < size-1; i++ {
			if err := zeroFn(retVal.Int8s(), t.Int8s()[start:start+split]); err != nil {
				panic(err)
			}
			start += split
		}
	case lastAxis:
		// second most efficient
		var at int
		for start := 0; start <= t.len()-size; start += size {
			retVal.SetI8(at, lastFn(t.Int8s()[start:start+size]))
			at++
		}
	default:
		outerSize := t.Shape()[0]
		outerStride := t.Strides()[0]
		stride := t.Strides()[axis]
		expected := retVal.Strides()[0]

		for i := 0; i < outerSize; i++ {
			start := i * outerStride
			tdata := t.Int8s()[start : start+outerStride]
			rdata := retVal.Int8s()
			var innerStart, strideTrack int
			for j := 0; j < expected; j++ {
				for k := 0; k < size; k++ {
					readFrom := innerStart + k*stride
					writeTo := i*expected + j
					a := rdata[writeTo]
					b := tdata[readFrom]
					if k == 0 {
						rdata[writeTo] = b
					} else {
						rdata[writeTo] = defFn(a, b)
					}
				}
				strideTrack++
				if strideTrack >= stride {
					strideTrack = 0
					innerStart += stride
				}
				innerStart++
			}
		}
	}
	return retVal
}

// sReduceI16 is a specialization for int16 reductions, used in methods such as Sum, Prod, Max etc
func (t *Dense) sReduceI16(axis int, zeroFn func(a, b []int16) error, lastFn func([]int16) int16, defFn func(a, b int16) int16) (retVal *Dense) {
	if t.IsScalar() {
		return t
	}

	var newShape Shape
	for i, s := range t.Shape() {
		if i == axis {
			continue
		}
		newShape = append(newShape, s)
	}
	retVal = New(Of(t.t), WithShape(newShape...))
	size := t.Shape()[axis]
	lastAxis := t.Dims() - 1
	switch axis {
	case 0:
		// most efficient
		split := t.len() / size
		copy(retVal.Int16s()[0:split], t.Int16s()[0:split])

		start := split
		for i := 0; i < size-1; i++ {
			if err := zeroFn(retVal.Int16s(), t.Int16s()[start:start+split]); err != nil {
				panic(err)
			}
			start += split
		}
	case lastAxis:
		// second most efficient
		var at int
		for start := 0; start <= t.len()-size; start += size {
			retVal.SetI16(at, lastFn(t.Int16s()[start:start+size]))
			at++
		}
	default:
		outerSize := t.Shape()[0]
		outerStride := t.Strides()[0]
		stride := t.Strides()[axis]
		expected := retVal.Strides()[0]

		for i := 0; i < outerSize; i++ {
			start := i * outerStride
			tdata := t.Int16s()[start : start+outerStride]
			rdata := retVal.Int16s()
			var innerStart, strideTrack int
			for j := 0; j < expected; j++ {
				for k := 0; k < size; k++ {
					readFrom := innerStart + k*stride
					writeTo := i*expected + j
					a := rdata[writeTo]
					b := tdata[readFrom]
					if k == 0 {
						rdata[writeTo] = b
					} else {
						rdata[writeTo] = defFn(a, b)
					}
				}
				strideTrack++
				if strideTrack >= stride {
					strideTrack = 0
					innerStart += stride
				}
				innerStart++
			}
		}
	}
	return retVal
}

// sReduceI32 is a specialization for int32 reductions, used in methods such as Sum, Prod, Max etc
func (t *Dense) sReduceI32(axis int, zeroFn func(a, b []int32) error, lastFn func([]int32) int32, defFn func(a, b int32) int32) (retVal *Dense) {
	if t.IsScalar() {
		return t
	}

	var newShape Shape
	for i, s := range t.Shape() {
		if i == axis {
			continue
		}
		newShape = append(newShape, s)
	}
	retVal = New(Of(t.t), WithShape(newShape...))
	size := t.Shape()[axis]
	lastAxis := t.Dims() - 1
	switch axis {
	case 0:
		// most efficient
		split := t.len() / size
		copy(retVal.Int32s()[0:split], t.Int32s()[0:split])

		start := split
		for i := 0; i < size-1; i++ {
			if err := zeroFn(retVal.Int32s(), t.Int32s()[start:start+split]); err != nil {
				panic(err)
			}
			start += split
		}
	case lastAxis:
		// second most efficient
		var at int
		for start := 0; start <= t.len()-size; start += size {
			retVal.SetI32(at, lastFn(t.Int32s()[start:start+size]))
			at++
		}
	default:
		outerSize := t.Shape()[0]
		outerStride := t.Strides()[0]
		stride := t.Strides()[axis]
		expected := retVal.Strides()[0]

		for i := 0; i < outerSize; i++ {
			start := i * outerStride
			tdata := t.Int32s()[start : start+outerStride]
			rdata := retVal.Int32s()
			var innerStart, strideTrack int
			for j := 0; j < expected; j++ {
				for k := 0; k < size; k++ {
					readFrom := innerStart + k*stride
					writeTo := i*expected + j
					a := rdata[writeTo]
					b := tdata[readFrom]
					if k == 0 {
						rdata[writeTo] = b
					} else {
						rdata[writeTo] = defFn(a, b)
					}
				}
				strideTrack++
				if strideTrack >= stride {
					strideTrack = 0
					innerStart += stride
				}
				innerStart++
			}
		}
	}
	return retVal
}

// sReduceI64 is a specialization for int64 reductions, used in methods such as Sum, Prod, Max etc
func (t *Dense) sReduceI64(axis int, zeroFn func(a, b []int64) error, lastFn func([]int64) int64, defFn func(a, b int64) int64) (retVal *Dense) {
	if t.IsScalar() {
		return t
	}

	var newShape Shape
	for i, s := range t.Shape() {
		if i == axis {
			continue
		}
		newShape = append(newShape, s)
	}
	retVal = New(Of(t.t), WithShape(newShape...))
	size := t.Shape()[axis]
	lastAxis := t.Dims() - 1
	switch axis {
	case 0:
		// most efficient
		split := t.len() / size
		copy(retVal.Int64s()[0:split], t.Int64s()[0:split])

		start := split
		for i := 0; i < size-1; i++ {
			if err := zeroFn(retVal.Int64s(), t.Int64s()[start:start+split]); err != nil {
				panic(err)
			}
			start += split
		}
	case lastAxis:
		// second most efficient
		var at int
		for start := 0; start <= t.len()-size; start += size {
			retVal.SetI64(at, lastFn(t.Int64s()[start:start+size]))
			at++
		}
	default:
		outerSize := t.Shape()[0]
		outerStride := t.Strides()[0]
		stride := t.Strides()[axis]
		expected := retVal.Strides()[0]

		for i := 0; i < outerSize; i++ {
			start := i * outerStride
			tdata := t.Int64s()[start : start+outerStride]
			rdata := retVal.Int64s()
			var innerStart, strideTrack int
			for j := 0; j < expected; j++ {
				for k := 0; k < size; k++ {
					readFrom := innerStart + k*stride
					writeTo := i*expected + j
					a := rdata[writeTo]
					b := tdata[readFrom]
					if k == 0 {
						rdata[writeTo] = b
					} else {
						rdata[writeTo] = defFn(a, b)
					}
				}
				strideTrack++
				if strideTrack >= stride {
					strideTrack = 0
					innerStart += stride
				}
				innerStart++
			}
		}
	}
	return retVal
}

// sReduceU is a specialization for uint reductions, used in methods such as Sum, Prod, Max etc
func (t *Dense) sReduceU(axis int, zeroFn func(a, b []uint) error, lastFn func([]uint) uint, defFn func(a, b uint) uint) (retVal *Dense) {
	if t.IsScalar() {
		return t
	}

	var newShape Shape
	for i, s := range t.Shape() {
		if i == axis {
			continue
		}
		newShape = append(newShape, s)
	}
	retVal = New(Of(t.t), WithShape(newShape...))
	size := t.Shape()[axis]
	lastAxis := t.Dims() - 1
	switch axis {
	case 0:
		// most efficient
		split := t.len() / size
		copy(retVal.Uints()[0:split], t.Uints()[0:split])

		start := split
		for i := 0; i < size-1; i++ {
			if err := zeroFn(retVal.Uints(), t.Uints()[start:start+split]); err != nil {
				panic(err)
			}
			start += split
		}
	case lastAxis:
		// second most efficient
		var at int
		for start := 0; start <= t.len()-size; start += size {
			retVal.SetU(at, lastFn(t.Uints()[start:start+size]))
			at++
		}
	default:
		outerSize := t.Shape()[0]
		outerStride := t.Strides()[0]
		stride := t.Strides()[axis]
		expected := retVal.Strides()[0]

		for i := 0; i < outerSize; i++ {
			start := i * outerStride
			tdata := t.Uints()[start : start+outerStride]
			rdata := retVal.Uints()
			var innerStart, strideTrack int
			for j := 0; j < expected; j++ {
				for k := 0; k < size; k++ {
					readFrom := innerStart + k*stride
					writeTo := i*expected + j
					a := rdata[writeTo]
					b := tdata[readFrom]
					if k == 0 {
						rdata[writeTo] = b
					} else {
						rdata[writeTo] = defFn(a, b)
					}
				}
				strideTrack++
				if strideTrack >= stride {
					strideTrack = 0
					innerStart += stride
				}
				innerStart++
			}
		}
	}
	return retVal
}

// sReduceU8 is a specialization for uint8 reductions, used in methods such as Sum, Prod, Max etc
func (t *Dense) sReduceU8(axis int, zeroFn func(a, b []uint8) error, lastFn func([]uint8) uint8, defFn func(a, b uint8) uint8) (retVal *Dense) {
	if t.IsScalar() {
		return t
	}

	var newShape Shape
	for i, s := range t.Shape() {
		if i == axis {
			continue
		}
		newShape = append(newShape, s)
	}
	retVal = New(Of(t.t), WithShape(newShape...))
	size := t.Shape()[axis]
	lastAxis := t.Dims() - 1
	switch axis {
	case 0:
		// most efficient
		split := t.len() / size
		copy(retVal.Uint8s()[0:split], t.Uint8s()[0:split])

		start := split
		for i := 0; i < size-1; i++ {
			if err := zeroFn(retVal.Uint8s(), t.Uint8s()[start:start+split]); err != nil {
				panic(err)
			}
			start += split
		}
	case lastAxis:
		// second most efficient
		var at int
		for start := 0; start <= t.len()-size; start += size {
			retVal.SetU8(at, lastFn(t.Uint8s()[start:start+size]))
			at++
		}
	default:
		outerSize := t.Shape()[0]
		outerStride := t.Strides()[0]
		stride := t.Strides()[axis]
		expected := retVal.Strides()[0]

		for i := 0; i < outerSize; i++ {
			start := i * outerStride
			tdata := t.Uint8s()[start : start+outerStride]
			rdata := retVal.Uint8s()
			var innerStart, strideTrack int
			for j := 0; j < expected; j++ {
				for k := 0; k < size; k++ {
					readFrom := innerStart + k*stride
					writeTo := i*expected + j
					a := rdata[writeTo]
					b := tdata[readFrom]
					if k == 0 {
						rdata[writeTo] = b
					} else {
						rdata[writeTo] = defFn(a, b)
					}
				}
				strideTrack++
				if strideTrack >= stride {
					strideTrack = 0
					innerStart += stride
				}
				innerStart++
			}
		}
	}
	return retVal
}

// sReduceU16 is a specialization for uint16 reductions, used in methods such as Sum, Prod, Max etc
func (t *Dense) sReduceU16(axis int, zeroFn func(a, b []uint16) error, lastFn func([]uint16) uint16, defFn func(a, b uint16) uint16) (retVal *Dense) {
	if t.IsScalar() {
		return t
	}

	var newShape Shape
	for i, s := range t.Shape() {
		if i == axis {
			continue
		}
		newShape = append(newShape, s)
	}
	retVal = New(Of(t.t), WithShape(newShape...))
	size := t.Shape()[axis]
	lastAxis := t.Dims() - 1
	switch axis {
	case 0:
		// most efficient
		split := t.len() / size
		copy(retVal.Uint16s()[0:split], t.Uint16s()[0:split])

		start := split
		for i := 0; i < size-1; i++ {
			if err := zeroFn(retVal.Uint16s(), t.Uint16s()[start:start+split]); err != nil {
				panic(err)
			}
			start += split
		}
	case lastAxis:
		// second most efficient
		var at int
		for start := 0; start <= t.len()-size; start += size {
			retVal.SetU16(at, lastFn(t.Uint16s()[start:start+size]))
			at++
		}
	default:
		outerSize := t.Shape()[0]
		outerStride := t.Strides()[0]
		stride := t.Strides()[axis]
		expected := retVal.Strides()[0]

		for i := 0; i < outerSize; i++ {
			start := i * outerStride
			tdata := t.Uint16s()[start : start+outerStride]
			rdata := retVal.Uint16s()
			var innerStart, strideTrack int
			for j := 0; j < expected; j++ {
				for k := 0; k < size; k++ {
					readFrom := innerStart + k*stride
					writeTo := i*expected + j
					a := rdata[writeTo]
					b := tdata[readFrom]
					if k == 0 {
						rdata[writeTo] = b
					} else {
						rdata[writeTo] = defFn(a, b)
					}
				}
				strideTrack++
				if strideTrack >= stride {
					strideTrack = 0
					innerStart += stride
				}
				innerStart++
			}
		}
	}
	return retVal
}

// sReduceU32 is a specialization for uint32 reductions, used in methods such as Sum, Prod, Max etc
func (t *Dense) sReduceU32(axis int, zeroFn func(a, b []uint32) error, lastFn func([]uint32) uint32, defFn func(a, b uint32) uint32) (retVal *Dense) {
	if t.IsScalar() {
		return t
	}

	var newShape Shape
	for i, s := range t.Shape() {
		if i == axis {
			continue
		}
		newShape = append(newShape, s)
	}
	retVal = New(Of(t.t), WithShape(newShape...))
	size := t.Shape()[axis]
	lastAxis := t.Dims() - 1
	switch axis {
	case 0:
		// most efficient
		split := t.len() / size
		copy(retVal.Uint32s()[0:split], t.Uint32s()[0:split])

		start := split
		for i := 0; i < size-1; i++ {
			if err := zeroFn(retVal.Uint32s(), t.Uint32s()[start:start+split]); err != nil {
				panic(err)
			}
			start += split
		}
	case lastAxis:
		// second most efficient
		var at int
		for start := 0; start <= t.len()-size; start += size {
			retVal.SetU32(at, lastFn(t.Uint32s()[start:start+size]))
			at++
		}
	default:
		outerSize := t.Shape()[0]
		outerStride := t.Strides()[0]
		stride := t.Strides()[axis]
		expected := retVal.Strides()[0]

		for i := 0; i < outerSize; i++ {
			start := i * outerStride
			tdata := t.Uint32s()[start : start+outerStride]
			rdata := retVal.Uint32s()
			var innerStart, strideTrack int
			for j := 0; j < expected; j++ {
				for k := 0; k < size; k++ {
					readFrom := innerStart + k*stride
					writeTo := i*expected + j
					a := rdata[writeTo]
					b := tdata[readFrom]
					if k == 0 {
						rdata[writeTo] = b
					} else {
						rdata[writeTo] = defFn(a, b)
					}
				}
				strideTrack++
				if strideTrack >= stride {
					strideTrack = 0
					innerStart += stride
				}
				innerStart++
			}
		}
	}
	return retVal
}

// sReduceU64 is a specialization for uint64 reductions, used in methods such as Sum, Prod, Max etc
func (t *Dense) sReduceU64(axis int, zeroFn func(a, b []uint64) error, lastFn func([]uint64) uint64, defFn func(a, b uint64) uint64) (retVal *Dense) {
	if t.IsScalar() {
		return t
	}

	var newShape Shape
	for i, s := range t.Shape() {
		if i == axis {
			continue
		}
		newShape = append(newShape, s)
	}
	retVal = New(Of(t.t), WithShape(newShape...))
	size := t.Shape()[axis]
	lastAxis := t.Dims() - 1
	switch axis {
	case 0:
		// most efficient
		split := t.len() / size
		copy(retVal.Uint64s()[0:split], t.Uint64s()[0:split])

		start := split
		for i := 0; i < size-1; i++ {
			if err := zeroFn(retVal.Uint64s(), t.Uint64s()[start:start+split]); err != nil {
				panic(err)
			}
			start += split
		}
	case lastAxis:
		// second most efficient
		var at int
		for start := 0; start <= t.len()-size; start += size {
			retVal.SetU64(at, lastFn(t.Uint64s()[start:start+size]))
			at++
		}
	default:
		outerSize := t.Shape()[0]
		outerStride := t.Strides()[0]
		stride := t.Strides()[axis]
		expected := retVal.Strides()[0]

		for i := 0; i < outerSize; i++ {
			start := i * outerStride
			tdata := t.Uint64s()[start : start+outerStride]
			rdata := retVal.Uint64s()
			var innerStart, strideTrack int
			for j := 0; j < expected; j++ {
				for k := 0; k < size; k++ {
					readFrom := innerStart + k*stride
					writeTo := i*expected + j
					a := rdata[writeTo]
					b := tdata[readFrom]
					if k == 0 {
						rdata[writeTo] = b
					} else {
						rdata[writeTo] = defFn(a, b)
					}
				}
				strideTrack++
				if strideTrack >= stride {
					strideTrack = 0
					innerStart += stride
				}
				innerStart++
			}
		}
	}
	return retVal
}

// sReduceF32 is a specialization for float32 reductions, used in methods such as Sum, Prod, Max etc
func (t *Dense) sReduceF32(axis int, zeroFn func(a, b []float32) error, lastFn func([]float32) float32, defFn func(a, b float32) float32) (retVal *Dense) {
	if t.IsScalar() {
		return t
	}

	var newShape Shape
	for i, s := range t.Shape() {
		if i == axis {
			continue
		}
		newShape = append(newShape, s)
	}
	retVal = New(Of(t.t), WithShape(newShape...))
	size := t.Shape()[axis]
	lastAxis := t.Dims() - 1
	switch axis {
	case 0:
		// most efficient
		split := t.len() / size
		copy(retVal.Float32s()[0:split], t.Float32s()[0:split])

		start := split
		for i := 0; i < size-1; i++ {
			if err := zeroFn(retVal.Float32s(), t.Float32s()[start:start+split]); err != nil {
				panic(err)
			}
			start += split
		}
	case lastAxis:
		// second most efficient
		var at int
		for start := 0; start <= t.len()-size; start += size {
			retVal.SetF32(at, lastFn(t.Float32s()[start:start+size]))
			at++
		}
	default:
		outerSize := t.Shape()[0]
		outerStride := t.Strides()[0]
		stride := t.Strides()[axis]
		expected := retVal.Strides()[0]

		for i := 0; i < outerSize; i++ {
			start := i * outerStride
			tdata := t.Float32s()[start : start+outerStride]
			rdata := retVal.Float32s()
			var innerStart, strideTrack int
			for j := 0; j < expected; j++ {
				for k := 0; k < size; k++ {
					readFrom := innerStart + k*stride
					writeTo := i*expected + j
					a := rdata[writeTo]
					b := tdata[readFrom]
					if k == 0 {
						rdata[writeTo] = b
					} else {
						rdata[writeTo] = defFn(a, b)
					}
				}
				strideTrack++
				if strideTrack >= stride {
					strideTrack = 0
					innerStart += stride
				}
				innerStart++
			}
		}
	}
	return retVal
}

// sReduceF64 is a specialization for float64 reductions, used in methods such as Sum, Prod, Max etc
func (t *Dense) sReduceF64(axis int, zeroFn func(a, b []float64) error, lastFn func([]float64) float64, defFn func(a, b float64) float64) (retVal *Dense) {
	if t.IsScalar() {
		return t
	}

	var newShape Shape
	for i, s := range t.Shape() {
		if i == axis {
			continue
		}
		newShape = append(newShape, s)
	}
	retVal = New(Of(t.t), WithShape(newShape...))
	size := t.Shape()[axis]
	lastAxis := t.Dims() - 1
	switch axis {
	case 0:
		// most efficient
		split := t.len() / size
		copy(retVal.Float64s()[0:split], t.Float64s()[0:split])

		start := split
		for i := 0; i < size-1; i++ {
			if err := zeroFn(retVal.Float64s(), t.Float64s()[start:start+split]); err != nil {
				panic(err)
			}
			start += split
		}
	case lastAxis:
		// second most efficient
		var at int
		for start := 0; start <= t.len()-size; start += size {
			retVal.SetF64(at, lastFn(t.Float64s()[start:start+size]))
			at++
		}
	default:
		outerSize := t.Shape()[0]
		outerStride := t.Strides()[0]
		stride := t.Strides()[axis]
		expected := retVal.Strides()[0]

		for i := 0; i < outerSize; i++ {
			start := i * outerStride
			tdata := t.Float64s()[start : start+outerStride]
			rdata := retVal.Float64s()
			var innerStart, strideTrack int
			for j := 0; j < expected; j++ {
				for k := 0; k < size; k++ {
					readFrom := innerStart + k*stride
					writeTo := i*expected + j
					a := rdata[writeTo]
					b := tdata[readFrom]
					if k == 0 {
						rdata[writeTo] = b
					} else {
						rdata[writeTo] = defFn(a, b)
					}
				}
				strideTrack++
				if strideTrack >= stride {
					strideTrack = 0
					innerStart += stride
				}
				innerStart++
			}
		}
	}
	return retVal
}

// sReduceC64 is a specialization for complex64 reductions, used in methods such as Sum, Prod, Max etc
func (t *Dense) sReduceC64(axis int, zeroFn func(a, b []complex64) error, lastFn func([]complex64) complex64, defFn func(a, b complex64) complex64) (retVal *Dense) {
	if t.IsScalar() {
		return t
	}

	var newShape Shape
	for i, s := range t.Shape() {
		if i == axis {
			continue
		}
		newShape = append(newShape, s)
	}
	retVal = New(Of(t.t), WithShape(newShape...))
	size := t.Shape()[axis]
	lastAxis := t.Dims() - 1
	switch axis {
	case 0:
		// most efficient
		split := t.len() / size
		copy(retVal.Complex64s()[0:split], t.Complex64s()[0:split])

		start := split
		for i := 0; i < size-1; i++ {
			if err := zeroFn(retVal.Complex64s(), t.Complex64s()[start:start+split]); err != nil {
				panic(err)
			}
			start += split
		}
	case lastAxis:
		// second most efficient
		var at int
		for start := 0; start <= t.len()-size; start += size {
			retVal.SetC64(at, lastFn(t.Complex64s()[start:start+size]))
			at++
		}
	default:
		outerSize := t.Shape()[0]
		outerStride := t.Strides()[0]
		stride := t.Strides()[axis]
		expected := retVal.Strides()[0]

		for i := 0; i < outerSize; i++ {
			start := i * outerStride
			tdata := t.Complex64s()[start : start+outerStride]
			rdata := retVal.Complex64s()
			var innerStart, strideTrack int
			for j := 0; j < expected; j++ {
				for k := 0; k < size; k++ {
					readFrom := innerStart + k*stride
					writeTo := i*expected + j
					a := rdata[writeTo]
					b := tdata[readFrom]
					if k == 0 {
						rdata[writeTo] = b
					} else {
						rdata[writeTo] = defFn(a, b)
					}
				}
				strideTrack++
				if strideTrack >= stride {
					strideTrack = 0
					innerStart += stride
				}
				innerStart++
			}
		}
	}
	return retVal
}

// sReduceC128 is a specialization for complex128 reductions, used in methods such as Sum, Prod, Max etc
func (t *Dense) sReduceC128(axis int, zeroFn func(a, b []complex128) error, lastFn func([]complex128) complex128, defFn func(a, b complex128) complex128) (retVal *Dense) {
	if t.IsScalar() {
		return t
	}

	var newShape Shape
	for i, s := range t.Shape() {
		if i == axis {
			continue
		}
		newShape = append(newShape, s)
	}
	retVal = New(Of(t.t), WithShape(newShape...))
	size := t.Shape()[axis]
	lastAxis := t.Dims() - 1
	switch axis {
	case 0:
		// most efficient
		split := t.len() / size
		copy(retVal.Complex128s()[0:split], t.Complex128s()[0:split])

		start := split
		for i := 0; i < size-1; i++ {
			if err := zeroFn(retVal.Complex128s(), t.Complex128s()[start:start+split]); err != nil {
				panic(err)
			}
			start += split
		}
	case lastAxis:
		// second most efficient
		var at int
		for start := 0; start <= t.len()-size; start += size {
			retVal.SetC128(at, lastFn(t.Complex128s()[start:start+size]))
			at++
		}
	default:
		outerSize := t.Shape()[0]
		outerStride := t.Strides()[0]
		stride := t.Strides()[axis]
		expected := retVal.Strides()[0]

		for i := 0; i < outerSize; i++ {
			start := i * outerStride
			tdata := t.Complex128s()[start : start+outerStride]
			rdata := retVal.Complex128s()
			var innerStart, strideTrack int
			for j := 0; j < expected; j++ {
				for k := 0; k < size; k++ {
					readFrom := innerStart + k*stride
					writeTo := i*expected + j
					a := rdata[writeTo]
					b := tdata[readFrom]
					if k == 0 {
						rdata[writeTo] = b
					} else {
						rdata[writeTo] = defFn(a, b)
					}
				}
				strideTrack++
				if strideTrack >= stride {
					strideTrack = 0
					innerStart += stride
				}
				innerStart++
			}
		}
	}
	return retVal
}
