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

		data := retVal.bools()
		for i := 0; i < size-1; i++ {
			for j := 0; j < split; j++ {
				data[j] = f(data[j], t.getB(j+start))
			}
			start += split
		}
	case reflect.Int:
		var f func(a, b int) int
		if f, ok = fn.(func(a, b int) int); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b int) int. Got %v instead", fn)
		}

		data := retVal.ints()
		for i := 0; i < size-1; i++ {
			for j := 0; j < split; j++ {
				data[j] = f(data[j], t.getI(j+start))
			}
			start += split
		}
	case reflect.Int8:
		var f func(a, b int8) int8
		if f, ok = fn.(func(a, b int8) int8); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b int8) int8. Got %v instead", fn)
		}

		data := retVal.int8s()
		for i := 0; i < size-1; i++ {
			for j := 0; j < split; j++ {
				data[j] = f(data[j], t.getI8(j+start))
			}
			start += split
		}
	case reflect.Int16:
		var f func(a, b int16) int16
		if f, ok = fn.(func(a, b int16) int16); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b int16) int16. Got %v instead", fn)
		}

		data := retVal.int16s()
		for i := 0; i < size-1; i++ {
			for j := 0; j < split; j++ {
				data[j] = f(data[j], t.getI16(j+start))
			}
			start += split
		}
	case reflect.Int32:
		var f func(a, b int32) int32
		if f, ok = fn.(func(a, b int32) int32); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b int32) int32. Got %v instead", fn)
		}

		data := retVal.int32s()
		for i := 0; i < size-1; i++ {
			for j := 0; j < split; j++ {
				data[j] = f(data[j], t.getI32(j+start))
			}
			start += split
		}
	case reflect.Int64:
		var f func(a, b int64) int64
		if f, ok = fn.(func(a, b int64) int64); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b int64) int64. Got %v instead", fn)
		}

		data := retVal.int64s()
		for i := 0; i < size-1; i++ {
			for j := 0; j < split; j++ {
				data[j] = f(data[j], t.getI64(j+start))
			}
			start += split
		}
	case reflect.Uint:
		var f func(a, b uint) uint
		if f, ok = fn.(func(a, b uint) uint); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b uint) uint. Got %v instead", fn)
		}

		data := retVal.uints()
		for i := 0; i < size-1; i++ {
			for j := 0; j < split; j++ {
				data[j] = f(data[j], t.getU(j+start))
			}
			start += split
		}
	case reflect.Uint8:
		var f func(a, b uint8) uint8
		if f, ok = fn.(func(a, b uint8) uint8); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b uint8) uint8. Got %v instead", fn)
		}

		data := retVal.uint8s()
		for i := 0; i < size-1; i++ {
			for j := 0; j < split; j++ {
				data[j] = f(data[j], t.getU8(j+start))
			}
			start += split
		}
	case reflect.Uint16:
		var f func(a, b uint16) uint16
		if f, ok = fn.(func(a, b uint16) uint16); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b uint16) uint16. Got %v instead", fn)
		}

		data := retVal.uint16s()
		for i := 0; i < size-1; i++ {
			for j := 0; j < split; j++ {
				data[j] = f(data[j], t.getU16(j+start))
			}
			start += split
		}
	case reflect.Uint32:
		var f func(a, b uint32) uint32
		if f, ok = fn.(func(a, b uint32) uint32); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b uint32) uint32. Got %v instead", fn)
		}

		data := retVal.uint32s()
		for i := 0; i < size-1; i++ {
			for j := 0; j < split; j++ {
				data[j] = f(data[j], t.getU32(j+start))
			}
			start += split
		}
	case reflect.Uint64:
		var f func(a, b uint64) uint64
		if f, ok = fn.(func(a, b uint64) uint64); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b uint64) uint64. Got %v instead", fn)
		}

		data := retVal.uint64s()
		for i := 0; i < size-1; i++ {
			for j := 0; j < split; j++ {
				data[j] = f(data[j], t.getU64(j+start))
			}
			start += split
		}
	case reflect.Uintptr:
		var f func(a, b uintptr) uintptr
		if f, ok = fn.(func(a, b uintptr) uintptr); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b uintptr) uintptr. Got %v instead", fn)
		}

		data := retVal.uintptrs()
		for i := 0; i < size-1; i++ {
			for j := 0; j < split; j++ {
				data[j] = f(data[j], t.getUintptr(j+start))
			}
			start += split
		}
	case reflect.Float32:
		var f func(a, b float32) float32
		if f, ok = fn.(func(a, b float32) float32); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b float32) float32. Got %v instead", fn)
		}

		data := retVal.float32s()
		for i := 0; i < size-1; i++ {
			for j := 0; j < split; j++ {
				data[j] = f(data[j], t.getF32(j+start))
			}
			start += split
		}
	case reflect.Float64:
		var f func(a, b float64) float64
		if f, ok = fn.(func(a, b float64) float64); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b float64) float64. Got %v instead", fn)
		}

		data := retVal.float64s()
		for i := 0; i < size-1; i++ {
			for j := 0; j < split; j++ {
				data[j] = f(data[j], t.getF64(j+start))
			}
			start += split
		}
	case reflect.Complex64:
		var f func(a, b complex64) complex64
		if f, ok = fn.(func(a, b complex64) complex64); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b complex64) complex64. Got %v instead", fn)
		}

		data := retVal.complex64s()
		for i := 0; i < size-1; i++ {
			for j := 0; j < split; j++ {
				data[j] = f(data[j], t.getC64(j+start))
			}
			start += split
		}
	case reflect.Complex128:
		var f func(a, b complex128) complex128
		if f, ok = fn.(func(a, b complex128) complex128); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b complex128) complex128. Got %v instead", fn)
		}

		data := retVal.complex128s()
		for i := 0; i < size-1; i++ {
			for j := 0; j < split; j++ {
				data[j] = f(data[j], t.getC128(j+start))
			}
			start += split
		}
	case reflect.String:
		var f func(a, b string) string
		if f, ok = fn.(func(a, b string) string); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b string) string. Got %v instead", fn)
		}

		data := retVal.strings()
		for i := 0; i < size-1; i++ {
			for j := 0; j < split; j++ {
				data[j] = f(data[j], t.getStr(j+start))
			}
			start += split
		}
	case reflect.UnsafePointer:
		var f func(a, b unsafe.Pointer) unsafe.Pointer
		if f, ok = fn.(func(a, b unsafe.Pointer) unsafe.Pointer); !ok {
			return errors.Errorf("Expected reduction function to be func(a, b unsafe.Pointer) unsafe.Pointer. Got %v instead", fn)
		}

		data := retVal.unsafePointers()
		for i := 0; i < size-1; i++ {
			for j := 0; j < split; j++ {
				data[j] = f(data[j], t.getUnsafePointer(j+start))
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
				args = append(args, reflect.ValueOf(retVal.get(j)))
				args = append(args, reflect.ValueOf(t.get(start+j)))
				v := f.Call(args)[0].Interface()
				retVal.set(j, v)
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
			r := reduceB(f, def, t.bools()[start:start+size]...)
			retVal.setB(at, r)
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
			r := reduceI(f, def, t.ints()[start:start+size]...)
			retVal.setI(at, r)
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
			r := reduceI8(f, def, t.int8s()[start:start+size]...)
			retVal.setI8(at, r)
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
			r := reduceI16(f, def, t.int16s()[start:start+size]...)
			retVal.setI16(at, r)
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
			r := reduceI32(f, def, t.int32s()[start:start+size]...)
			retVal.setI32(at, r)
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
			r := reduceI64(f, def, t.int64s()[start:start+size]...)
			retVal.setI64(at, r)
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
			r := reduceU(f, def, t.uints()[start:start+size]...)
			retVal.setU(at, r)
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
			r := reduceU8(f, def, t.uint8s()[start:start+size]...)
			retVal.setU8(at, r)
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
			r := reduceU16(f, def, t.uint16s()[start:start+size]...)
			retVal.setU16(at, r)
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
			r := reduceU32(f, def, t.uint32s()[start:start+size]...)
			retVal.setU32(at, r)
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
			r := reduceU64(f, def, t.uint64s()[start:start+size]...)
			retVal.setU64(at, r)
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
			r := reduceUintptr(f, def, t.uintptrs()[start:start+size]...)
			retVal.setUintptr(at, r)
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
			r := reduceF32(f, def, t.float32s()[start:start+size]...)
			retVal.setF32(at, r)
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
			r := reduceF64(f, def, t.float64s()[start:start+size]...)
			retVal.setF64(at, r)
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
			r := reduceC64(f, def, t.complex64s()[start:start+size]...)
			retVal.setC64(at, r)
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
			r := reduceC128(f, def, t.complex128s()[start:start+size]...)
			retVal.setC128(at, r)
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
			r := reduceStr(f, def, t.strings()[start:start+size]...)
			retVal.setStr(at, r)
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
			r := reduceUnsafePointer(f, def, t.unsafePointers()[start:start+size]...)
			retVal.setUnsafePointer(at, r)
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
			retVal.set(at, r)
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
			tdata := t.bools()[start : start+oStride]
			rdata := retVal.bools()
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
			tdata := t.ints()[start : start+oStride]
			rdata := retVal.ints()
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
			tdata := t.int8s()[start : start+oStride]
			rdata := retVal.int8s()
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
			tdata := t.int16s()[start : start+oStride]
			rdata := retVal.int16s()
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
			tdata := t.int32s()[start : start+oStride]
			rdata := retVal.int32s()
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
			tdata := t.int64s()[start : start+oStride]
			rdata := retVal.int64s()
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
			tdata := t.uints()[start : start+oStride]
			rdata := retVal.uints()
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
			tdata := t.uint8s()[start : start+oStride]
			rdata := retVal.uint8s()
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
			tdata := t.uint16s()[start : start+oStride]
			rdata := retVal.uint16s()
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
			tdata := t.uint32s()[start : start+oStride]
			rdata := retVal.uint32s()
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
			tdata := t.uint64s()[start : start+oStride]
			rdata := retVal.uint64s()
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
			tdata := t.uintptrs()[start : start+oStride]
			rdata := retVal.uintptrs()
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
			tdata := t.float32s()[start : start+oStride]
			rdata := retVal.float32s()
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
			tdata := t.float64s()[start : start+oStride]
			rdata := retVal.float64s()
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
			tdata := t.complex64s()[start : start+oStride]
			rdata := retVal.complex64s()
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
			tdata := t.complex128s()[start : start+oStride]
			rdata := retVal.complex128s()
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
			tdata := t.strings()[start : start+oStride]
			rdata := retVal.strings()
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
			tdata := t.unsafePointers()[start : start+oStride]
			rdata := retVal.unsafePointers()
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
					args = append(args, reflect.ValueOf(retVal.get(writeTo)))
					args = append(args, reflect.ValueOf(sliced.get(readFrom)))
					v := f.Call(args)[0].Interface()
					retVal.set(writeTo, v)
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

// reduceS is a specialization for number reductions, used in methods such as Sum, Prod, Max etc
func (t *Dense) reduceS(axis int, zeroFn, oneFn, defFn interface{}) (retVal *Dense) {
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
		copySliced(retVal, 0, split, t, 0, split)

		start := split
		switch t.t.Kind() {

		case reflect.Int:
			vecvecFn := zeroFn.(func(a, b []int))
			for i := 0; i < size-1; i++ {
				vecvecFn(retVal.ints(), t.ints()[start:start+split])
				start += split
			}

		case reflect.Int8:
			vecvecFn := zeroFn.(func(a, b []int8))
			for i := 0; i < size-1; i++ {
				vecvecFn(retVal.int8s(), t.int8s()[start:start+split])
				start += split
			}

		case reflect.Int16:
			vecvecFn := zeroFn.(func(a, b []int16))
			for i := 0; i < size-1; i++ {
				vecvecFn(retVal.int16s(), t.int16s()[start:start+split])
				start += split
			}

		case reflect.Int32:
			vecvecFn := zeroFn.(func(a, b []int32))
			for i := 0; i < size-1; i++ {
				vecvecFn(retVal.int32s(), t.int32s()[start:start+split])
				start += split
			}

		case reflect.Int64:
			vecvecFn := zeroFn.(func(a, b []int64))
			for i := 0; i < size-1; i++ {
				vecvecFn(retVal.int64s(), t.int64s()[start:start+split])
				start += split
			}

		case reflect.Uint:
			vecvecFn := zeroFn.(func(a, b []uint))
			for i := 0; i < size-1; i++ {
				vecvecFn(retVal.uints(), t.uints()[start:start+split])
				start += split
			}

		case reflect.Uint8:
			vecvecFn := zeroFn.(func(a, b []uint8))
			for i := 0; i < size-1; i++ {
				vecvecFn(retVal.uint8s(), t.uint8s()[start:start+split])
				start += split
			}

		case reflect.Uint16:
			vecvecFn := zeroFn.(func(a, b []uint16))
			for i := 0; i < size-1; i++ {
				vecvecFn(retVal.uint16s(), t.uint16s()[start:start+split])
				start += split
			}

		case reflect.Uint32:
			vecvecFn := zeroFn.(func(a, b []uint32))
			for i := 0; i < size-1; i++ {
				vecvecFn(retVal.uint32s(), t.uint32s()[start:start+split])
				start += split
			}

		case reflect.Uint64:
			vecvecFn := zeroFn.(func(a, b []uint64))
			for i := 0; i < size-1; i++ {
				vecvecFn(retVal.uint64s(), t.uint64s()[start:start+split])
				start += split
			}

		case reflect.Float32:
			vecvecFn := zeroFn.(func(a, b []float32))
			for i := 0; i < size-1; i++ {
				vecvecFn(retVal.float32s(), t.float32s()[start:start+split])
				start += split
			}

		case reflect.Float64:
			vecvecFn := zeroFn.(func(a, b []float64))
			for i := 0; i < size-1; i++ {
				vecvecFn(retVal.float64s(), t.float64s()[start:start+split])
				start += split
			}

		case reflect.Complex64:
			vecvecFn := zeroFn.(func(a, b []complex64))
			for i := 0; i < size-1; i++ {
				vecvecFn(retVal.complex64s(), t.complex64s()[start:start+split])
				start += split
			}

		case reflect.Complex128:
			vecvecFn := zeroFn.(func(a, b []complex128))
			for i := 0; i < size-1; i++ {
				vecvecFn(retVal.complex128s(), t.complex128s()[start:start+split])
				start += split
			}
		}
	case lastAxis:
		// second most efficient
		var at int
		switch t.t.Kind() {

		case reflect.Int:
			lastFn := oneFn.(func([]int) int)
			for start := 0; start < t.len()-size; start += size {
				retVal.setI(at, lastFn(t.ints()[start:start+size]))
				at++
			}

		case reflect.Int8:
			lastFn := oneFn.(func([]int8) int8)
			for start := 0; start < t.len()-size; start += size {
				retVal.setI8(at, lastFn(t.int8s()[start:start+size]))
				at++
			}

		case reflect.Int16:
			lastFn := oneFn.(func([]int16) int16)
			for start := 0; start < t.len()-size; start += size {
				retVal.setI16(at, lastFn(t.int16s()[start:start+size]))
				at++
			}

		case reflect.Int32:
			lastFn := oneFn.(func([]int32) int32)
			for start := 0; start < t.len()-size; start += size {
				retVal.setI32(at, lastFn(t.int32s()[start:start+size]))
				at++
			}

		case reflect.Int64:
			lastFn := oneFn.(func([]int64) int64)
			for start := 0; start < t.len()-size; start += size {
				retVal.setI64(at, lastFn(t.int64s()[start:start+size]))
				at++
			}

		case reflect.Uint:
			lastFn := oneFn.(func([]uint) uint)
			for start := 0; start < t.len()-size; start += size {
				retVal.setU(at, lastFn(t.uints()[start:start+size]))
				at++
			}

		case reflect.Uint8:
			lastFn := oneFn.(func([]uint8) uint8)
			for start := 0; start < t.len()-size; start += size {
				retVal.setU8(at, lastFn(t.uint8s()[start:start+size]))
				at++
			}

		case reflect.Uint16:
			lastFn := oneFn.(func([]uint16) uint16)
			for start := 0; start < t.len()-size; start += size {
				retVal.setU16(at, lastFn(t.uint16s()[start:start+size]))
				at++
			}

		case reflect.Uint32:
			lastFn := oneFn.(func([]uint32) uint32)
			for start := 0; start < t.len()-size; start += size {
				retVal.setU32(at, lastFn(t.uint32s()[start:start+size]))
				at++
			}

		case reflect.Uint64:
			lastFn := oneFn.(func([]uint64) uint64)
			for start := 0; start < t.len()-size; start += size {
				retVal.setU64(at, lastFn(t.uint64s()[start:start+size]))
				at++
			}

		case reflect.Float32:
			lastFn := oneFn.(func([]float32) float32)
			for start := 0; start < t.len()-size; start += size {
				retVal.setF32(at, lastFn(t.float32s()[start:start+size]))
				at++
			}

		case reflect.Float64:
			lastFn := oneFn.(func([]float64) float64)
			for start := 0; start < t.len()-size; start += size {
				retVal.setF64(at, lastFn(t.float64s()[start:start+size]))
				at++
			}

		case reflect.Complex64:
			lastFn := oneFn.(func([]complex64) complex64)
			for start := 0; start < t.len()-size; start += size {
				retVal.setC64(at, lastFn(t.complex64s()[start:start+size]))
				at++
			}

		case reflect.Complex128:
			lastFn := oneFn.(func([]complex128) complex128)
			for start := 0; start < t.len()-size; start += size {
				retVal.setC128(at, lastFn(t.complex128s()[start:start+size]))
				at++
			}
		}
	default:
		outerSize := t.Shape()[0]
		outerStride := t.Strides()[0]
		stride := t.Strides()[axis]
		expected := retVal.Strides()[0]
		switch t.t.Kind() {
		case reflect.Int:
			def := defFn.(func(a, b int) int)
			for i := 0; i < outerSize; i++ {
				start := i * outerStride
				tdata := t.ints()[start : start+outerStride]
				rdata := retVal.ints()
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
							rdata[writeTo] = def(a, b)
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
		case reflect.Int8:
			def := defFn.(func(a, b int8) int8)
			for i := 0; i < outerSize; i++ {
				start := i * outerStride
				tdata := t.int8s()[start : start+outerStride]
				rdata := retVal.int8s()
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
							rdata[writeTo] = def(a, b)
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
		case reflect.Int16:
			def := defFn.(func(a, b int16) int16)
			for i := 0; i < outerSize; i++ {
				start := i * outerStride
				tdata := t.int16s()[start : start+outerStride]
				rdata := retVal.int16s()
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
							rdata[writeTo] = def(a, b)
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
		case reflect.Int32:
			def := defFn.(func(a, b int32) int32)
			for i := 0; i < outerSize; i++ {
				start := i * outerStride
				tdata := t.int32s()[start : start+outerStride]
				rdata := retVal.int32s()
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
							rdata[writeTo] = def(a, b)
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
		case reflect.Int64:
			def := defFn.(func(a, b int64) int64)
			for i := 0; i < outerSize; i++ {
				start := i * outerStride
				tdata := t.int64s()[start : start+outerStride]
				rdata := retVal.int64s()
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
							rdata[writeTo] = def(a, b)
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
		case reflect.Uint:
			def := defFn.(func(a, b uint) uint)
			for i := 0; i < outerSize; i++ {
				start := i * outerStride
				tdata := t.uints()[start : start+outerStride]
				rdata := retVal.uints()
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
							rdata[writeTo] = def(a, b)
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
		case reflect.Uint8:
			def := defFn.(func(a, b uint8) uint8)
			for i := 0; i < outerSize; i++ {
				start := i * outerStride
				tdata := t.uint8s()[start : start+outerStride]
				rdata := retVal.uint8s()
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
							rdata[writeTo] = def(a, b)
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
		case reflect.Uint16:
			def := defFn.(func(a, b uint16) uint16)
			for i := 0; i < outerSize; i++ {
				start := i * outerStride
				tdata := t.uint16s()[start : start+outerStride]
				rdata := retVal.uint16s()
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
							rdata[writeTo] = def(a, b)
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
		case reflect.Uint32:
			def := defFn.(func(a, b uint32) uint32)
			for i := 0; i < outerSize; i++ {
				start := i * outerStride
				tdata := t.uint32s()[start : start+outerStride]
				rdata := retVal.uint32s()
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
							rdata[writeTo] = def(a, b)
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
		case reflect.Uint64:
			def := defFn.(func(a, b uint64) uint64)
			for i := 0; i < outerSize; i++ {
				start := i * outerStride
				tdata := t.uint64s()[start : start+outerStride]
				rdata := retVal.uint64s()
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
							rdata[writeTo] = def(a, b)
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
		case reflect.Float32:
			def := defFn.(func(a, b float32) float32)
			for i := 0; i < outerSize; i++ {
				start := i * outerStride
				tdata := t.float32s()[start : start+outerStride]
				rdata := retVal.float32s()
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
							rdata[writeTo] = def(a, b)
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
		case reflect.Float64:
			def := defFn.(func(a, b float64) float64)
			for i := 0; i < outerSize; i++ {
				start := i * outerStride
				tdata := t.float64s()[start : start+outerStride]
				rdata := retVal.float64s()
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
							rdata[writeTo] = def(a, b)
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
		case reflect.Complex64:
			def := defFn.(func(a, b complex64) complex64)
			for i := 0; i < outerSize; i++ {
				start := i * outerStride
				tdata := t.complex64s()[start : start+outerStride]
				rdata := retVal.complex64s()
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
							rdata[writeTo] = def(a, b)
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
		case reflect.Complex128:
			def := defFn.(func(a, b complex128) complex128)
			for i := 0; i < outerSize; i++ {
				start := i * outerStride
				tdata := t.complex128s()[start : start+outerStride]
				rdata := retVal.complex128s()
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
							rdata[writeTo] = def(a, b)
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
	}
	return retVal
}
