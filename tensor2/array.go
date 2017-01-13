package tensor

import (
	"github.com/chewxy/vecf32"
	"github.com/chewxy/vecf64"
	"github.com/pkg/errors"
)

// An Array is a representation of a backing for a Dense Tensor
type Array interface {
	Len() int // returns the length of the array
	Cap() int // returns the cap of the array

	Get(i int) interface{}          // Gets the value at index i
	Set(i int, v interface{}) error // Sets the value at index i to the value
	Map(fn interface{}) error       // fn should be func(x T) T, but your mileage may vary

	Eq
	Dataer
	Zeroer
	MemSetter
}

// provided Arrays

type f64s []float64
type f32s []float32
type ints []int
type i64s []int64
type i32s []int32
type u8s []byte
type bs []bool

// Array creation functions

func makeArray(dt Dtype, size int) Array {
	switch dt {
	case Float64:
		return make(f64s, size, size)
	case Float32:
		return make(f32s, size, size)
	case Int:
		return make(ints, size, size)
	case Int64:
		return make(i64s, size, size)
	case Int32:
		return make(i32s, size, size)
	case Byte:
		return make(u8s, size, size)
	case Bool:
		return make(bs, size, size)
	}

	if am, ok := dt.(ArrayMaker); ok {
		return am.MakeArray(size)
	}
	panic("Unsupported Dtype")
}

func arrayFromInterface(a interface{}) Array {
	switch at := a.(type) {
	case Array:
		return at
	case []float64:
		return f64s(at)
	case []float32:
		return f32s(at)
	case []int:
		return ints(at)
	case []int64:
		return i64s(at)
	case []int32:
		return i32s(at)
	case []byte:
		return u8s(at)
	case []bool:
		return bs(at)
	}
	panic("Unreachable")
}

func fromInterfaceSlice(dt Dtype, s []interface{}) Array {
	size := len(s)
	switch dt {
	case Float64:
		arr := make(f64s, size, size)
		for i, v := range s {
			arr[i] = v.(float64)
		}
		return arr
	case Float32:
		arr := make(f32s, size, size)
		for i, v := range s {
			arr[i] = v.(float32)
		}
		return arr
	case Int:
		arr := make(ints, size, size)
		for i, v := range s {
			arr[i] = v.(int)
		}
		return arr
	case Int64:
		arr := make(i64s, size, size)
		for i, v := range s {
			arr[i] = v.(int64)
		}
		return arr
	case Int32:
		arr := make(i32s, size, size)
		for i, v := range s {
			arr[i] = v.(int32)
		}
		return arr
	case Byte:
		arr := make(u8s, size, size)
		for i, v := range s {
			arr[i] = v.(byte)
		}
		return arr
	case Bool:
		arr := make(bs, size, size)
		for i, v := range s {
			arr[i] = v.(bool)
		}
		return arr
	}

	if am, ok := dt.(FromInterfaceSlicer); ok {
		return am.FromInterfaceSlice(s)
	}
	panic("Unsupported Dtype")
}

/* ARRAY OP STUFF */

// Slicer slices the array. It's a standin for doing a[start:end]
type Slicer interface {
	Slice(start, end int) (Array, error)
}

// CopierFrom copies from source to the receiver. It returns an int indicating how many bytes or elements have been copied
type CopierFrom interface {
	CopyFrom(src interface{}) (int, error)
}

// CopierTo copies from the receiver to the dest. It returns an int indicating how many bytes or elements have been copied
type CopierTo interface {
	CopyTo(dest interface{}) (int, error)
}

// Transposer is any array that provides a specialization for transposing.
type Transposer interface {
	Transpose(oldShape, oldStrides, axes, newStrides []int)
}

// MapIncr is a specialization for map
type IncrMapper interface {
	MapIncr(fn interface{}) error
}

// IterateMapper is a specialization for map
type IterMapper interface {
	IterMap(other Array, it, ot *FlatIterator, fn interface{}, incr bool) error
}

// Tracer is any array that provides a specialization for a linear algebra trace. Do note that while the mathematical definition
// of a trace is only defined on a square matrix, package Tensor actually supports non-square matrices. This is provided by the min
// argument. Here is a sample implementation of Trace():
// 		type foos []foo
//		func (fs foos) Trace(rowStride, colStride, min int) (interface{}, error){
//			var trace foo
//			for i := 0; i < m; i++ {
//				trace += fs[i * (rowStride+colStride)]
//			}
//			return trace, nil
//		}
type Tracer interface {
	Trace(rowStride, colStride, min int) (interface{}, error)
}

/* BASIC ARRAY TYPE HANDLING */

// Float64ser is any array that can turn into a []float64
type Float64ser interface {
	Float64s() []float64
}

// Float64ser is any array that can turn into a []float32
type Float32ser interface {
	Float32s() []float32
}

// Float64ser is any array that can turn into a []int
type Intser interface {
	Ints() []int
}

// Int64ser is any array that can turn into a []int64
type Int64ser interface {
	Int64s() []int64
}

// Int32ser is any array that can turn into a []int32
type Int32ser interface {
	Int32s() []int32
}

// Byteser is any array that can turn into a []byte
type Byteser interface {
	Bytes() []byte
}

// Boolser is any array that can turn into a []bool
type Boolser interface {
	Bools() []bool
}

// Dtyper is for any array implementation that knows its own Dtype
type Dtyper interface {
	Dtype() Dtype
}

/* OTHER TYPE CLASSES */

// Number is any array where you can perform basic arithmetic on. The arithmethic methods are expected to clober the value of the receiver
type Number interface {
	Array

	// Array-Array interactions
	Add(Number) error
	Sub(Number) error
	Mul(Number) error
	Div(Number) error
	Pow(Number) error

	// Array-Scalar interactions
	Trans(interface{}) error
	TransInv(interface{}) error
	TransInvR(interface{}) error
	Scale(interface{}) error
	ScaleInv(interface{}) error
	ScaleInvR(interface{}) error
	PowOf(interface{}) error
	PowOfR(interface{}) error

	// Incremental interactions
	IncrAdd(other, incr Number) error
	IncrSub(other, incr Number) error
	IncrMul(other, incr Number) error
	IncrDiv(other, incr Number) error
	IncrPow(other, incr Number) error
	IncrTrans(x interface{}, incr Number) error
	IncrTransInv(x interface{}, incr Number) error
	IncrTransInvR(x interface{}, incr Number) error
	IncrScale(x interface{}, incr Number) error
	IncrScaleInv(x interface{}, incr Number) error
	IncrScaleInvR(x interface{}, incr Number) error
	IncrPowOf(x interface{}, incr Number) error
	IncrPowOfR(x interface{}, incr Number) error
}

// Float is any array where you can perform floating point operations. Arrays that also implement Float will have linalg performed
type Float interface {
	Number
	HasNaN() bool
	HasInf() bool
}

// ElemEq is any array type that you can perform elementwise equality on
type ElEq interface {
	Array
	Oner

	ElEq(other ElEq, same bool) (Array, error)
}

// ElOrd is any array where you can perform an ordered comparison elementwise
type ElOrd interface {
	ElEq
	Lt(other ElOrd, same bool) (Array, error)
	Lte(other ElOrd, same bool) (Array, error)
	Gt(other ElOrd, same bool) (Array, error)
	Gte(other ElOrd, same bool) (Array, error)
}

/* FUNCTIONS */

// Range creates a ranged array with a given type. It panics if the dt is not the provided ones
func Range(dt Dtype, start, end int) Array {
	size := end - start
	incr := true
	if start > end {
		incr = false
		size = start - end
	}

	if size < 0 {
		panic("Cannot create a range that is negative in size")
	}

	switch dt {
	case Float64:
		return f64s(vecf64.Range(start, end))
	case Float32:
		return f32s(vecf32.Range(start, end))
	case Int:
		r := make([]int, size)
		for i, v := 0, int(start); i < size; i++ {
			r[i] = v

			if incr {
				v++
			} else {
				v--
			}
		}
		return ints(r)
	case Int64:
		r := make([]int64, size)
		for i, v := 0, int64(start); i < size; i++ {
			r[i] = v

			if incr {
				v++
			} else {
				v--
			}
		}
		return i64s(r)
	case Int32:
		// TODO: Overflow checks
		r := make([]int32, size)
		for i, v := 0, int32(start); i < size; i++ {
			r[i] = v

			if incr {
				v++
			} else {
				v--
			}
		}
		return i32s(r)
	case Byte:
		// TODO: Overflow checks
		r := make([]byte, size)
		for i, v := 0, byte(start); i < size; i++ {
			r[i] = v

			if incr {
				v++
			} else {
				v--
			}
		}
		return u8s(r)
	default:
		panic("Unrangeable dt")
	}
}

func copyArray(dest, src Array) (int, error) {
	var ok bool

	// switch on known arrays
	switch dt := dest.(type) {
	case f64s:
		var st f64s
		if st, ok = src.(f64s); !ok {
			return 0, errors.Errorf(typeMismatch, dest, src)
		}
		return copy(dt, st), nil
	case f32s:
		var st f32s
		if st, ok = src.(f32s); !ok {
			return 0, errors.Errorf(typeMismatch, dest, src)
		}
		return copy(dt, st), nil
	case ints:
		var st ints
		if st, ok = src.(ints); !ok {
			return 0, errors.Errorf(typeMismatch, dest, src)
		}
		return copy(dt, st), nil
	case i64s:
		var st i64s
		if st, ok = src.(i64s); !ok {
			return 0, errors.Errorf(typeMismatch, dest, src)
		}
		return copy(dt, st), nil
	case i32s:
		var st i32s
		if st, ok = src.(i32s); !ok {
			return 0, errors.Errorf(typeMismatch, dest, src)
		}
		return copy(dt, st), nil
	case u8s:
		var st u8s
		if st, ok = src.(u8s); !ok {
			return 0, errors.Errorf(typeMismatch, dest, src)
		}
		return copy(dt, st), nil
	case bs:
		var st bs
		if st, ok = src.(bs); !ok {
			return 0, errors.Errorf(typeMismatch, dest, src)
		}
		return copy(dt, st), nil
	default:
		// go down
	}

	if cf, ok := dest.(CopierFrom); ok {
		return cf.CopyFrom(src)
	}

	if ct, ok := src.(CopierTo); ok {
		return ct.CopyTo(dest)
	}

	return 0, errors.Errorf("Unable to copy %v to %v", src, dest)
}

func copySlicedArray(dest Array, dStart, dEnd int, src Array, sStart, sEnd int) (int, error) {

	var ok bool
	// switch on known arrays
	switch dt := dest.(type) {
	case f64s:
		var st f64s
		if st, ok = src.(f64s); !ok {
			return 0, errors.Errorf(typeMismatch, dest, src)
		}
		return copy(dt[dStart:dEnd], st[sStart:sEnd]), nil
	case f32s:
		var st f32s
		if st, ok = src.(f32s); !ok {
			return 0, errors.Errorf(typeMismatch, dest, src)
		}
		return copy(dt[dStart:dEnd], st[sStart:sEnd]), nil
	case ints:
		var st ints
		if st, ok = src.(ints); !ok {
			return 0, errors.Errorf(typeMismatch, dest, src)
		}
		return copy(dt[dStart:dEnd], st[sStart:sEnd]), nil
	case i64s:
		var st i64s
		if st, ok = src.(i64s); !ok {
			return 0, errors.Errorf(typeMismatch, dest, src)
		}
		return copy(dt[dStart:dEnd], st[sStart:sEnd]), nil
	case i32s:
		var st i32s
		if st, ok = src.(i32s); !ok {
			return 0, errors.Errorf(typeMismatch, dest, src)
		}
		return copy(dt[dStart:dEnd], st[sStart:sEnd]), nil
	case u8s:
		var st u8s
		if st, ok = src.(u8s); !ok {
			return 0, errors.Errorf(typeMismatch, dest, src)
		}
		return copy(dt[dStart:dEnd], st[sStart:sEnd]), nil
	case bs:
		var st bs
		if st, ok = src.(bs); !ok {
			return 0, errors.Errorf(typeMismatch, dest, src)
		}
		return copy(dt[dStart:dEnd], st[sStart:sEnd]), nil
	default:
		// go down
	}

	var destS, srcS Slicer
	if destS, ok = dest.(Slicer); !ok {
		return 0, errors.Errorf("Cannot copy to sliced array. %T is not a Slicer.", dest)
	}

	if srcS, ok = src.(Slicer); !ok {
		return 0, errors.Errorf("Cannot copy to sliced array. %T is not a Slicer.", src)
	}

	var err error
	if dest, err = destS.Slice(dStart, dEnd); err != nil {
		return 0, errors.Wrapf(err, "Slicing of dest[%d:%d] failed. ", dStart, dEnd)
	}

	if src, err = srcS.Slice(sStart, sEnd); err != nil {
		return 0, errors.Wrapf(err, "Slicing of src[%d:%d] failed. ", sStart, sEnd)
	}

	return copyArray(dest, src)
}

func sliceArray(a Array, start, end int) (Array, error) {
	// switch on known arrays
	switch at := a.(type) {
	case f64s:
		return at[start:end], nil
	case f32s:
		return at[start:end], nil
	case ints:
		return at[start:end], nil
	case i64s:
		return at[start:end], nil
	case i32s:
		return at[start:end], nil
	case u8s:
		return at[start:end], nil
	case bs:
		return at[start:end], nil
	default:
		// go down
	}

	if as, ok := a.(Slicer); ok {
		return as.Slice(start, end)
	}
	return nil, errors.Errorf("Unable to slice %T: does not implement Slicer", a)
}

func typeOf(a Array) (Dtype, error) {
	switch at := a.(type) {
	case f64s:
		return Float64, nil
	case f32s:
		return Float32, nil
	case ints:
		return Int, nil
	case i64s:
		return Int64, nil
	case i32s:
		return Int32, nil
	case u8s:
		return Byte, nil
	case bs:
		return Bool, nil

	case Dtyper:
		return at.Dtype(), nil
	}

	return nil, errors.Errorf("Array %T has no known Dtype", a)
}

// adapted from roger pepe's code here  https://groups.google.com/d/msg/golang-nuts/_Pj9S_Ljp9o/GMo5uPzHbeAJ
func overlaps(a, b Array) bool {
	if a.Cap() == 0 || b.Cap() == 0 {
		return false
	}

	var ok bool
	switch at := a.(type) {
	case f64s:
		var bt f64s
		if bt, ok = b.(f64s); !ok {
			return false
		}
		if &at[0:cap(at)][cap(at)-1] != &bt[0:cap(bt)][cap(bt)-1] {
			return false
		}

	case f32s:
		var bt f32s
		if bt, ok = b.(f32s); !ok {
			return false
		}
		if &at[0:cap(at)][cap(at)-1] != &bt[0:cap(bt)][cap(bt)-1] {
			return false
		}
	case ints:
		var bt ints
		if bt, ok = b.(ints); !ok {
			return false
		}
		if &at[0:cap(at)][cap(at)-1] != &bt[0:cap(bt)][cap(bt)-1] {
			return false
		}
	case i64s:
		var bt i64s
		if bt, ok = b.(i64s); !ok {
			return false
		}
		if &at[0:cap(at)][cap(at)-1] != &bt[0:cap(bt)][cap(bt)-1] {
			return false
		}
	case i32s:
		var bt i32s
		if bt, ok = b.(i32s); !ok {
			return false
		}
		if &at[0:cap(at)][cap(at)-1] != &bt[0:cap(bt)][cap(bt)-1] {
			return false
		}
	case u8s:
		var bt u8s
		if bt, ok = b.(u8s); !ok {
			return false
		}
		if &at[0:cap(at)][cap(at)-1] != &bt[0:cap(bt)][cap(bt)-1] {
			return false
		}
	case bs:
		var bt bs
		if bt, ok = b.(bs); !ok {
			return false
		}
		if &at[0:cap(at)][cap(at)-1] != &bt[0:cap(bt)][cap(bt)-1] {
			return false
		}
	default:
		// we don't check for overlaps in unknown slices
		// TODO: fix this
		return false
	}

	a0 := -a.Cap()
	a1 := a0 + a.Len()
	b0 := -b.Cap()
	b1 := b0 + b.Len()
	return a1 > b0 && b1 > a0
}

func assignArray(dest, src *Dense) (err error) {
	// var copiedSrc bool

	if src.IsScalar() {
		panic("HELP")
	}

	dd := dest.Dims()
	sd := src.Dims()

	ds := dest.Strides()
	ss := src.Strides()

	// when dd == 1, and the strides point in the same direction
	// we copy to a temporary if there is an overlap of data
	if ((dd == 1 && sd >= 1 && ds[0]*ss[sd-1] < 0) || dd > 1) && overlaps(dest.data, src.data) {
		// create temp
		// copiedSrc = true
	}

	// broadcast src to dest for raw iteration
	tmpShape := Shape(BorrowInts(sd))
	tmpStrides := BorrowInts(len(src.Strides()))
	copy(tmpShape, src.Shape())
	copy(tmpStrides, src.Strides())
	defer ReturnInts(tmpShape)
	defer ReturnInts(tmpStrides)

	if sd > dd {
		tmpDim := sd
		for tmpDim > dd && tmpShape[0] == 1 {
			tmpDim--

			// this is better than tmpShape = tmpShape[1:]
			// because we are going to return these ints later
			copy(tmpShape, tmpShape[1:])
			copy(tmpStrides, tmpStrides[1:])
		}
	}

	var newStrides []int
	if newStrides, err = BroadcastStrides(dest.Shape(), tmpShape, ds, tmpStrides); err != nil {
		return
	}

	dap := dest.AP
	sap := NewAP(tmpShape, newStrides)

	diter := NewFlatIterator(dap)
	siter := NewFlatIterator(sap)
	// dch := diter.Chan()
	// sch := siter.Chan()

	if im, ok := dest.data.(IterMapper); ok {
		return im.IterMap(src.data, diter, siter, nil, false)
	}

	// slow methods used if not IterMap
	var i, j int
	// var ok bool
	for {
		if i, err = diter.Next(); err != nil {
			if _, ok := err.(NoOpError); !ok {
				return err
			}
			err = nil
			break
		}
		if j, err = siter.Next(); err != nil {
			if _, ok := err.(NoOpError); !ok {
				return err
			}
			err = nil
			break
		}
		// dest.data[i] = src.data[j]
		dest.data.Set(i, src.data.Get(j))
	}

	return
}
