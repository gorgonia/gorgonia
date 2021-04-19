package shapes

import (
	"reflect"
	"unsafe"

	"github.com/pkg/errors"
)

type exprtup struct {
	a, b Expr
}

func (t exprtup) freevars() varset {
	retVal := t.a.freevars()
	retVal = append(retVal, t.b.freevars()...)
	return unique(retVal)
}

func prodInts(a []int) int {
	if len(a) == 0 {
		return 0
	}
	retVal := a[0]
	for i := 1; i < len(a); i++ {
		retVal *= a[i]
	}
	return retVal
}

func axesToInts(a Axes) []int {
	return *(*[]int)(unsafe.Pointer(&a))
}

func arrowToTup(a *Arrow) *exprtup {
	return (*exprtup)(unsafe.Pointer(a))
}

// IsMonotonicInts returns true if the slice of ints is monotonically increasing. It also returns true for incr1 if every succession is a succession of 1
func IsMonotonicInts(a []int) (monotonic bool, incr1 bool) {
	var prev int
	incr1 = true
	for i, v := range a {
		if i == 0 {
			prev = v
			continue
		}

		if v < prev {
			return false, false
		}
		if v != prev+1 {
			incr1 = false
		}
		prev = v
	}
	monotonic = true
	return
}

// UnsafePermute permutes the xs according to the pattern. Each x in xs must have the same length as the pattern's length.
func UnsafePermute(pattern []int, xs ...[]int) (err error) {
	if len(xs) == 0 {
		err = errors.New("Permute requres something to permute")
		return
	}

	dims := -1
	patLen := len(pattern)
	for _, x := range xs {
		d := len(x)
		if dims == -1 {
			dims = len(x)
		}
		if d != dims || d != patLen {
			err = errors.Errorf(dimsMismatch, len(x), len(pattern))
			return
		}

	}
	if err = patternCheck(pattern, dims); err != nil {
		return
	}
	unsafePermuteInts(pattern, xs...)
	return nil
}

// unsafePermuteInts is a fast path.
func unsafePermuteInts(pattern []int, xs ...[]int) {
	dims := len(pattern)
	switch dims {
	case 0, 1:
	case 2:
		for _, x := range xs {
			x[0], x[1] = x[1], x[0]
		}
	default:
		for i := 0; i < dims; i++ {
			to := pattern[i]
			for to < i {
				to = pattern[to]
			}
			for _, x := range xs {
				x[i], x[to] = x[to], x[i]
			}
		}
	}
}

// strided slice is a very lightweight "tensor"
type stridedSlice struct {
	data   []byte
	stride int
}

// genericUnsafePermute will permute slices according to the given pattern
//
// FUTURE:go2generics - genericUnsafePermute[T](pattern []int, xs ...[]T) error
func genericUnsafePermute(pattern []int, xs ...interface{}) (err error) {
	if len(xs) == 0 {
		return errors.New("Permute requires something to permute")
	}

	dims := -1
	patLen := len(pattern)

	xs2 := make([]stridedSlice, 0, len(xs))
	allIntSlices := true
	for i, x := range xs {
		// check all are slices
		T := reflect.TypeOf(x)
		if T.Kind() != reflect.Slice {
			return errors.Errorf("Cannot permute %v (%dth of xs). Expected a slice. Got %T instead", x, i, x)
		}
		// check all are ints
		if T.Elem().Kind() != reflect.Int {
			allIntSlices = false
		}

		v := reflect.ValueOf(x)

		// check the dims
		var d int
		if dims == -1 {
			dims = v.Len()
		}
		d = v.Len()
		if d != dims || d != patLen {
			return errors.Errorf(dimsMismatch, d, len(pattern))
		}

		// all good? now we cast the data into a byte slice.
		stride := int(T.Elem().Size())
		data := *(*[]byte)(unsafe.Pointer(&reflect.SliceHeader{Data: v.Pointer(), Len: v.Len() * stride, Cap: v.Cap() * stride}))
		xs2 = append(xs2, stridedSlice{data: data, stride: stride})
	}

	// we redirect to a fast path
	// FUTURE:go2generics - when there is generics, genericUnsafePermute will be the exported function UnsafePermute.
	if allIntSlices {
		intses := make([][]int, 0, len(xs2))
		for _, x := range xs2 {
			hdr := reflect.SliceHeader{
				Data: uintptr(unsafe.Pointer(&x.data[0])),
				Len:  len(x.data) / x.stride,
				Cap:  len(x.data) / x.stride,
			}

			intses = append(intses, *(*[]int)(unsafe.Pointer(&hdr)))
		}
		unsafePermuteInts(pattern, intses...)
		return nil
	}

	// check that patterns are valid, non monotonic and increasing
	if err = patternCheck(pattern, dims); err != nil {
		return err
	}

	// perform permutation
	switch dims {
	case 0, 1:
	case 2:
		var tmp []byte
		for _, x := range xs2 {
			if tmp == nil {
				tmp = make([]byte, x.stride)
			}
			swap2(x, 0, 1, tmp)
		}
	default:
		for i := 0; i < dims; i++ {
			to := pattern[i]
			for to < i {
				to = pattern[to]
			}
			var tmp []byte
			for _, x := range xs2 {
				if tmp == nil {
					tmp = make([]byte, x.stride)
				}
				if len(tmp) != x.stride {
					tmp = make([]byte, x.stride)
				}
				swap2(x, i, to, tmp)
			}
		}

	}
	return nil
}

// patternCheck checks patterns
func patternCheck(pattern []int, dims int) (err error) {
	// check that all the axes are < nDims
	// and that there are no axis repeated
	seen := make(map[int]struct{})
	for _, a := range pattern {
		if a >= dims {
			err = errors.Errorf(invalidAxis, a, dims)
			return
		}

		if _, ok := seen[a]; ok {
			err = errors.Errorf(repeatedAxis, a)
			return
		}

		seen[a] = struct{}{}
	}

	// no op really... we did the checks for no reason too. Maybe move this up?
	if monotonic, incr1 := IsMonotonicInts(pattern); monotonic && incr1 {
		err = noopError{}
		return
	}
	return nil
}

func swap2(x stridedSlice, i, j int, tmp []byte) {
	if i == j {
		// nothing to swap
		return
	}
	bs := x.data
	stride := x.stride
	switch stride {
	case 1:
		bs[i], bs[j] = bs[j], bs[i]
	default:
		a := bs[i*stride : i*stride+stride]
		b := bs[j*stride : j*stride+stride]
		copy(tmp, a)
		copy(a, b)
		copy(b, tmp)
	}
}

// CheckSlice checks a slice to see if it's sane
func CheckSlice(s Slice, size int) error {
	start := s.Start()
	end := s.End()
	step := s.Step()

	if start > end {
		return errors.Errorf(invalidSliceIndex, start, end)
	}

	if start < 0 {
		return errors.Errorf(invalidSliceIndex, start, 0)
	}

	if step == 0 && end-start > 1 {
		return errors.Errorf("Slice has 0 steps. Start is %d and end is %d", start, end)
	}

	if start >= size {
		return errors.Errorf("Start %d is greater than size %d", start, size)
	}

	return nil
}

// SliceDetails is a function that takes a slice and spits out its details. The whole reason for this is to handle the nil Slice, which is this: a[:]
func SliceDetails(s Slice, size int) (start, end, step int, err error) {
	if s == nil {
		start = 0
		end = size
		step = 1
	} else {
		if err = CheckSlice(s, size); err != nil {
			return
		}

		start = s.Start()
		end = s.End()
		step = s.Step()

		if end > size {
			end = size
		}
	}
	return
}

// ShapeOf returns the shape of a given datum.
func ShapeOf(a interface{}) Expr {
	switch at := a.(type) {
	case Shaper:
		return at.Shape()
	case Exprer:
		return at.Shape()
	}

	t := reflect.TypeOf(a)
	switch t.Kind() {
	case reflect.Func:
		in := t.NumIn()
		out := t.NumOut()
		if out != 1 {
			panic("Cannot handle functions with multiple outputs. Please feel free to file a Pull Request")
		}
		outExpr := ShapeOf(t.Out(0))
		inExpr := make([]Expr, 0, in)
		for i := 0; i < in; i++ {
			it := t.In(i)
			inExpr = append(inExpr, ShapeOf(it))
		}

		B := Arrow{A: inExpr[len(inExpr)-1], B: outExpr}
		for i := in - 2; i >= 0; i-- {
			A := inExpr[i]
			B = Arrow{A: A, B: B}
		}
		return B

	case reflect.Array:
		return Shape{t.Len()}
	case reflect.Slice, reflect.Chan:
		v := reflect.ValueOf(a)
		return Shape{v.Len()}
	case reflect.Interface:
		panic("Cannot turn Interface type into shape. Please feel free to file a Pull Request")
	case reflect.Map:
		panic("Cannot turn Map type into shape.")
	default:
		return Shape{}
	}
	panic("Unreachable")
}
