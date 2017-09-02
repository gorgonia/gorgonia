package tensor

import (
	"errors"
	"math"
	"math/cmplx"
	"math/rand"
	"reflect"
	"testing"
	"testing/quick"
	"time"
	"unsafe"

	"github.com/chewxy/gorgonia/tensor/internal/storage"
	"github.com/chewxy/math32"
)

func randomBool() bool {
	i := rand.Intn(11)
	return i > 5
}

// from : https://stackoverflow.com/a/31832326/3426066
const letterBytes = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
const (
	letterIdxBits = 6                    // 6 bits to represent a letter index
	letterIdxMask = 1<<letterIdxBits - 1 // All 1-bits, as many as letterIdxBits
	letterIdxMax  = 63 / letterIdxBits   // # of letter indices fitting in 63 bits
)

var src = rand.NewSource(time.Now().UnixNano())

func randomString() string {
	n := rand.Intn(10)
	b := make([]byte, n)
	// A src.Int63() generates 63 random bits, enough for letterIdxMax characters!
	for i, cache, remain := n-1, src.Int63(), letterIdxMax; i >= 0; {
		if remain == 0 {
			cache, remain = src.Int63(), letterIdxMax
		}
		if idx := int(cache & letterIdxMask); idx < len(letterBytes) {
			b[i] = letterBytes[idx]
			i--
		}
		cache >>= letterIdxBits
		remain--
	}

	return string(b)
}

// taken from the Go Stdlib package math
func tolerancef64(a, b, e float64) bool {
	d := a - b
	if d < 0 {
		d = -d
	}

	// note: b is correct (expected) value, a is actual value.
	// make error tolerance a fraction of b, not a.
	if b != 0 {
		e = e * b
		if e < 0 {
			e = -e
		}
	}
	return d < e
}
func closeenoughf64(a, b float64) bool { return tolerancef64(a, b, 1e-8) }
func closef64(a, b float64) bool       { return tolerancef64(a, b, 1e-14) }
func veryclosef64(a, b float64) bool   { return tolerancef64(a, b, 4e-16) }
func soclosef64(a, b, e float64) bool  { return tolerancef64(a, b, e) }
func alikef64(a, b float64) bool {
	switch {
	case math.IsNaN(a) && math.IsNaN(b):
		return true
	case a == b:
		return math.Signbit(a) == math.Signbit(b)
	}
	return false
}

// taken from math32, which was taken from the Go std lib
func tolerancef32(a, b, e float32) bool {
	d := a - b
	if d < 0 {
		d = -d
	}

	// note: b is correct (expected) value, a is actual value.
	// make error tolerance a fraction of b, not a.
	if b != 0 {
		e = e * b
		if e < 0 {
			e = -e
		}
	}
	return d < e
}
func closef32(a, b float32) bool      { return tolerancef32(a, b, 1e-5) } // the number gotten from the cfloat standard. Haskell's Linear package uses 1e-6 for floats
func veryclosef32(a, b float32) bool  { return tolerancef32(a, b, 1e-6) } // from wiki
func soclosef32(a, b, e float32) bool { return tolerancef32(a, b, e) }
func alikef32(a, b float32) bool {
	switch {
	case math32.IsNaN(a) && math32.IsNaN(b):
		return true
	case a == b:
		return math32.Signbit(a) == math32.Signbit(b)
	}
	return false
}

// taken from math/cmplx test
func cTolerance(a, b complex128, e float64) bool {
	d := cmplx.Abs(a - b)
	if b != 0 {
		e = e * cmplx.Abs(b)
		if e < 0 {
			e = -e
		}
	}
	return d < e
}

func cClose(a, b complex128) bool              { return cTolerance(a, b, 1e-14) }
func cSoclose(a, b complex128, e float64) bool { return cTolerance(a, b, e) }
func cVeryclose(a, b complex128) bool          { return cTolerance(a, b, 4e-16) }
func cAlike(a, b complex128) bool {
	switch {
	case cmplx.IsNaN(a) && cmplx.IsNaN(b):
		return true
	case a == b:
		return math.Signbit(real(a)) == math.Signbit(real(b)) && math.Signbit(imag(a)) == math.Signbit(imag(b))
	}
	return false
}

func allClose(a, b interface{}, approxFn ...interface{}) bool {
	switch at := a.(type) {
	case []float64:
		closeness := closef64
		var ok bool
		if len(approxFn) > 0 {
			if closeness, ok = approxFn[0].(func(a, b float64) bool); !ok {
				closeness = closef64
			}
		}
		bt := b.([]float64)
		for i, v := range at {
			if math.IsNaN(v) {
				if !math.IsNaN(bt[i]) {
					return false
				}
				continue
			}
			if math.IsInf(v, 0) {
				if !math.IsInf(bt[i], 0) {
					return false
				}
				continue
			}
			if !closeness(v, bt[i]) {
				return false
			}
		}
		return true
	case []float32:
		closeness := closef32
		var ok bool
		if len(approxFn) > 0 {
			if closeness, ok = approxFn[0].(func(a, b float32) bool); !ok {
				closeness = closef32
			}
		}
		bt := b.([]float32)
		for i, v := range at {
			if math32.IsNaN(v) {
				if !math32.IsNaN(bt[i]) {
					return false
				}
				continue
			}
			if math32.IsInf(v, 0) {
				if !math32.IsInf(bt[i], 0) {
					return false
				}
				continue
			}
			if !closeness(v, bt[i]) {
				return false
			}
		}
		return true
	case []complex64:
		bt := b.([]complex64)
		for i, v := range at {
			if cmplx.IsNaN(complex128(v)) {
				if !cmplx.IsNaN(complex128(bt[i])) {
					return false
				}
				continue
			}
			if cmplx.IsInf(complex128(v)) {
				if !cmplx.IsInf(complex128(bt[i])) {
					return false
				}
				continue
			}
			if !cSoclose(complex128(v), complex128(bt[i]), 1e-5) {
				return false
			}
		}
		return true
	case []complex128:
		bt := b.([]complex128)
		for i, v := range at {
			if cmplx.IsNaN(v) {
				if !cmplx.IsNaN(bt[i]) {
					return false
				}
				continue
			}
			if cmplx.IsInf(v) {
				if !cmplx.IsInf(bt[i]) {
					return false
				}
				continue
			}
			if !cClose(v, bt[i]) {
				return false
			}
		}
		return true
	default:
		return reflect.DeepEqual(a, b)
	}
}
func checkErr(t *testing.T, expected bool, err error, name string, id interface{}) (cont bool) {
	switch {
	case expected:
		if err == nil {
			t.Errorf("Expected error in test %v (%v)", name, id)
		}
		return true
	case !expected && err != nil:
		t.Errorf("Test %v (%v) errored: %+v", name, id, err)
		return true
	}
	return false
}

func sliceApproxf64(a, b []float64, fn func(a, b float64) bool) bool {
	if len(a) != len(b) {
		return false
	}

	for i, v := range a {
		if math.IsNaN(v) {
			if !alikef64(v, b[i]) {
				return false
			}
		}
		if !fn(v, b[i]) {
			return false
		}
	}
	return true
}

func RandomFloat64(size int) []float64 {
	r := make([]float64, size)
	for i := range r {
		r[i] = rand.NormFloat64()
	}
	return r
}

func factorize(a int) []int {
	if a <= 0 {
		return nil
	}
	// all numbers are divisible by at least 1
	retVal := make([]int, 1)
	retVal[0] = 1

	fill := func(a int, e int) {
		n := len(retVal)
		for i, p := 0, a; i < e; i, p = i+1, p*a {
			for j := 0; j < n; j++ {
				retVal = append(retVal, retVal[j]*p)
			}
		}
	}
	// find factors of 2
	// rightshift by 1 = division by 2
	var e int
	for ; a&1 == 0; e++ {
		a >>= 1
	}
	fill(2, e)

	// find factors of 3 and up
	for next := 3; a > 1; next += 2 {
		if next*next > a {
			next = a
		}
		for e = 0; a%next == 0; e++ {
			a /= next
		}
		if e > 0 {
			fill(next, e)
		}
	}
	return retVal
}

func shuffleInts(a []int, r *rand.Rand) {
	for i := range a {
		j := r.Intn(i + 1)
		a[i], a[j] = a[j], a[i]
	}
}

func (t *Dense) Generate(r *rand.Rand, size int) reflect.Value {
	// generate type
	ri := r.Intn(len(specializedTypes.set))
	of := specializedTypes.set[ri]
	datatyp := reflect.SliceOf(of.Type)
	gendat, _ := quick.Value(datatyp, r)
	// generate dims
	var scalar bool
	var s Shape
	dims := r.Intn(5) // dims4 is the max we'll generate even though we can handle much more
	l := gendat.Len()

	// generate shape based on inputs
	switch {
	case dims == 0 || l == 0:
		scalar = true
		gendat, _ = quick.Value(of.Type, r)
	case dims == 1:
		s = Shape{gendat.Len()}
	default:
		factors := factorize(l)
		s = Shape(BorrowInts(dims))
		// fill with 1s so that we can get a non-zero TotalSize
		for i := 0; i < len(s); i++ {
			s[i] = 1
		}

		for i := 0; i < dims; i++ {
			j := rand.Intn(len(factors))
			s[i] = factors[j]
			size := s.TotalSize()
			if q, r := divmod(l, size); r != 0 {
				factors = factorize(r)
			} else if size != l {
				if i < dims-2 {
					factors = factorize(q)
				} else if i == dims-2 {
					s[i+1] = q
					break
				}
			} else {
				break
			}
		}
		shuffleInts(s, r)
	}

	// generate flags
	flag := MemoryFlag(r.Intn(4))

	// generate order
	order := DataOrder(r.Intn(4))

	var v *Dense
	if scalar {
		v = New(FromScalar(gendat.Interface()))
	} else {
		v = New(Of(of), WithShape(s...), WithBacking(gendat.Interface()))
	}

	v.flag = flag
	v.AP.o = order

	// generate engine
	eint := r.Intn(3)
	switch eint {
	case 0:
		v.e = StdEng{}
	case 1:
		// check is to prevent panics which Float64Engine will do if asked to allocate memory for non float64s
		if of == Float64 {
			v.e = Float64Engine{}
		} else {
			v.e = StdEng{}
		}
	case 2:
		v.e = dummyEngine(true)
	}

	return reflect.ValueOf(v)
}

// fakemem is a byteslice, while making it a Memory
type fakemem []byte

func (m fakemem) Uintptr() uintptr        { return uintptr(unsafe.Pointer(&m[0])) }
func (m fakemem) MemSize() uintptr        { return uintptr(len(m)) }
func (m fakemem) Pointer() unsafe.Pointer { return unsafe.Pointer(&m[0]) }

// dummyEngine implements Engine. The bool indicates whether the data is native-accessible
type dummyEngine bool

func (e dummyEngine) AllocAccessible() bool { return bool(e) }
func (e dummyEngine) Alloc(size int64) (Memory, error) {
	ps := make(fakemem, int(size))
	return ps, nil
}
func (e dummyEngine) Free(mem Memory, size int64) error        { return nil }
func (e dummyEngine) Memset(mem Memory, val interface{}) error { return nil }
func (e dummyEngine) Memclr(mem Memory)                        {}
func (e dummyEngine) Memcpy(dst, src Memory) error {
	if e {
		var a, b storage.Header
		a.Ptr = src.Pointer()
		a.L = int(src.MemSize())
		a.C = int(src.MemSize())

		b.Ptr = dst.Pointer()
		b.L = int(dst.MemSize())
		b.C = int(dst.MemSize())

		abs := *(*[]byte)(unsafe.Pointer(&a))
		bbs := *(*[]byte)(unsafe.Pointer(&b))
		copy(bbs, abs)
		return nil
	}
	return errors.New("Unable to copy ")
}
func (e dummyEngine) Accessible(mem Memory) (Memory, error) { return mem, nil }
func (e dummyEngine) WorksWith(order DataOrder) bool        { return true }

func willerr(a *Dense, tc, eqtc *typeclass) (retVal, willFailEq bool) {
	if err := typeclassCheck(a.Dtype(), eqtc); err == nil {
		willFailEq = true
	}
	if err := typeclassCheck(a.Dtype(), tc); err != nil {
		return true, willFailEq
	}

	retVal = retVal || !a.IsNativelyAccessible()
	return
}

func qcErrCheck(t *testing.T, name string, a Dtyper, b interface{}, we bool, err error) (e error, retEarly bool) {
	switch {
	case !we && err != nil:
		t.Errorf("Tests for %v (%v) was unable to proceed: %v", name, a.Dtype(), err)
		return err, true
	case we && err == nil:
		if b == nil {
			t.Errorf("Expected error when performing %v on %T of %v ", name, a, a.Dtype())
			return errors.New("Error"), true
		}
		if bd, ok := b.(Dtyper); ok {
			t.Errorf("Expected error when performing %v on %T of %v and %T of %v", name, a, a.Dtype(), b, bd.Dtype())
		} else {
			t.Errorf("Expected error when performing %v on %T of %v and %v of %T", name, a, a.Dtype(), b, b)
		}
		return errors.New("Error"), true
	case we && err != nil:
		return nil, true
	}
	return nil, false
}

func qcIsFloat(dt Dtype) bool {
	if err := typeclassCheck(dt, floatcmplxTypes); err == nil {
		return true
	}
	return false
}

func qcEqCheck(t *testing.T, dt Dtype, willFailEq bool, correct, got interface{}) bool {
	isFloatTypes := qcIsFloat(dt)
	if !willFailEq && (isFloatTypes && !allClose(correct, got) || (!isFloatTypes && !reflect.DeepEqual(correct, got))) {
		t.Errorf("q.Dtype: %v", dt)
		t.Errorf("correct\n%v", correct)
		t.Errorf("got\n%v", got)
		return false
	}
	return true
}
