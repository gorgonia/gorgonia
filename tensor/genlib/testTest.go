package main

import (
	"fmt"
	"io"
	"text/template"
)

const testHeaderRaw = `func randomBool() bool {
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
func closeenoughf64(a, b float64) bool {return tolerancef64(a, b, 1e-8)}
func closef64(a, b float64) bool      { return tolerancef64(a, b, 1e-14) }
func veryclosef64(a, b float64) bool  { return tolerancef64(a, b, 4e-16) }
func soclosef64(a, b, e float64) bool { return tolerancef64(a, b, e) }
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

func cClose(a, b complex128) bool {return cTolerance(a, b, 1e-14)}
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
				closeness=closef64
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
				if !math.IsInf(bt[i], 0){
					return false
				}
				continue
			}
			if !closeness(v, bt[i]){
				return false
			}
		}
		return true
	case []float32:
		closeness := closef32
		var ok bool
		if len(approxFn) > 0 {
			if closeness, ok = approxFn[0].(func(a, b float32) bool); !ok {
				closeness=closef32
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
				if !math32.IsInf(bt[i], 0){
					return false
				}
				continue
			}
			if !closeness(v, bt[i]){
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
				if !cmplx.IsInf(complex128(bt[i])){
					return false
				}
				continue
			}
			if !cSoclose(complex128(v), complex128(bt[i]),1e-5){
				return false
			}
		}
		return true
	case []complex128:
		bt := b.([]complex128)
		for i, v := range at{
			if cmplx.IsNaN(v) {
				if !cmplx.IsNaN(bt[i]) {
					return false
				}
				continue
			}
			if cmplx.IsInf(v) {
				if !cmplx.IsInf(bt[i]){
					return false
				}
				continue
			}
			if !cClose(v, bt[i]){
				return false
			}
		}
		return true
	default:
		return reflect.DeepEqual(a, b)
	}
}

func anyToFloat64s(x interface{}) (retVal []float64) {
	switch xt := x.(type) {
	{{range  .Kinds -}}
	{{if isNumber . -}}
	case []{{asType .}}:
		{{if eq .String "float64" -}}
			return xt
		{{else if eq .String "float32" -}}
			retVal = make([]float64, len(xt))
			for i, v := range xt {
				switch {
				case math32.IsNaN(v):
					retVal[i] = math.NaN()
				case math32.IsInf(v, 1):
					retVal[i] = math.Inf(1)
				case math32.IsInf(v, -1):
					retVal[i] = math.Inf(-1)
				default:
					retVal[i] = float64(v)
				}
			}
		{{else if eq .String "complex64" -}}
			retVal = make([]float64, len(xt))
			for i, v := range xt {
				switch {
				case cmplx.IsNaN(complex128(v)):
					retVal[i] = math.NaN()
				case cmplx.IsInf(complex128(v)):
					retVal[i] = math.Inf(1)
				default:
					retVal[i] = float64(real(v))
				}
			}
		{{else if eq .String "complex128" -}}
			retVal = make([]float64, len(xt))
			for i, v := range xt {
				switch {
				case cmplx.IsNaN(v):
					retVal[i] = math.NaN()
				case cmplx.IsInf(v):
					retVal[i] = math.Inf(1)
				default:
					retVal[i] = real(v)
				}
			}
		{{else -}}
			retVal = make([]float64, len(xt))
			for i, v := range xt {
				retVal[i]=  float64(v)
			}
		{{end -}}
		return
	{{end -}}
	{{end -}}
	}
	panic("Unreachable")
}
`

const testQCRaw = `type QCDense{{short .}} struct {
	*Dense 
}
func (*QCDense{{short .}}) Generate(r *rand.Rand, size int) reflect.Value {
	s := make([]{{asType .}}, size)
	for i := range s {
		{{if hasPrefix .String "uint" -}}
			s[i] = {{asType .}}(r.Uint32())
		{{else if hasPrefix .String "int" -}}
			s[i] = {{asType .}}(r.Int())
		{{else if eq .String "float64" -}}
			s[i] = r.Float64()
		{{else if eq .String "float32" -}}
			s[i] = r.Float32()
		{{else if eq .String "complex64" -}}
			s[i] = complex(r.Float32(), r.Float32())
		{{else if eq .String "complex128" -}}
			s[i] = complex(r.Float64(), r.Float64())
		{{else if eq .String "bool" -}}
			s[i] = randomBool()
		{{else if eq .String "string" -}}
			s[i] = randomString()
		{{else if eq .String "unsafe.Pointer" -}}
			s[i] = nil
		{{end -}}
	}
	d := recycledDense({{asType . | title | strip}}, Shape{size}, WithBacking(s))
	q := new(QCDense{{short .}})
	q.Dense = d
	return reflect.ValueOf(q)
}
`

var (
	testHeader *template.Template
	testQC     *template.Template
)

func init() {
	testQC = template.Must(template.New("testQCs").Funcs(funcs).Parse(testQCRaw))
	testHeader = template.Must(template.New("TestHeader").Funcs(funcs).Parse(testHeaderRaw))
}

func testtest(f io.Writer, generic *ManyKinds) {
	testHeader.Execute(f, generic)
	fmt.Fprint(f, "\n")
	for _, k := range generic.Kinds {
		if !isParameterized(k) {
			testQC.Execute(f, k)
			fmt.Fprint(f, "\n")
		}
	}
}
