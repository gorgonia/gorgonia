package main

import (
	"io"
	"text/template"
)

const conversionsRaw = `func convFromFloat64s(to Dtype, data []float64) interface{} {
	switch to.Kind(){
	{{range .Kinds -}}
	{{if isNumber . -}}
	case reflect.{{reflectKind .}}:
		{{if eq .String "float64" -}}
			retVal := make([]float64, len(data))
			copy(retVal, data)
			return retVal
		{{else if eq .String "float32" -}}
			retVal := make([]float32, len(data))
			for i, v := range data {
				switch {
				case math.IsNaN(v):
					retVal[i] = math32.NaN()
				case math.IsInf(v, 1):
					retVal[i] = math32.Inf(1)
				case math.IsInf(v, -1):
					retVal[i] = math32.Inf(-1)
				default:
					retVal[i] = float32(v)
				}
			}
			return retVal
		{{else if eq .String "complex64" -}}
			retVal := make([]complex64, len(data))
			for i, v := range data {
				switch {
				case math.IsNaN(v):
					retVal[i] = complex64(cmplx.NaN())
				case math.IsInf(v, 0):
					retVal[i] = complex64(cmplx.Inf())
				default:
					retVal[i] = complex(float32(v), float32(0))
				}
			}
			return retVal
		{{else if eq .String "complex128" -}}
			retVal := make([]complex128, len(data))
			for i, v := range data {
				switch {
				case math.IsNaN(v):
					retVal[i] = cmplx.NaN()
				case math.IsInf(v, 0):
					retVal[i] = cmplx.Inf()
				default:
					retVal[i] = complex(v, float64(0))
				}
			}
			return retVal
		{{else -}}
			retVal := make([]{{asType .}}, len(data))
			for i, v :=range data{
				switch {
				case math.IsNaN(v), math.IsInf(v, 0):
					retVal[i] = 0
				default:
					retVal[i] = {{asType .}}(v)
				}
			}
			return retVal
		{{end -}}
	{{end -}}
	{{end -}}
	default:
		panic("Unsupported Dtype")
	}
}

func convToFloat64s(t *Dense) (retVal []float64){
	retVal = make([]float64, t.len())
	switch t.t.Kind() {
	{{range .Kinds -}}
	{{if isNumber . -}}
	case reflect.{{reflectKind .}}:
		{{if eq .String "float64" -}}
			return t.float64s()
		{{else if eq .String "float32" -}}
			for i, v := range t.float32s() {
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
			for i, v := range t.complex64s() {
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
			for i, v := range t.complex128s() {
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
			for i, v := range t.{{sliceOf .}} {
				retVal[i]=  float64(v)
			}
		{{end -}}
		return retVal
	{{end -}}
	{{end -}}
	default:
		panic(fmt.Sprintf("Cannot convert *Dense of %v to []float64", t.t))
	}
}

func convToFloat64(x interface{}) float64 {
	switch xt := x.(type) {
	{{range .Kinds -}}
	{{if isNumber . -}}
	case {{asType .}}:
		{{if eq .String "float64 -"}}
			return xt
		{{else if eq .String "complex64" -}}
			return float64(real(xt))
		{{else if eq .String "complex128" -}}
			return real(xt)
		{{else -}}
			return float64(xt)
		{{end -}}
	{{end -}}
	{{end -}}
	default:
		panic("Cannot convert to float64")
	}
}
`

const compatRaw = `// FromMat64 converts a *"gonum/matrix/mat64".Dense into a *tensorf64.Tensor.
func FromMat64(m *mat64.Dense, opts ...FuncOpt) *Dense {
	r, c := m.Dims()
	fo := ParseFuncOpts(opts...)
	toCopy := fo.Safe()
	as := fo.As()
	if as.Type == nil {
		as = Float64
	}

	switch as.Kind() {
	{{range .Kinds -}}
	{{if isNumber . -}}
	case reflect.{{reflectKind .}}:
		{{if eq .String "float64" -}}
			var backing []float64
			if toCopy {
				backing = make([]float64, len(m.RawMatrix().Data))
				copy(backing, m.RawMatrix().Data)
			} else {
				backing = m.RawMatrix().Data
			}
		{{else -}}
			backing := convFromFloat64s({{asType . | title}}, m.RawMatrix().Data).([]{{asType .}})
		{{end -}}
		retVal := New(WithBacking(backing), WithShape(r, c))
		return retVal
	{{end -}}
	{{end -}}
	default:
		panic(fmt.Sprintf("Unsupported Dtype - cannot convert float64 to %v", as))
	}
	panic("Unreachable")
}


// ToMat64 converts a *Dense to a *mat64.Dense. All the values are converted into float64s.
// This function will only convert matrices. Anything *Dense with dimensions larger than 2 will cause an error.
func ToMat64(t *Dense, opts ...FuncOpt) (retVal *mat64.Dense, err error) {
	// checks:
	if !t.IsMatrix() {
		// error
		err = errors.Errorf("Cannot convert *Dense to *mat64.Dense. Expected number of dimensions: <=2, T has got %d dimensions (Shape: %v)", t.Dims(), t.Shape())
		return
	}

	fo := ParseFuncOpts(opts...)
	toCopy := fo.Safe()

	// fix dims
	r := t.Shape()[0]
	c := t.Shape()[1]

	var data []float64
	switch {
	case t.t.Kind() == reflect.Float64 && toCopy  && !t.IsMaterializable():
		data = make([]float64, t.len())
		copy(data, t.float64s())
	case !t.IsMaterializable():	
		data = convToFloat64s(t)
	default:
		it := NewFlatIterator(t.AP)
		var next int
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			if err = handleNoOp(err); err != nil {
				return
			}
			data = append(data, convToFloat64(t.Get(next)))
		}
		err = nil
		
	}

	retVal = mat64.NewDense(r, c, data)
	return
}
`

var (
	conversions *template.Template
	compats     *template.Template
)

func init() {
	conversions = template.Must(template.New("conversions").Funcs(funcs).Parse(conversionsRaw))
	compats = template.Must(template.New("compat").Funcs(funcs).Parse(compatRaw))
}

func compat(f io.Writer, generic *ManyKinds) {
	conversions.Execute(f, generic)
	compats.Execute(f, generic)
}
