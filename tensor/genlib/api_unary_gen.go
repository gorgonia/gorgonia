package main

import (
	"fmt"
	"io"
	"text/template"
)

const clampRaw = `// Clamp clamps the values of the Tensor to the min and max provided. The min and max provided must be the same type as the Tensor type.
// Incr is not supported (it doesn't make sense anyway)
func Clamp(a Tensor, minVal, maxVal interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	switch t := a.(type) {
	case *Dense:
		if t.IsMaterializable() {
			var f interface{}
			switch t.t.Kind() {
			{{range .Kinds -}}
			{{if isNumber . -}}
			{{if isOrd . -}}
			case reflect.{{reflectKind .}}:
				var min, max {{asType .}}
				var ok bool
				if min, ok = minVal.({{asType .}}); !ok {
					err = errors.Wrapf(errors.Errorf(typeMismatch, min, minVal), "Clamp() min")
					return
				}
				if max, ok = maxVal.({{asType .}}); !ok {
					err = errors.Wrapf(errors.Errorf(typeMismatch, max, maxVal), "Clamp() max")
					return
				}
				f = func(x {{asType .}}) {{asType .}}{
					if x < min {{if eq .String "float64"}}|| math.IsInf(x, -1){{else if eq .String "float32"}}|| math32.IsInf(x, -1){{end}} {
						return min
					}
					if x > max {{if eq .String "float64"}}|| math.IsInf(x, 1){{else if eq .String "float32"}}|| math32.IsInf(x, 1){{end}} {
						return max
					}
					return x
				}
			{{end -}}
			{{end -}}
			{{end -}}
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
		{{range .Kinds -}}
		{{if isNumber . -}}
		{{if isOrd . -}}
		case reflect.{{reflectKind .}}:
			var min, max {{asType .}}
			var ok bool
			if min, ok = minVal.({{asType .}}); !ok {
				err = errors.Wrapf(errors.Errorf(typeMismatch, min, minVal), "Clamp() min")
				return
			}
			if max, ok = maxVal.({{asType .}}); !ok {
				err = errors.Wrapf(errors.Errorf(typeMismatch, max, maxVal), "Clamp() max")
				return
			}
			data := ret.{{sliceOf .}}

			if !ret.IsMasked(){
				for i, v := range data {					
					if v < min {{if eq .String "float64"}}|| math.IsInf(v, -1){{else if eq .String "float32"}}|| math32.IsInf(v, -1){{end}} {
						data[i] = min
						continue
					}
					if v > max {{if eq .String "float64"}}|| math.IsInf(v, 1){{else if eq .String "float32"}}|| math32.IsInf(v, 1){{end}} {
						data[i] = max
					}
				}
			}	else	{					
				for i, v := range data {
					if !ret.mask[i]{
							if v < min {{if eq .String "float64"}}|| math.IsInf(v, -1){{else if eq .String "float32"}}|| math32.IsInf(v, -1){{end}} {
								data[i] = min
								continue
							}
							if v > max {{if eq .String "float64"}}|| math.IsInf(v, 1){{else if eq .String "float32"}}|| math32.IsInf(v, 1){{end}} {
								data[i] = max
							}
						}
					}
				}
		{{end -}}
		{{end -}}
		{{end -}}
		}
		retVal = ret
		return
	default:
		return nil, errors.Errorf(typeNYI, "Clamp", a)
	}
}

`

const signRaw = `// Sign returns the sign function as applied to each element in the ndarray. It does not yet support the incr option.
// Incr is not supported (it doesn't make sense anyway)
func Sign(a Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	switch t := a.(type) {
	case *Dense:
		if t.IsMaterializable() {
			var f interface{}
			switch t.t.Kind() {
			{{range .Kinds -}}
			{{$isInt := hasPrefix .String "int" -}}
			{{$isFloat := hasPrefix .String "float" -}}
			{{if isNumber . -}}
			{{if or $isInt $isFloat -}}
			case reflect.{{reflectKind .}}:
				f = func(x {{asType .}}) {{asType .}}{
					if x < 0 {
						return -1
					}
					if x > 0 {
						return 1
					}
					return 0
				}
			{{end -}}
			{{end -}}
			{{end -}}
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
		{{range .Kinds -}}
		{{$isInt := hasPrefix .String "int" -}}
		{{$isFloat := hasPrefix .String "float" -}}
		{{if isNumber . -}}
		{{if or $isInt $isFloat -}}
		case reflect.{{reflectKind .}}:
			data := ret.{{sliceOf .}}
			if !ret.IsMasked(){
				for i, v := range data {
					if v < 0 {
						data[i] = -1
						continue
					}
					if v > 0 {
						data[i] = 1
					}
				}
			} else {
				for i, v := range data {
					if !ret.mask[i]{
						if v < 0 {
							data[i] = -1
							continue
						}
						if v > 0 {
							data[i] = 1
						}
					}
				}
			}

		{{end -}}
		{{end -}}
		{{end -}}
		}
		retVal = ret
		return
	default:
		return nil, errors.Errorf(typeNYI, "Clamp", a)
	}
}
`

var (
	clamp *template.Template
	sign  *template.Template
)

func init() {
	clamp = template.Must(template.New("clamp").Funcs(funcs).Parse(clampRaw))
	sign = template.Must(template.New("sign").Funcs(funcs).Parse(signRaw))
}

func generateUnaryAPIFuncs(f io.Writer, generic *ManyKinds) {
	clamp.Execute(f, generic)
	fmt.Fprint(f, "\n")
	sign.Execute(f, generic)
}
