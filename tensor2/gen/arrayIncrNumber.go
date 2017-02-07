package main

import (
	"fmt"
	"io"
	"text/template"
)

const incrBinOpRaw = `func (a {{.Name}}) Incr{{.OpName}}(other, incrArr Number) (err error) {
	var b, incr []{{.Of}}
	if b, err = get{{title .Of}}s(other); err != nil {
		return errors.Wrapf(err, opFail, "Incr{{.OpName}}")
	}
	
	if incr, err = get{{title .Of}}s(incrArr); err != nil{
		return errors.Wrapf(err, opFail, "Incr{{.OpName}}")
	}

	if len(b) != len(a) {
		return errors.Errorf(lenMismatch,  len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch,  len(a), len(incr))
	}

	{{if ne .VecPkg "" -}} 
	{{.VecPkg}}.Incr{{.OpName}}([]{{.Of}}(a), b, incr)
	{{else -}}
		{{$scaleInv := hasPrefix .OpName "ScaleInv" -}}
		{{$div := hasPrefix .OpName "Div" -}}
		{{if or $scaleInv $div -}}var errs errorIndices{{end}}
		for i, v := range b {
			{{if or $scaleInv $div -}}
			if v == {{.Of}}(0) {
				errs = append(errs, i)
				a[i] = 0
				continue
			}

			{{end -}}
			{{if .IsFunc -}}
				incr[i] += {{.Of}}({{.OpSymb}} (float64(a[i]), float64(v)))
			{{else -}}
				incr[i] += a[i] {{.OpSymb}} v
			{{end -}}
		}

		{{if or $scaleInv $div -}}
			if errs != nil {
				return errs
			}
		{{end -}}
	{{end -}}
	return nil
}
`

const incrBinOpTestRaw = `func Test_{{.Name}}_Incr{{.OpName}}(t *testing.T){
	a, b, _, _ := prep{{.Name}}Test()
	incr := {{.Name}}{ {{.IncrTestData}} }

	correct := make({{.Name}}, len(a))
	for i, v := range a {
		correct[i] = {{if .IsFunc -}}
			{{.Of}}({{.OpSymb}}(float64(v), float64(b[i]))) + incr[i]
		{{else -}} 
			v {{.OpSymb}} b[i]  + incr[i]
		{{end -}}
	}

	// same type
	if err := a.Incr{{.OpName}}(b, incr); err != nil {
		t.Error(err)
	}

	for i, v := range incr {
		if v != correct[i]{
			t.Errorf("Incr{{.OpName}} is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}
}
`

const vsIncrBinOpRaw = `func (a {{.Name}}) Incr{{.OpName}}(other interface{}, incrArr Number) (err error) {
	var b {{.Of}}
	if b, err = get{{title .Of}}(other); err != nil{
		return errors.Wrapf(err, opFail, "Incr{{.OpName}}")
	}

	var incr []{{.Of}}
	if incr, err = get{{title .Of}}s(incrArr); err != nil{
		return errors.Wrapf(err, opFail, "Incr{{.OpName}}")
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch,  len(a), len(incr))
	}


	{{if ne .VecPkg "" -}}
	{{.VecPkg}}.Incr{{.OpName}}([]{{.Of}}(a), b, incr)
	{{else -}}
	{{$scaleInv := hasPrefix .OpName "ScaleInv" -}}
	{{$div := hasPrefix .OpName "Div" -}}
	{{if or $scaleInv $div -}}var errs errorIndices{{end}}
		for i, v := range a {
			{{if or $scaleInv $div -}}
			if v == {{.Of}}(0) {
				errs = append(errs, i)
				incr[i] = 0 
				continue
			}
			{{end -}}

			incr[i] += {{if hasSuffix .OpName "R" -}}
				{{if .IsFunc -}} 
					{{.Of}}({{.OpSymb}}(float64(b), float64(v)))
				{{else -}} 
					b {{.OpSymb}} v 
				{{end -}}
			{{else -}} 
				{{if .IsFunc -}} 
					{{.Of}}({{.OpSymb}}(float64(v), float64(b)))
				{{else -}} 
					v {{.OpSymb}} b
				{{end -}}
			{{end -}}
		}
		{{if or $scaleInv $div -}}
			if errs != nil {
				return errs
			}
		{{end -}}
	{{end -}}
	return nil
}

`

const vsIncrBinOpTestRaw = `func Test_{{.Name}}_Incr{{.OpName}}(t *testing.T){
	a, _, _, b := prep{{.Name}}Test()
	incr := {{.Name}}{ {{.IncrTestData}} }

	correct := make({{.Name}}, len(a))
	for i, v := range a {
		{{if hasPrefix .OpName "ScaleInv" -}}
			if v == {{.Of}}(0) {
				correct[i] = 0
				continue
			}

		{{end -}}
		correct[i] = {{if hasSuffix .OpName "R" -}}
			{{if .IsFunc -}} 
				{{.Of}}({{.OpSymb}}(float64(b), float64(v))) + incr[i]
			{{else -}} 
				b {{.OpSymb}} v  + incr[i]
			{{end -}}
		{{else -}} 
			{{if .IsFunc -}} 
				{{.Of}}({{.OpSymb}}(float64(v), float64(b))) + incr[i]
			{{else -}} 
				v {{.OpSymb}} b + incr[i]
			{{end -}}
		{{end -}}
	}

	{{if hasPrefix .OpName "ScaleInv" -}}
		{{if hasPrefix .Of "float" -}}
			if err := a.Incr{{.OpName}}(b, incr); err != nil{
				t.Fatal(err)
			}
		{{else -}}
			err := a.Incr{{.OpName}}(b, incr)
			if err == nil {
				t.Error("Expected error (division by zero)")
			}
			if _, ok := err.(errorIndices); !ok{
				t.Fatal(err)
			}
		{{end -}}
	{{else -}}
	if err := a.Incr{{.OpName}}(b, incr); err != nil {
		t.Fatal(err)
	}
	{{end -}}

	for i, v := range incr {
		{{if hasPrefix .Of "float" -}}
			// for floats we don't bother checking the incorrect stuff
			if v != correct[i] && i != 0 {
		{{else -}}
			if v != correct[i] {
		{{end -}}
			t.Errorf("Incr{{.OpName}} is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.{{.OpName}}({{if eq .Name "f64s"}}float32(2){{else}}2.0{{end}}); err == nil{
		t.Error("Expected an error when performing {{.OpName}} on a differing type")
	}
}

`

var (
	incrBinOpTmpl       *template.Template
	incrBinOpTestTmpl   *template.Template
	vsIncrBinOpTmpl     *template.Template
	vsIncrBinOpTestTmpl *template.Template
)

func init() {
	incrBinOpTmpl = template.Must(template.New("IncrBinOp").Funcs(funcMap).Parse(incrBinOpRaw))
	incrBinOpTestTmpl = template.Must(template.New("IncrBinOpTest").Funcs(funcMap).Parse(incrBinOpTestRaw))

	vsIncrBinOpTmpl = template.Must(template.New("VSIncrBinOp").Funcs(funcMap).Parse(vsIncrBinOpRaw))
	vsIncrBinOpTestTmpl = template.Must(template.New("VSIncrBinOpTest").Funcs(funcMap).Parse(vsIncrBinOpTestRaw))
}

func generateNumbersIncr(f io.Writer, m []ArrayType) {
	for _, bo := range binOps {
		fmt.Fprintf(f, "/* %s */ \n\n", bo.OpName)
		for _, v := range m {
			if v.isNumber {
				op := BinOp{v, bo.OpName, bo.OpSymb, bo.IsFunc}
				incrBinOpTmpl.Execute(f, op)
				fmt.Fprintln(f, "\n")
			}
		}
		fmt.Fprintln(f, "\n")
	}

	// vec-scalar
	for _, bo := range vecscalarOps {
		fmt.Fprintf(f, "/* %s */ \n\n", bo.OpName)
		for _, v := range m {
			if v.isNumber {
				op := BinOp{v, bo.OpName, bo.OpSymb, bo.IsFunc}
				vsIncrBinOpTmpl.Execute(f, op)
				fmt.Fprintln(f, "\n")
			}
		}
		fmt.Fprintln(f, "\n")
	}
}

func generateNumbersIncrTests(f io.Writer, m []ArrayType) {
	for _, bo := range binOps {
		fmt.Fprintf(f, "/* %s */\n\n", bo.OpName)
		for _, v := range m {
			if v.isNumber {
				op := BinOp{v, bo.OpName, bo.OpSymb, bo.IsFunc}
				op.IncrTestData = "100, 100, 100, 100, 100"
				incrBinOpTestTmpl.Execute(f, op)
				fmt.Fprintf(f, "\n")
			}
		}
		fmt.Fprintf(f, "\n")
	}

	// vec-scalar
	for _, bo := range vecscalarOps {
		fmt.Fprintf(f, "/* %s */ \n\n", bo.OpName)
		for _, v := range m {
			if v.isNumber {
				op := BinOp{v, bo.OpName, bo.OpSymb, bo.IsFunc}
				op.IncrTestData = "100, 100, 100, 100, 100"
				vsIncrBinOpTestTmpl.Execute(f, op)
				fmt.Fprintln(f, "\n")
			}
		}
		fmt.Fprintln(f, "\n")
	}
}
