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
		return errors.Errorf("lenMismatch", "{{.OpName}}", len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "{{.OpName}}", len(a), len(incr))
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

const incrBinOpTestRaw = `func Test_{{.Name}}_{{.OpName}}(t *testing.T){
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
	if err := a.{{.OpName}}(b); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i]{
			t.Errorf("{{.OpName}} is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}
}
`

var (
	incrBinOpTmpl     *template.Template
	incrBinOpTestTmpl *template.Template
)

func init() {
	incrBinOpTmpl = template.Must(template.New("IncrBinOp").Funcs(funcMap).Parse(incrBinOpRaw))
	incrBinOpTestTmpl = template.Must(template.New("IncrBinOpTest").Funcs(funcMap).Parse(incrBinOpTestRaw))
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
}

func generateNumbersIncrTests(f io.Writer, m []ArrayType) {
	for _, bo := range binOps {
		fmt.Fprintf(f, "/* %s */\n\n", bo.OpName)
		for _, v := range m {
			if v.isNumber {
				op := BinOp{v, bo.OpName, bo.OpSymb, bo.IsFunc}
				incrBinOpTmpl.Execute(f, op)
				fmt.Fprintf(f, "\n")
			}
		}
		fmt.Fprintf(f, "\n")
	}
}
