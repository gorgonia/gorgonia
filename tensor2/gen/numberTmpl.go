package main

import "text/template"

type BinOp struct {
	ArrayType
	OpName string
	OpSymb string
}

var binOps = []struct {
	OpName string
	OpSymb string
}{
	{"Add", "+"},
	{"Sub", "-"},
	{"Mul", "*"},
	{"Div", "/"},
}

const binOpRaw = `func (a {{.Name}}) {{.OpName}}(other Number) error {
	var b []{{.Of}}

	switch ot:= other.(type) {
	case {{.Name}}:
		b = []{{.Of}}(ot)
	case {{.Compatible}}er:
		b = ot.{{.Compatible}}()
	default:
		return errors.Errorf(typeMismatch, "{{.OpName}}", a, other)
	}

	if len(a) != len(b){
		return errors.Errorf("lenMismatch", "{{.OpName}}", len(a), len(b))
	}


	{{if ne .VecPkg ""}}{{.VecPkg}}.{{.OpName}}([]{{.Of}}(a), b)
	return nil
	{{else}}for i, v := range b {
		a[i] {{.OpSymb}}= v
	}
	return nil{{end}}
}
`
const binOpTestRaw = `func Test_{{.Name}}_{{.OpName}}(t *testing.T){
	var a, b {{.Name}}
	var c {{.Name}}Dummy

	a = {{.Name}}{ {{.TestData0}} }
	b = {{.Name}}{ {{.TestData1}} }
	c = {{.Name}}Dummy{ {{.TestData1}} }

	correct := make({{.Name}}, len(a))
	for i, v := range a {
		correct[i] = v {{.OpSymb}} b[i]
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

	// compatible type
	a = {{.Name}}{ {{.TestData0}} }
	if err := a.{{.OpName}}(c); err != nil{
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i]{
			t.Errorf("{{.OpName}} is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}

	// idiotsville 1
	if err := a.{{.OpName}}(b[:3]); err == nil {
		t.Error("Expected an error when performing {{.OpName}} on differing lengths")
	}

	// idiotsville 2
	{{if eq .Name "f64s"}}if err := a.{{.OpName}}(f32s{}); err == nil {{else}}if err := a.{{.OpName}}(f64s{}); err == nil {{end}} {
		t.Errorf("Expected an error when performing {{.OpName}} on a non-compatible type")
	}
	
}
`

var (
	binOpTmpl *template.Template

	binOpTestTmpl *template.Template
)

func init() {
	binOpTmpl = template.Must(template.New("BinOp").Parse(binOpRaw))
	binOpTestTmpl = template.Must(template.New("BinOpTest").Parse(binOpTestRaw))
}
