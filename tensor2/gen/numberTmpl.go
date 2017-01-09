package main

import "text/template"

type BinOp struct {
	ArrayType
	OpName string
	OpSymb string
	IsFunc bool
}

var binOps = []struct {
	OpName string
	OpSymb string

	IsFunc bool
}{
	{"Add", "+", false},
	{"Sub", "-", false},
	{"Mul", "*", false},
	{"Div", "/", false},
	{"Pow", "math.Pow", true},
}

var vecscalarOps = []struct {
	OpName string
	OpSymb string

	IsFunc bool
}{
	{"Trans", "+", false},
	{"TransR", "-", false},
	{"Scale", "*", false},
	{"DivR", "/", false},
	{"PowOf", "math.Pow", true},
	{"PowOfR", "math.Pow", true},
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
		{{if .IsFunc}}a[i] = {{.Of}}({{.OpSymb}} (float64(a[i]), float64(v)))
		{{else}} a[i] {{.OpSymb}}= v 
		{{end}}
	}
	return nil{{end}}
}
`

const vecScalarOpRaw = `func (a {{.Name}}) {{.OpName}}(other interface{}) error {
	var b {{.Of}}
	var ok bool

	if b, ok = other.({{.Of}}); !ok{
		return errors.Errorf("Expected {{.Of}}. Got %T instead", other)
	}

	{{if ne .VecPkg ""}}{{.VecPkg}}.{{.OpName}}(b, []{{.Of}}(a))
	return nil
	{{else}}for i, v := range a {
		{{if .IsFunc}}a[i] = {{.Of}}({{.OpSymb}} (float64(v), float64(b)))
		{{else}} a[i] = {{if hasPrefix .OpName "R"}}  v {{.OpSymb}} b {{else}} b {{.OpSymb}} v {{end}}
		{{end}}
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
		correct[i] = {{if .IsFunc}} {{.Of}}({{.OpSymb}}(float64(v), float64(b[i]))) 
		{{else}} v {{.OpSymb}} b[i] 
		{{end}}
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

const vecScalarTestRaw = `func Test{{.Test}}_{{.OpName}}(t *testing.T){
	var a {{.Name}}
	var b {{.Of}}
	a = {{.Name}}{ {{.TestData0}} }
	b = 2

	correct := make({{.Name}}, len(a))
	for i, v := range a {
		correct[i] = {{if .IsFunc}}{{.Of}}({{.OpSymb}} (float64(a[i], b))) {{else}} a[i] {{.OpSymb}} b {{end}}
	}

	if err := a.{{.OpName}}(b); err != nil {
		t.Error(err)
	}

	for i, v := range a {
		if v != correct[i] {
			t.Errorf("{{.OpName}} is incorrect. Expected %v. Got %v", correct[i], v)
			break
		}
	}
}
`

var (
	binOpTmpl       *template.Template
	binOpTestTmpl   *template.Template
	vecScalarOpTmpl *template.Template
)

func init() {
	binOpTmpl = template.Must(template.New("BinOp").Parse(binOpRaw))
	binOpTestTmpl = template.Must(template.New("BinOpTest").Parse(binOpTestRaw))

	vecScalarOpTmpl = template.Must(template.New("vecScalarOp").Funcs(funcMap).Parse(vecScalarOpRaw))
}
