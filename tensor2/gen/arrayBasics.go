package main

import (
	"fmt"
	"io"
	"text/template"
)

const lenRaw = `func (a {{.Name}}) Len() int {return len(a)}`
const capRaw = `func (a {{.Name}}) Cap() int {return cap(a)}`
const dataRaw = `func (a {{.Name}}) Data() interface{} {return []{{.Of}}(a) }`
const getRaw = `func (a {{.Name}}) Get(i int) interface{} {return a[i]}`

const setRaw = `func (a {{.Name}}) Set(i int, v interface{}) error {
	if f, ok := v.({{.Of}}); ok {
		a[i] = f
		return nil
	}
	return errors.Errorf("Cannot set %v of %T to []{{.Of}}", v, v)
}
`

const eqRaw = `func (a {{.Name}}) Eq(other interface{}) bool {
	if b, ok := other.({{.Name}}); ok {
		if len(a) != len(b) {
			return false
		}

		for i, v := range a {
			if v != b[i] {
				return false
			}
		}
		return true
	}

	if b, ok := other.([]{{.Of}}); ok {
		if len(a) != len(b){
			return false
		}
		for i, v := range a {
			if v != b[i] {
				return false
			}
		}
		return true
	}
	return false
}
`

const zeroRaw = `func (a {{.Name}}) Zero() {
	for i := range a {
		a[i] = {{.DefaultZero}}
	}
}
`

const oneRaw = `func (a {{.Name}}) One() {
	for i := range a {
		a[i] = {{.DefaultOne}}
	}
}
`

const copyFromRaw = `func (a {{.Name}}) CopyFrom(other interface{}) (int, error){
	switch b := other.(type) {
	case []{{.Of}}:
		return copy(a, b), nil
	case {{.Compatible}}er:
		return copy(a, b.{{.Compatible}}()), nil
	}

	return 0, errors.Errorf("Cannot copy from %T", other)
}
`

const compatibleRaw = `func (a {{.Name}}) {{.Compatible}}() []{{.Of}}{ return []{{.Of}}(a)}`

const arrayMapRaw = `func (a {{.Name}}) Map(fn interface{}) error{
	if f, ok := fn.(func({{.Of}}){{.Of}}); ok{
		for i, v := range a {
			a[i] = f(v)
		}
		return nil
	}
	return errors.Errorf(extractionFail, "func(x {{.Of}}){{.Of}}", fn)
}
`

const arrayMapTestRaw = `func inv{{title .Of}}(x {{.Of}}) {{.Of}} {	{{if eq .Of "bool" -}}
	return !x
	{{else -}}
	return -x
	{{end -}}
}

func Test_{{.Name}}_Map(t *testing.T) {
	a := {{.Name}}{ {{.TestData0}} }
	b := {{.Name}}{ {{.TestData0}} }
	if err := a.Map(inv{{title .Of}}); err != nil {
		t.Error(err)
	}

	for i, v :=range a {
		if v != inv{{title .Of}}(b[i]) {
			t.Fatal("inverse function not applied")
		}
	}

	{{if eq .Of "float64" -}}
		if err := a.Map(invFloat32); err == nil{
			t.Error("Expected an error!")
		}
	{{else -}}
		if err := a.Map(invFloat64); err == nil{
			t.Error("Expected an error!")
		}
	{{end -}}
}

`

const extractionHelpersRaw = `func get{{title .Of}}s(a Array) ([]{{.Of}}, error){
	if at, ok := a.({{.Compatible}}er); ok {
		return at.{{.Compatible}}(), nil
	}
	return nil, errors.Errorf(extractionFail, "[]{{.Of}}", a)
}

func get{{title .Of}}(a interface{}) (retVal {{.Of}}, err error) {
	if b, ok := a.({{.Of}}); ok{
		return b, nil
	}
	err = errors.Errorf(extractionFail, "{{.Of}}", a)
	return
}

`

var (
	lenTmpl      *template.Template
	capTmpl      *template.Template
	dataTmpl     *template.Template
	getTmpl      *template.Template
	setTmpl      *template.Template
	arrayMapTmpl *template.Template
	eqTmpl       *template.Template
	zeroTmpl     *template.Template
	oneTmpl      *template.Template
	copyFromTmpl *template.Template

	compatibleTmpl *template.Template

	extractionHelpersTmpl *template.Template

	// tests
	arrayMapTestTmpl *template.Template

	basics      []*template.Template
	basicsTests []*template.Template
)

func init() {
	lenTmpl = template.Must(template.New("Len").Parse(lenRaw))
	capTmpl = template.Must(template.New("Cap").Parse(capRaw))
	dataTmpl = template.Must(template.New("Data").Parse(dataRaw))
	getTmpl = template.Must(template.New("Get").Parse(getRaw))
	setTmpl = template.Must(template.New("Set").Parse(setRaw))
	arrayMapTmpl = template.Must(template.New("Map").Funcs(funcMap).Parse(arrayMapRaw))
	eqTmpl = template.Must(template.New("Eq").Parse(eqRaw))
	zeroTmpl = template.Must(template.New("Zeror").Parse(zeroRaw))
	oneTmpl = template.Must(template.New("Oner").Parse(oneRaw))
	copyFromTmpl = template.Must(template.New("CopierFrom").Parse(copyFromRaw))

	compatibleTmpl = template.Must(template.New("Compat").Parse(compatibleRaw))

	extractionHelpersTmpl = template.Must(template.New("ExtractionHelper").Funcs(funcMap).Parse(extractionHelpersRaw))

	// tests
	arrayMapTestTmpl = template.Must(template.New("MapTest").Funcs(funcMap).Parse(arrayMapTestRaw))

	basics = []*template.Template{lenTmpl, capTmpl, compatibleTmpl, dataTmpl, getTmpl, setTmpl, arrayMapTmpl, eqTmpl, zeroTmpl, oneTmpl, copyFromTmpl}
	basicsTests = []*template.Template{arrayMapTestTmpl}
}

func generateImpl(f io.Writer, m []ArrayType) {
	generateHelpers(f, m)
	generateBasics(f, m)
}

func generateHelpers(f io.Writer, m []ArrayType) {
	fmt.Fprintf(f, "/* extraction functions */\n")
	for _, v := range m {
		extractionHelpersTmpl.Execute(f, v)
		fmt.Fprint(f, "\n")
	}
	fmt.Fprint(f, "\n")
}

func generateBasics(f io.Writer, m []ArrayType) {
	for _, tmpl := range basics {
		fmt.Fprintf(f, "/* %s */\n\n", tmpl.Name())
		for _, v := range m {
			tmpl.Execute(f, v)
			fmt.Fprint(f, "\n")
		}
		fmt.Fprint(f, "\n")
	}

	fmt.Fprintln(f, "/* Transpose Specialization */\n")
	for _, v := range m {
		transposeTmpl.Execute(f, v)
		fmt.Fprint(f, "\n")
	}

	fmt.Fprintln(f, "/* IncrMapper specialization */\n")
	for _, v := range m {
		if v.isNumber {
			incrMapperTmpl.Execute(f, v)
			fmt.Fprint(f, "\n")
		}
	}

	fmt.Fprintln(f, "/* IterMapper specialization */\n")
	for _, v := range m {
		iterMapperTmpl.Execute(f, v)
		fmt.Fprint(f, "\n")
	}

}

func generateBasicsTest(f io.Writer, m []ArrayType) {
	for _, tmpl := range basicsTests {
		fmt.Fprintf(f, "/* %s */\n\n", tmpl.Name())
		for _, v := range m {
			tmpl.Execute(f, v)
			fmt.Fprint(f, "\n")
		}
		fmt.Fprint(f, "\n")
	}
}
