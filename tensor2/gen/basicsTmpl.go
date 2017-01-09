package main

import "text/template"

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
	case {{.Name}}:
		return copy(a, b), nil
	case []{{.Of}}:
		return copy(a, b), nil
	case {{.Compatible}}er:
		return copy(a, b.{{.Compatible}}()), nil
	}

	return 0, errors.Errorf("Cannot copy from %T", other)
}
`

const compatibleRaw = `func (a {{.Name}}) {{.Compatible}}() []{{.Of}}{ return []{{.Of}}(a)}`

var (
	lenTmpl      *template.Template
	capTmpl      *template.Template
	dataTmpl     *template.Template
	getTmpl      *template.Template
	setTmpl      *template.Template
	eqTmpl       *template.Template
	zeroTmpl     *template.Template
	oneTmpl      *template.Template
	copyFromTmpl *template.Template

	compatibleTmpl *template.Template

	basics []*template.Template
)

func init() {
	lenTmpl = template.Must(template.New("Len").Parse(lenRaw))
	capTmpl = template.Must(template.New("Cap").Parse(capRaw))
	dataTmpl = template.Must(template.New("Data").Parse(dataRaw))
	getTmpl = template.Must(template.New("Get").Parse(getRaw))
	setTmpl = template.Must(template.New("Set").Parse(setRaw))
	eqTmpl = template.Must(template.New("Eq").Parse(eqRaw))
	zeroTmpl = template.Must(template.New("Zeror").Parse(zeroRaw))
	oneTmpl = template.Must(template.New("Oner").Parse(oneRaw))
	copyFromTmpl = template.Must(template.New("CopierFrom").Parse(copyFromRaw))

	compatibleTmpl = template.Must(template.New("Compat").Parse(compatibleRaw))

	basics = []*template.Template{lenTmpl, capTmpl, dataTmpl, getTmpl, setTmpl, eqTmpl, zeroTmpl, oneTmpl, copyFromTmpl}
}
