package main

import (
	"fmt"
	"io"
	"text/template"
)

type ElOrdBinOp struct {
	ArrayType
	OpName    string
	OpSymb    string
	TypeClass string
}

var eleqordBinOps = []struct {
	OpName    string
	OpSymb    string
	TypeClass string
}{
	{"ElEq", "==", "ElemEq"},
	{"Gt", ">", "ElemOrd"},
	{"Gte", ">=", "ElemOrd"},
	{"Lt", "<", "ElemOrd"},
	{"Lte", "<=", "ElemOrd"},
}

const eleqordRaw = `func (a {{.Name}}) {{.OpName}}(other {{.TypeClass}}, same bool) (Array, error) {
	var compat {{.Compatible}}er
	var ok bool
	if compat, ok = other.({{.Compatible}}er); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.{{.Compatible}}()
	
	if len(a) != len(b){
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make({{.Name}}, len(a))
		for i, v := range a {
			if v {{.OpSymb}} b[i] {
				retVal[i] = {{.DefaultOne}}
			}
		}

		return retVal, nil
	}

	retVal := make(bs,  len(a)) 
	for i, v := range a{
		retVal[i] = v {{.OpSymb}} b[i]
	}
	return retVal, nil
}
`

const eleqordTestRaw = `func Test_{{.Name}}_{{.OpName}}(t *testing.T) {
	var a, b {{.Name}}
	var c {{.Name}}Dummy
	var res Array
	var err error

	a = {{.Name}}{ {{.TestData0}} }
	b = {{.Name}}{ {{.TestData1}} }
	c = {{.Name}}Dummy{ {{.TestData1}} }

	correct := make(bs, len(a))
	correctSame := make({{.Name}}, len(a))
	for i, v := range a {
		correct[i] = v {{.OpSymb}} b[i]

		if v {{.OpSymb}} b[i] {
			correctSame[i] = {{.DefaultOne}}
		} else {
			correctSame[i] = {{.DefaultZero}}
		}
	}

	// return bools
	if res, err = a.{{.OpName}}(b, false); err != nil {
		t.Error(err)
	}

	for i , v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("{{.OpName}} is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = {{.Name}}{ {{.TestData0}} }
	if res, err = a.{{.OpName}}(b, true); err != nil {
		t.Error(err)
	}

	for i , v := range res.({{.Name}}) {
		if v != correctSame[i] {
			t.Errorf("{{.OpName}} is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// now for compatible types

	// return bools
	a = {{.Name}}{ {{.TestData0}} }
	if res, err = a.{{.OpName}}(c, false); err != nil {
		t.Error(err)
	}

	for i , v := range res.(bs) {
		if v != correct[i] {
			t.Errorf("{{.OpName}} is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}

	// return same
	a = {{.Name}}{ {{.TestData0}} }
	if res, err = a.{{.OpName}}(c, true); err != nil {
		t.Error(err)
	}

	for i , v := range res.({{.Name}}) {
		if v != correctSame[i] {
			t.Errorf("{{.OpName}} is incorrect when returning bools. i: %d Expected %v, Got %v", i, correct[i], v)
			break
		}
	}


	// stupids # 1 : differing length
	if _,  err := a.{{.OpName}}(b[:3], true); err == nil{
		t.Errorf("Expected an error when performing {{.OpName}} on differing lengths")
	}

	// stupids #2 : different types (which are checked before lengths)
	{{if eq .Name "f64s"}}if _, err := a.{{.OpName}}(f32s{}, true); err == nil {{else}}if _, err:= a.{{.OpName}}(f64s{}, true); err == nil {{end}}{
		t.Errorf("Expected an error when performing {{.OpName}} on a non-compatible type")
	}
}

`

var (
	eleqordTmpl     *template.Template
	eleqordTestTmpl *template.Template
)

func init() {
	eleqordTmpl = template.Must(template.New("ElEqOrd").Parse(eleqordRaw))
	eleqordTestTmpl = template.Must(template.New("ElEqOrdTest").Parse(eleqordTestRaw))
}

func generateElEqOrds(f io.Writer, m []ArrayType) {
	for _, bo := range eleqordBinOps {
		fmt.Fprintf(f, "/* %s */\n\n", bo.OpName)
		for _, v := range m {
			if bo.OpName == "ElEq" || (bo.OpName != "ElEq" && v.elOrd) {
				op := ElOrdBinOp{v, bo.OpName, bo.OpSymb, bo.TypeClass}
				eleqordTmpl.Execute(f, op)
				fmt.Fprintf(f, "\n")
			}
		}
		fmt.Fprintf(f, "\n")
	}
}

func generateElEqOrdsTests(f io.Writer, m []ArrayType) {
	for _, bo := range eleqordBinOps {
		fmt.Fprintf(f, "/* %s */\n\n", bo.OpName)
		for _, v := range m {
			if bo.OpName == "ElEq" || (bo.OpName != "ElEq" && v.elOrd) {
				op := ElOrdBinOp{v, bo.OpName, bo.OpSymb, bo.TypeClass}
				eleqordTestTmpl.Execute(f, op)
				fmt.Fprintf(f, "\n")
			}
		}
		fmt.Fprintf(f, "\n")
	}
}
