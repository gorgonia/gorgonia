package main

import (
	"fmt"
	"io"
	"text/template"
)

const testDDCmpOpTransitivityRaw = `func TestDense_{{lower .OpName}}DD_Transitivity(t *testing.T){
	{{$op := .OpName -}}
	{{range .Kinds -}}
		f{{short .}} := func(a, b, c *QCDense{{short .}}) bool {
			var axb, bxc, axc *Dense
			var err error
			if axb, err = a.{{lower $op}}DD(b.Dense); err != nil {
				t.Errorf("Test {{$op}} transitivity for {{.}} failed (axb) : %v ", err)
				return false
			}
			if bxc, err = b.{{lower $op}}DD(c.Dense); err != nil {
				t.Errorf("Test {{$op}} transitivity for {{.}} failed (bxc): %v", err)
				return false
			}
			if axc, err = a.{{lower $op}}DD(c.Dense); err != nil {
				t.Errorf("Test {{$op}} transitivity for {{.}} failed (axc): %v", err)
				return false
			}

			ab := axb.bools()
			bc := bxc.bools()
			ac := axc.bools()

			for i, vab := range ab {
				if vab && bc[i] {
					if !ac[i]{
						return false
					}
				}
			}
			return true
		}
		if err := quick.Check(f{{short .}}, nil); err != nil {
			t.Error(err)
		}
	{{end -}}
}
`

const testDDCmpOpFuncOptsRaw = `func Test_Dense_{{lower .OpName}}DD_funcOpts(t *testing.T){
	{{$op := .OpName -}}
	{{range .Kinds -}}
		f{{short .}} := func(a, b *QCDense{{short .}}) bool {
			var reuse, axb, ret *Dense
			var err error
			reuse = recycledDense({{asType . | title | strip}}, Shape{a.len()})
			if axb, err = a.{{lower $op}}DD(b.Dense, AsSameType()); err != nil {
				t.Errorf("Test {{$op}} reuse for {{.}} failed(axb): %v",err)
			}
			if ret, err = a.{{lower $op}}DD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
				t.Errorf("Test {{$op}} reuse for {{.}} failed: %v", err)
				return false
			}
			if ret != reuse {
				t.Errorf("Expected ret to be equal reuse")
				return false
			}
			if !reflect.DeepEqual(axb.Data(), ret.Data()){
				return false
			}
			{{if ne .String "bool" -}}
			if ret, err = a.{{lower $op}}DD(b.Dense, WithReuse(reuse)); err == nil {
				t.Error("Expected an error")
				return false
			}
			{{end -}}

			return true
		}
		if err := quick.Check(f{{short .}}, nil); err != nil{
			t.Error(err)
		}
	{{end -}}
}

`

var (
	testDDCmpOpTransitivity *template.Template
	testDDCmpOpFuncOpts     *template.Template
)

func init() {
	testDDCmpOpTransitivity = template.Must(template.New("cmpDD Transitivity").Funcs(funcs).Parse(testDDCmpOpTransitivityRaw))
	testDDCmpOpFuncOpts = template.Must(template.New("cmpDD funcopts").Funcs(funcs).Parse(testDDCmpOpFuncOptsRaw))
}

func denseCmpTests(f io.Writer, generic *ManyKinds) {
	for _, bo := range cmpBinOps {
		fmt.Fprintf(f, "/* %s */\n\n\n", bo.OpName)
		mk := &ManyKinds{filter(generic.Kinds, bo.is)}
		op := ArithBinOps{mk, bo.OpName, bo.OpSymb, false}
		testDDCmpOpTransitivity.Execute(f, op)
		fmt.Fprintln(f, "\n")
		testDDCmpOpFuncOpts.Execute(f, op)
		fmt.Fprintln(f, "\n")
	}
}
