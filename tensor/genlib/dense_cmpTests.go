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

		fIter{{short .}} := func(a, b, c *QCDense{{short .}}) bool {
			var axb, bxc, axc *Dense
			var a1, b1, c1 *Dense // sliced
			var a2, b2, c2 *Dense // materialized slice

			var abb, bcb, acb []bool
			{{if isNumber . -}}
			var abs, bcs, acs []{{asType .}}
			{{end -}}

			// set up
			a1, _ = sliceDense(a.Dense, makeRS(0, 5))
			a2 = a1.Materialize().(*Dense)
			b1, _ = sliceDense(a.Dense, makeRS(0, 5))
			b2 = b1.Materialize().(*Dense)
			c1, _ = sliceDense(c.Dense, makeRS(0, 5))
			c2 = c1.Materialize().(*Dense)

			// a iter {{if isNumber .}}bools{{end}}
			axb, _ = a1.{{lower $op}}DD(b2)
			bxc, _ = b1.{{lower $op}}DD(c2)
			axc, _ = a1.{{lower $op}}DD(c2)

			abb = axb.bools()
			bcb = bxc.bools()
			acb = axc.bools()

			for i, vab := range abb {
				if vab && bcb[i] {
					if !acb[i] {
						return false
					}
				}
			}

			{{if isNumber . -}}
			// a iter asSame
			axb, _ = a1.{{lower $op}}DD(b2, AsSameType())
			bxc, _ = b1.{{lower $op}}DD(c2, AsSameType())
			axc, _ = a1.{{lower $op}}DD(c2, AsSameType())

			abs = axb.{{sliceOf .}}
			bcs = bxc.{{sliceOf .}}
			acs = axc.{{sliceOf .}}

			for i, vab := range abs {
				if vab == 1 && bcs[i]==1 {
					if acs[i] != 1 {
						return false
					}
				}
			}
			{{end -}}

			// b iter {{if isNumber .}}bools{{end}}
			axb, _ = a2.{{lower $op}}DD(b1)
			bxc, _ = b2.{{lower $op}}DD(c1)
			axc, _ = a2.{{lower $op}}DD(c1)

			abb = axb.bools()
			bcb = bxc.bools()
			acb = axc.bools()

			for i, vab := range abb {
				if vab && bcb[i] {
					if !acb[i] {
						return false
					}
				}
			}

			{{if isNumber . -}}
			// a iter asSame
			axb, _ = a2.{{lower $op}}DD(b1, AsSameType())
			bxc, _ = b2.{{lower $op}}DD(c1, AsSameType())
			axc, _ = a2.{{lower $op}}DD(c1, AsSameType())

			abs = axb.{{sliceOf .}}
			bcs = bxc.{{sliceOf .}}
			acs = axc.{{sliceOf .}}

			for i, vab := range abs {
				if vab == 1 && bcs[i]==1 {
					if acs[i] != 1 {
						return false
					}
				}
			}
			{{end -}}

			// both a and b iter {{if isNumber .}}bools{{end}}
			axb, _ = a1.{{lower $op}}DD(b1)
			bxc, _ = b1.{{lower $op}}DD(c1)
			axc, _ = a1.{{lower $op}}DD(c1)

			abb = axb.bools()
			bcb = bxc.bools()
			acb = axc.bools()

			for i, vab := range abb {
				if vab && bcb[i] {
					if !acb[i] {
						return false
					}
				}
			}

			{{if isNumber . -}}
			// a iter asSame
			axb, _ = a1.{{lower $op}}DD(b1, AsSameType())
			bxc, _ = b1.{{lower $op}}DD(c1, AsSameType())
			axc, _ = a1.{{lower $op}}DD(c1, AsSameType())

			abs = axb.{{sliceOf .}}
			bcs = bxc.{{sliceOf .}}
			acs = axc.{{sliceOf .}}

			for i, vab := range abs {
				if vab == 1 && bcs[i]==1 {
					if acs[i] != 1 {
						return false
					}
				}
			}
			{{end -}}

			return true
		}
		if err := quick.Check(fIter{{short .}}, nil); err != nil {
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

			reuse  = recycledDense(Bool, Shape{a.len()})
			if axb, err = a.{{lower $op}}DD(b.Dense); err != nil {
				t.Errorf("Test {{$op}} reuse for {{.}} failed(axb): %v",err)
			}
			if ret, err = a.{{lower $op}}DD(b.Dense, WithReuse(reuse)); err != nil {
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


			{{if isNumber . -}}
			// reuse as same type
			reuse = recycledDense({{asType . | title | strip}}, Shape{a.len()})
			if axb, err = a.{{lower $op}}DD(b.Dense, AsSameType()); err != nil {
				t.Errorf("Test {{$op}} as same type reuse for {{.}} failed(axb): %v",err)
			}
			if ret, err = a.{{lower $op}}DD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
				t.Errorf("Test {{$op}} as same type reuse for {{.}} failed: %v", err)
				return false
			}
			if ret != reuse {
				t.Errorf("Expected ret to be equal reuse")
				return false
			}
			if !reflect.DeepEqual(axb.Data(), ret.Data()){
				return false
			}
			if ret, err = a.{{lower $op}}DD(b.Dense, WithReuse(reuse)); err == nil {
				t.Error("Expected an error")
				return false
			}

			// unsafe
			if ret, err = a.{{lower $op}}DD(b.Dense, UseUnsafe()); err != nil{
				t.Errorf("Unsafe {{$op}} for {{.}} failed %v", err)
				return false
			}
			if ret != a.Dense {
				t.Error("Expected ret to be equal to a")
			}
			if !reflect.DeepEqual(axb.Data(), ret.Data()){
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
		op := BinOps{mk, bo.OpName, bo.OpSymb, false}
		testDDCmpOpTransitivity.Execute(f, op)
		fmt.Fprintln(f, "\n")
		testDDCmpOpFuncOpts.Execute(f, op)
		fmt.Fprintln(f, "\n")
	}
}
