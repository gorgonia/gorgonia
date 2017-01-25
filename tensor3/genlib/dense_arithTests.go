package main

import (
	"fmt"
	"io"
	"text/template"
)

const testBasicPropertiesRaw = `func Test{{.OpName}}BasicProperties(t *testing.T){

	{{$op := .OpName -}}
	{{$hasIden := .HasIdentity -}}
	{{$iden := .Identity -}}
	{{$isComm := .IsCommutative -}}
	{{$isAssoc := .IsAssociative -}}
	{{$isInv := .IsInv -}}
	{{$invOpName := .InvOpName -}}
	{{range .Kinds -}}
		{{if $hasIden -}}
			// identity
			iden{{short .}} := func(a *QCDense{{short .}}) bool {
				var ret, correct, identity *Dense
				identity = newDense({{asType . | title | strip}}, a.len())
				{{if ne $iden 0 -}}
					identity.Memset({{asType .}}({{$iden}}))
				{{end -}}
				correct = newDense({{asType . | title | strip}}, a.len())
				copyDense(correct, a.Dense)

				ret, _ = a.{{$op}}(identity)

				if !allClose(correct.Data(), ret.Data()){
					return false
				}
				return true
			}
			if err := quick.Check(iden{{short .}}, nil); err != nil {
				t.Errorf("Identity test for {{.}} failed %v",err)
			}
		{{end -}}
		{{if eq $op "Pow" -}}
			pow0{{short .}} := func(a *QCDense{{short .}}) bool {
				var ret, correct, zero *Dense 
				zero = newDense({{asType . | title | strip}}, a.len())
				correct = newDense({{asType . | title | strip}}, a.len())
				correct.Memset({{asType .}}(1))
				ret, _ = a.{{$op}}(zero)
				if !allClose(correct.Data(), ret.Data()){
					return false
				}
				return true
			}
			if err := quick.Check(pow0{{short .}}, nil); err != nil {
				t.Errorf("Pow 0 failed")
			}
		{{end -}}
		{{if $isComm -}}
			// commutativity
			comm{{short .}} := func(a, b *QCDense{{short .}}) bool {
				ret1, _ := a.{{$op}}(b.Dense)
				ret2, _ := b.{{$op}}(a.Dense)
				if !allClose(ret1.Data(), ret2.Data()){
					return false
				}
				return true
			}
			if err := quick.Check(comm{{short .}}, nil); err != nil {
				t.Errorf("Commutativity test for {{.}} failed %v",err)
			}
		{{end -}}
		{{if $isAssoc -}}
			// asociativity
			assoc{{short .}} := func(a, b, c *QCDense{{short .}}) bool {
				ret1, _ := a.{{$op}}(b.Dense)
				ret1, _ = ret1.{{$op}}(c.Dense)

				ret2, _ := b.{{$op}}(c.Dense)
				ret2, _ = a.{{$op}}(ret2)

				if !allClose(ret1.Data(), ret2.Data()){
					return false
				}
				return true
			}
			if err := quick.Check(assoc{{short .}}, nil); err != nil {
				t.Errorf("Associativity test for {{.}} failed %v",err)
			}
		{{end -}}
		{{if $isInv -}}
			// invertible property - reserved to test inverse functions
			inv{{short .}} := func(a, b *QCDense{{short .}}) bool {
				ret1, _ := a.{{$op}}(b.Dense)
				ret2, _ := ret1.{{$invOpName}}(b.Dense)
				if !allClose(ret2.Data(), a.Data()){
					t.Errorf("E: %v\nG: %v", a.Data(), ret2.Data())
					return false
				}
				return true
			}
			if err := quick.Check(inv{{short .}}, nil); err != nil {
				t.Errorf("Inverse function test for {{.}} failed %v",err)
			}
		{{end -}}
	{{end -}}
}
`

const testFuncOptRaw = `func Test{{.OpName}}FuncOpts(t *testing.T){
	var f func(*QCDenseF64) bool

	f = func(a *QCDenseF64) bool {
		identity := newDense(Float64, a.len()+1)
		{{if ne .Identity 0 -}}
			identity.Memset({{.Identity}})
		{{end -}}
		if _, err := a.{{.OpName}}(identity); err == nil {
			t.Error("Failed length mismatch test")
			return false
		}
		return true
	}
	if err := quick.Check(f, nil); err != nil {
		t.Errorf("len test for {{.OpName}} failed : %v ", err)
	}

	// safe
	f = func(a *QCDenseF64) bool {
		var identity, ret *Dense
		var err error
		identity = newDense(Float64, a.len())
		{{if ne .Identity 0 -}}
			identity.Memset({{.Identity}})
		{{end -}}
		if ret, err = a.{{.OpName}}(identity); err != nil {
			t.Error(err)
			return false
		}
		if ret == identity || ret == a.Dense {
			t.Errorf("Failed safe test for {{.OpName}}")
			return false
		}
		return true
	}
	if err := quick.Check(f, nil); err != nil {
		t.Errorf("safe test for {{.OpName}} failed : %v ", err)
	}

	// reuse
	f = func(a *QCDenseF64) bool {
		var identity, ret, correct, reuse *Dense
		var err error
		identity = newDense(Float64, a.len())
		{{if ne .Identity 0 -}}
			identity.Memset(float64({{.Identity}}))
		{{end -}}
		reuse = newDense(Float64, a.len())
		correct = newDense(Float64, a.len())
		copyDense(correct, a.Dense)
		if ret, err = a.{{.OpName}}(identity, WithReuse(reuse)); err != nil {
			t.Error(err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret %p == reuse %p", ret, reuse)
			return false
		}
		if !allClose(correct.Data(), ret.Data()){
			t.Errorf("Expected Reuse: %v\nGot reuse : %v", correct.Data(), ret.Data())
			return false
		}

		// wrong reuse type
		reuse = newDense(Bool, a.len())
		if _, err = a.{{.OpName}}(identity, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when doing {{.OpName}} using a reuse with a type mismatch")
			return false
		}

		// wrong reuse length
		reuse = newDense(Float64, a.len()+1)
		if _, err = a.{{.OpName}}(identity, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when doing {{.OpName}} using a reuse with a size mismatch")
			return false
		}
		return true
	}
	if err := quick.Check(f, nil); err != nil {
		t.Errorf("reuse test for {{.OpName}} failed : %v ", err)
	}

	// unsafe 
	f = func(a *QCDenseF64) bool {
		var identity, ret, correct *Dense
		var err error
		identity = newDense(Float64, a.len())
		{{if ne .Identity 0 -}}
			identity.Memset(float64({{.Identity}}))
		{{end -}}
		correct = newDense(Float64, a.len())
		copyDense(correct, a.Dense)

		if ret, err = a.{{.OpName}}(identity, UseUnsafe()) ; err != nil {
			t.Error(err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret == a")
			return false
		}
		if !allClose(correct.Data(), ret.Data()){
			return false
		}
		return true
	}
	if err := quick.Check(f, nil); err != nil {
		t.Errorf("unsafe test for {{.OpName}} failed : %v ", err)
	}
}
`

var (
	testBasicProperties *template.Template
	testFuncOpts        *template.Template
	testSubtraction     *template.Template
	testMultiplication  *template.Template
)

func init() {
	testBasicProperties = template.Must(template.New("testAdditionBasicProp").Funcs(funcs).Parse(testBasicPropertiesRaw))
	testFuncOpts = template.Must(template.New("testAdditionFuncOpt").Funcs(funcs).Parse(testFuncOptRaw))
}

func denseArithTests(f io.Writer, generic *ManyKinds) {
	numbers := filter(generic.Kinds, isNumber)
	mk := &ManyKinds{numbers}

	for _, bo := range binOps {
		fmt.Fprintf(f, "/* %s */\n\n", bo.OpName)
		op := ArithBinOps{
			BinOps: BinOps{
				ManyKinds: mk,
				OpName:    bo.OpName,
				OpSymb:    bo.OpSymb,
				IsFunc:    bo.IsFunc,
			},

			HasIdentity:   bo.HasIdentity,
			Identity:      bo.Identity,
			IsCommutative: bo.IsCommutative,
			IsAssociative: bo.IsAssociative,
			IsInv:         bo.IsInv,
			InvOpName:     bo.InvOpName,
			InvOpSymb:     bo.InvOpSymb,
		}
		testBasicProperties.Execute(f, op)
		testFuncOpts.Execute(f, bo)
	}
}
