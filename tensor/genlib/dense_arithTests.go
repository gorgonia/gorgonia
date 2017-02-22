package main

import (
	"fmt"
	"io"
	"text/template"
)

const testDDBasicPropertiesRaw = `func Test{{.OpName}}BasicProperties(t *testing.T){
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

			idenSliced{{short .}} := func(a *QCDense{{short .}}) bool {
				var ret, correct, identity *Dense

				// t requires iterator
				a1, _ := sliceDense(a.Dense, makeRS(0, 5))
				identity = newDense({{asType . | title | strip}}, 5)
				{{if ne $iden 0 -}}
					identity.Memset({{asType .}}({{$iden}}))
				{{end -}}
				correct = newDense({{asType . | title | strip}}, 5)
				copyDense(correct, a.Dense)

				ret, _ = a1.{{$op}}(identity, UseUnsafe())
				if !allClose(correct.Data(), ret.Data()){
					return false
				}

				// other requires iterator
				a2 := a1.Materialize().(*Dense)
				identity = newDense({{asType . | title | strip}}, a.len())
				identity, _ = sliceDense(identity, makeRS(0, 5))
				{{if ne $iden 0 -}}
					identity.Memset({{asType .}}({{$iden}}))
				{{end -}}

				ret,_ = a2.{{$op}}(identity, UseUnsafe())
				if !allClose(correct.Data(), ret.Data()){
					return false
				}

				// both requires iterator
				ret, _ = a1.{{$op}}(identity, UseUnsafe())
				if !allClose(correct.Data(), ret.Data()){
					return false
				}



				return true
			}
			if err := quick.Check(idenSliced{{short .}}, nil); err != nil {
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

			pow0Iter{{short .}} := func(a *QCDense{{short .}}) bool {
				var ret, correct, zero *Dense

				// t requires iterator
				a1, _ := sliceDense(a.Dense, makeRS(0,5))
				zero = newDense({{asType . | title | strip}}, 5)
				correct = newDense({{asType . | title | strip}}, 5)
				correct.Memset({{asType .}}(1))
				ret, _ = a1.{{$op}}(zero, UseUnsafe())
				if !allClose(correct.Data(), ret.Data()){
					return false
				}

				// zero requires iterator
				a2 := a1.Materialize().(*Dense)
				zero = newDense({{asType . | title | strip}}, a.len())
				zero, _ = sliceDense(zero, makeRS(0,5))
				ret, _ = a2.{{$op}}(zero, UseUnsafe())
				if !allClose(correct.Data(), ret.Data()){
					return false
				}

				// both requires iterator
				a1, _ = sliceDense(a.Dense, makeRS(6,11))
				ret, _ = a1.{{$op}}(zero, UseUnsafe())
				if !allClose(correct.Data(), ret.Data()) {
					return false
				}

				return true
			}
			if err := quick.Check(pow0Iter{{short .}}, nil); err != nil {
				t.Errorf("Pow 0 with iterator failed")
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
		// incr
		incr{{short .}} := func(a, b, incr *QCDense{{short .}}) bool {
			var correct, clonedIncr, ret, check *Dense

			// build correct
			{{if eq $op "Div" -}} 
				b.Dense.Memset({{asType .}}(1))
			{{end -}}
			ret, _ = a.{{$op}}(b.Dense)
			correct, _ = incr.Add(ret)

			clonedIncr = incr.Clone().(*Dense)
			check, _ = a.{{$op}}(b.Dense, WithIncr(clonedIncr))
			if check != clonedIncr {
				t.Error("Expected clonedIncr == check")
				return false
			}
			if !allClose(correct.Data(), check.Data()) {
				t.Errorf("Failed close")
				return false
			}

			// incr iter
			var oncr, a1, a2, b1, b2 *Dense
			clonedIncr = incr.Dense.Clone().(*Dense)
			oncr, _ = sliceDense(clonedIncr, makeRS(0,5))
			a1, _ = sliceDense(a.Dense, makeRS(0,5))
			a2 = a1.Materialize().(*Dense)
			b1, _ = sliceDense(b.Dense, makeRS(0,5))
			b2 = b1.Materialize().(*Dense)
			// build correct for incr
			correct, _ = sliceDense(correct, makeRS(0,5))

			// check: a requires iter
			check, _ = a1.{{$op}}(b2, WithIncr(oncr))
			if check !=  oncr {
				t.Errorf("expected check == oncr when a requires iter")
				return false
			}
			if !allClose(correct.Data(), check.Data()) {
				return false
			}

			// check: b requires iter
			clonedIncr = incr.Dense.Clone().(*Dense)
			oncr, _ = sliceDense(clonedIncr, makeRS(0,5))
			check, _ = a2.{{$op}}(b1, WithIncr(oncr))
			if check !=  oncr {
				t.Errorf("expected check == oncr when b requires iter")
				return false
			}
			if !allClose(correct.Data(), check.Data()) {
				return false
			}

			// check: both a and b requires iter
			clonedIncr = incr.Dense.Clone().(*Dense)
			oncr, _ = sliceDense(clonedIncr, makeRS(0,5))
			check, _ = a1.{{$op}}(b1, WithIncr(oncr))
			if check !=  oncr {
				t.Errorf("expected check == oncr when b requires iter")
				return false
			}
			if !allClose(correct.Data(), check.Data()) {
				return false
			}

			// check both don't require iter
			clonedIncr = incr.Dense.Clone().(*Dense)
			oncr, _ = sliceDense(clonedIncr, makeRS(0,5))
			check, _ = a2.{{$op}}(b2, WithIncr(oncr))
			if check !=  oncr {
				t.Errorf("expected check == oncr when b requires iter")
				return false
			}
			if !allClose(correct.Data(), check.Data()) {
				return false
			}


			// incr noiter
			clonedIncr = incr.Dense.Clone().(*Dense)
			oncr, _ = sliceDense(clonedIncr, makeRS(0,5))
			oncr = oncr.Materialize().(*Dense)
			correct = correct.Materialize().(*Dense)

			// check: a requires iter
			check, _ = a1.{{$op}}(b2, WithIncr(oncr))
			if check !=  oncr {
				t.Errorf("expected check == oncr when a requires iter")
				return false
			}
			if !allClose(correct.Data(), check.Data()) {
				return false
			}

			// check: b requires iter
			clonedIncr = incr.Dense.Clone().(*Dense)
			oncr, _ = sliceDense(clonedIncr, makeRS(0,5))
			oncr = oncr.Materialize().(*Dense)
			check, _ = a2.{{$op}}(b1, WithIncr(oncr))
			if check !=  oncr {
				t.Errorf("expected check == oncr when b requires iter")
				return false
			}
			if !allClose(correct.Data(), check.Data()) {
				return false
			}

			// check: both a and b requires iter
			clonedIncr = incr.Dense.Clone().(*Dense)
			oncr, _ = sliceDense(clonedIncr, makeRS(0,5))
			oncr = oncr.Materialize().(*Dense)
			check, _ = a1.{{$op}}(b1, WithIncr(oncr))
			if check !=  oncr {
				t.Errorf("expected check == oncr when b requires iter")
				return false
			}
			if !allClose(correct.Data(), check.Data()) {
				return false
			}			



			return true
		}
		if err := quick.Check(incr{{short .}}, nil); err != nil {
			t.Errorf("Incr function test for {{.}} failed %v", err)
		}
	{{end -}}
}
`

const testDDFuncOptRaw = `func Test{{.OpName}}FuncOpts(t *testing.T){
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

const testDSBasicPropertiesRaw = `func Test{{.OpName}}BasicProperties(t *testing.T){
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
				var ret, correct *Dense
				var identity {{asType .}}
				{{if ne $iden 0 -}}
					identity = {{$iden}}
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

			idenIter{{short .}} := func(a *QCDense{{short .}}) bool {
				var a1, ret, correct *Dense
				var identity {{asType .}}
				{{if ne $iden 0 -}}
					identity = {{$iden}}
				{{end -}}
				correct = newDense({{asType . | title | strip}}, a.len())
				copyDense(correct, a.Dense)
				correct, _ = sliceDense(correct, makeRS(0,5))
				correct = correct.Materialize().(*Dense)

				a1, _ = sliceDense(a.Dense, makeRS(0, 5))
				ret, _ = a1.{{$op}}(identity, UseUnsafe())
				if !allClose(correct.Data(), ret.Data()) {
					return false
				}

				// safe:
				ret, _ = a1.{{$op}}(identity)
				if !allClose(correct.Data(), ret.Data()) {
					return false
				}

				return true
			}
			if err := quick.Check(idenIter{{short .}}, nil); err != nil {
				t.Errorf("Identity test with iterable for {{.}} failed %v",err)
			}
		{{end -}}
		{{if hasSuffix $op "R" -}}

		
		{{end -}}
		{{if eq $op "PowOf" -}}
			pow0{{short .}} := func(a *QCDense{{short .}}) bool {
				var ret, correct *Dense 
				var zero {{asType .}}
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

			pow0Iter{{short .}} := func(a *QCDense{{short .}}) bool {
				var a1, ret, correct *Dense 
				var zero {{asType .}}
				correct = newDense({{asType . | title | strip}}, a.len())
				correct.Memset({{asType .}}(1))

				correct, _ = sliceDense(correct, makeRS(0,5))
				correct = correct.Materialize().(*Dense)

				a1, _ = sliceDense(a.Dense, makeRS(0, 5))
				ret, _ = a1.{{$op}}(zero, UseUnsafe())
				if !allClose(correct.Data(), ret.Data()) {
					return false
				}

				// safe 
				ret, _ = a1.{{$op}}(zero)
				if !allClose(correct.Data(), ret.Data()) {
					return false
				}
				return true
			}
			if err := quick.Check(pow0Iter{{short .}}, nil); err != nil {
				t.Errorf("Pow 0 failed")
			}
		{{end -}}
		{{if $isInv -}}
			// invertible property - reserved to test inverse functions
			inv{{short .}} := func(a *QCDense{{short .}}, b {{asType .}}) bool {
				ret1, _ := a.{{$op}}(b)
				ret2, _ := ret1.{{$invOpName}}(b)
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
		incr{{short .}} := func(a, incr *QCDense{{short .}}, b {{asType .}}) bool {
			// build correct
			ret, _ := a.{{$op}}(b)
			correct, _ := incr.Add(ret)

			check, _ := a.{{$op}}(b, WithIncr(incr.Dense))
			if check != incr.Dense {
				t.Error("Expected incr.Dense == check")
				return false
			}
			if !allClose(correct.Data(), check.Data()) {
				t.Errorf("Correct: %v, check %v", correct.Data().([]{{asType .}})[0:10], check.Data().([]{{asType .}})[0:10])
				return false
			}

			return true
		}
		if err := quick.Check(incr{{short .}}, nil); err != nil {
			t.Errorf("Incr function test for {{.}} failed %v", err)
		}
	{{end -}}
}
`

var (
	testDDBasicProperties *template.Template
	testDDFuncOpts        *template.Template

	testDSBasicProperties *template.Template
)

func init() {
	testDDBasicProperties = template.Must(template.New("testDDBasicProp").Funcs(funcs).Parse(testDDBasicPropertiesRaw))
	testDDFuncOpts = template.Must(template.New("testDDFuncOpt").Funcs(funcs).Parse(testDDFuncOptRaw))

	testDSBasicProperties = template.Must(template.New("testDSBasicProp").Funcs(funcs).Parse(testDSBasicPropertiesRaw))
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
		testDDBasicProperties.Execute(f, op)
		testDDFuncOpts.Execute(f, bo)
	}

	for _, bo := range vecscalarOps {
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

		testDSBasicProperties.Execute(f, op)
	}
}
