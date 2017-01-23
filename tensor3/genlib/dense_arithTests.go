package main

import (
	"fmt"
	"io"
	"text/template"
)

const testAdditionRaw = `func TestAddition{{short .}}(t *testing.T){
	var f func(*QCDense{{short .}}) bool
	var err error

	// basic length test 
	f = func(x *QCDense{{short .}}) bool {
		zero := newDense({{asType . | title | strip}}, x.len()+1)

		_, err := zero.Add(x.Dense)
		if err == nil {
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic unsafe identity test
	f = func(x *QCDense{{short .}}) bool {
		zero := newDense({{asType . | title | strip}}, x.len())
		correct := newDense({{asType . | title | strip}}, x.len())
		copyDense(correct, x.Dense)

		ret, err := zero.Add(x.Dense, UseUnsafe())
		if err != nil {
			return false
		}
		if ret != zero {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic safe identity test
	f = func(x *QCDense{{short .}}) bool {
		zero := newDense({{asType . | title | strip}}, x.len())		

		ret, err := zero.Add(x.Dense)
		if err != nil {
			return false
		}
		if ret == zero {
			return false
		}
		return reflect.DeepEqual(ret.Data(), x.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Identity test: %v", err)
	}

	// reuse identity test
	f = func(x *QCDense{{short .}}) bool {
		zero := newDense({{asType . | title | strip}}, x.len())
		reuse := newDense({{asType . | title | strip}}, x.len())
		ret, err := zero.Add(x.Dense, WithReuse(reuse))
		if err != nil {
			return false
		}
		if ret != reuse {
			return false
		}
		return reflect.DeepEqual(ret.Data(), x.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Reuse Identity test: %v", err)
	}

	// commutativity test
	f = func(x *QCDense{{short .}}) bool {
		zero := newDense({{asType . | title | strip}}, x.len())
		correct := newDense({{asType . | title | strip}}, x.len())
		copyDense(correct, x.Dense)
		ret, err := x.Add(zero)
		if err != nil {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Commutativity/Associativity Identity test: %v", err)
	}

	// Safe Trans
	f = func(x *QCDense{{short .}}) bool {
		zero := {{asType .}}(0)
		correct := newDense({{asType . | title | strip}}, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Trans(zero)
		if err != nil {
			return false
		}
		if ret == x.Dense {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Trans Identity test: %v", err)
	}

	// Unsafe Trans
	f = func(x *QCDense{{short .}}) bool {
		zero := {{asType .}}(0)
		correct := newDense({{asType . | title | strip}}, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Trans(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if ret != x.Dense {
			return false
		}

		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Trans Identity test: %v", err)
	}

	// reuse Trans
	f = func(x *QCDense{{short .}}) bool {
		zero := {{asType .}}(0)
		reuse := newDense({{asType . | title | strip}}, x.len())
		correct := newDense({{asType . | title | strip}}, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Trans(zero, WithReuse(reuse))
		if err != nil {
			return false
		}
		if ret != reuse {
			return false
		}

		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Reuse Trans Identity test: %v", err)
	}
}
`

const testSubtractionRaw = `func TestSubtraction{{short .}}(t *testing.T){
	var f func(*QCDense{{short .}}) bool
	var err error

	// basic length test 
	f = func(x *QCDense{{short .}}) bool {
		zero := newDense({{asType . | title | strip}}, x.len()+1)

		_, err := x.Sub(zero)
		if err == nil {
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic unsafe identity test
	f = func(x *QCDense{{short .}}) bool {
		zero := newDense({{asType . | title | strip}}, x.len())
		correct := newDense({{asType . | title | strip}}, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Sub(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if ret != x.Dense {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic safe identity test
	f = func(x *QCDense{{short .}}) bool {
		zero := newDense({{asType . | title | strip}}, x.len())		
		correct := newDense({{asType . | title | strip}}, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Sub(zero)
		if err != nil {
			return false
		}
		if ret == zero {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Identity test: %v", err)
	}

	// reuse identity test
	f = func(x *QCDense{{short .}}) bool {
		zero := newDense({{asType . | title | strip}}, x.len())
		reuse := newDense({{asType . | title | strip}}, x.len())
		ret, err := x.Sub(zero, WithReuse(reuse))
		if err != nil {
			return false
		}
		if ret != reuse {
			return false
		}
		return reflect.DeepEqual(ret.Data(), x.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Reuse Identity test: %v", err)
	}

	// TransInv - tests of commutativity  - safe one
	f = func(x *QCDense{{short .}}) bool {
		zero := {{asType .}}(0)
		xminus, err := x.TransInv(zero)
		if err != nil {
			return false
		}

		zerominus, err := x.TransInvR(zero)
		if err != nil {
			return false
		}
		return !reflect.DeepEqual(xminus.Data(), zerominus.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe TransInc test: %v", err)
	}

	// TransInv - tests of commutativity  - unsafe
	f = func(x *QCDense{{short .}}) bool {
		zero := {{asType .}}(0)
		cloned := x.Clone().(*Dense)
		xminus, err := x.TransInv(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if xminus != x.Dense {
			return false
		}

		zerominus, err := x.TransInvR(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if zerominus != x.Dense {
			return false
		}
		return !reflect.DeepEqual(cloned.Data(), zerominus.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe TransInv test: %v", err)
	}
}
`

var (
	testAddition    *template.Template
	testSubtraction *template.Template
)

func init() {
	testAddition = template.Must(template.New("testAddition").Funcs(funcs).Parse(testAdditionRaw))
	testSubtraction = template.Must(template.New("testSubtraction").Funcs(funcs).Parse(testSubtractionRaw))
}

func denseArithTests(f io.Writer, generic *ManyKinds) {
	for _, k := range generic.Kinds {
		if isNumber(k) {
			fmt.Fprintf(f, "/* %s */\n\n", k)
			testAddition.Execute(f, k)
			testSubtraction.Execute(f, k)
		}
	}
}
