package main

import (
	"fmt"
	"io"
	"text/template"
)

const testAdditionBasicPropertiesRaw = `func TestAdditionBasicProperties(t *testing.T){
	{{range .Kinds -}}
	iden{{short .}} := func(a *QCDense{{short .}}) bool {
		var ret, correct, identity *Dense
		identity = newDense({{asType . | title | strip}}, a.len())
		correct = newDense({{asType . | title | strip}}, a.len())
		copyDense(correct, a.Dense)

		ret, _ = identity.Add(a.Dense)
		if !reflect.DeepEqual(correct.Data(), ret.Data()){
			return false
		}
		return true
	}
	if err := quick.Check(iden{{short .}}, nil); err != nil {
		t.Error(err)
	}
	comm{{short .}} := func(a, b *QCDense{{short .}}) bool {
		var ret1, ret2 *Dense
		ret1, _ = a.Add(b.Dense)
		ret2, _ = b.Add(a.Dense)
		if !reflect.DeepEqual(ret1.Data(), ret2.Data()){
			return false
		}
		return true
	}
	if err := quick.Check(comm{{short .}}, nil); err != nil {
		t.Error(err)
	}
	assoc{{short .}} := func(a, b, c *QCDense{{short .}}) bool {
		var ret1, ret2 *Dense
		ret1, _ = a.Add(b.Dense)
		ret1, _ = ret1.Add(c.Dense)

		ret2, _ = b.Add(c.Dense)
		ret2, _ = a.Add(ret2)

		if !reflect.DeepEqual(ret1.Data(), ret2.Data()){
			t.Errorf("%v\n%v",ret1.Data(), ret2.Data())
			return false
		}
		return true
	}
	if err := quick.Check(assoc{{short .}}, nil); err != nil {
		t.Error(err)
	}
	{{end -}}
}
`

const testAdditionFuncOptRaw = `func TestAdditionFuncOpts(t *testing.T){
	var f func(*QCDenseI) bool
	f = func(a *QCDenseI) bool {
		identity := newDense(Int, a.len()+1)
		if _, err := identity.Add(a.Dense); err == nil {
			t.Error("Failed length mismatch test")
			return false
		}
		return true
	}
	if err := quick.Check(f, nil); err != nil {
		t.Error(err)
	}

	// safe
	f = func(a *QCDenseI) bool {
		var identity, ret *Dense
		var err error
		identity = newDense(Int, a.len())
		if ret, err = identity.Add(a.Dense); err != nil {
			t.Error(err)
			return false
		}
		if ret == identity || ret == a.Dense {
			t.Errorf("Failed safe test for Add")
			return false
		}
		return true
	}
	if err := quick.Check(f, nil); err != nil {
		t.Error(err)
	}

	// reuse
	f = func(a *QCDenseI) bool {
		var identity, ret, correct, reuse *Dense
		var err error
		identity = newDense(Int, a.len())
		reuse = newDense(Int, a.len())
		correct = newDense(Int, a.len())
		copyDense(correct, a.Dense)
		if ret, err = identity.Add(a.Dense, WithReuse(reuse)); err != nil {
			t.Error(err)
			return false
		}
		if ret != reuse {
			t.Error("Expected ret == reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()){
			return false
		}

		// wrong reuse type
		reuse = newDense(Bool, a.len())
		if _, err = identity.Add(a.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when using a reuse with a type mismatch")
			return false
		}

		// wrong reuse length
		reuse = newDense(Int, a.len()+1)
		if _, err = identity.Add(a.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when using a reuse with a size mismatch")
			return false
		}
		return true
	}
	if err := quick.Check(f, nil); err != nil {
		t.Error(err)
	}

	// unsafe 
	f = func(a *QCDenseI) bool {
		var identity, ret, correct *Dense
		var err error
		identity = newDense(Int, a.len())
		correct = newDense(Int, a.len())
		copyDense(correct, a.Dense)

		if ret, err = identity.Add(a.Dense, UseUnsafe()) ; err != nil {
			t.Error(err)
			return false
		}
		if ret != identity {
			t.Error("Expected ret == reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()){
			return false
		}
		return true
	}
	if err := quick.Check(f, nil); err != nil {
		t.Error(err)
	}
}
`

const testSubtractionRaw = `func TestSubtraction{{short .}}(t *testing.T){
	var f func(*QCDense{{short .}}) bool
	var err error

	f = func(x *QCDense{{short .}}) bool {
		var ret, correct, zero, reuse *Dense
		var err error
		zero = newDense({{asType . | title | strip}}, x.len()+1)

		if _, err = x.Sub(zero); err == nil {
			t.Error("Failed length test")
			return false
		}

		// safe identity
		zero = newDense({{asType . | title | strip}}, x.len())
		correct = newDense({{asType . | title | strip}}, x.len())
		copyDense(correct, x.Dense)

		if ret, err = x.Sub(zero); err != nil {
			t.Errorf("Failed to minus 0 :%v", err)
			return false
		}
		if ret == zero || ret == x.Dense {
			t.Error("Expected safe identity function to return a completely new *Dense")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()){
			t.Errorf("Safe Identity Test failed. \nWant:%v\nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// reuse identity - wrong length
		reuse = newDense({{asType . | title | strip}}, x.len()+1)
		if _, err = x.Sub(zero, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when reuse is of a differing size")
			return false
		}

		reuse = newDense({{asType . | title | strip}}, x.len())
		if ret, err = x.Sub(zero, WithReuse(reuse)); err != nil {
			t.Errorf("Failed reuse identity test: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Failed reuse identity test. Expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()){
			t.Errorf("Failed resue identity test: Operation incorrect: \nWant: %v \nGot: %v", correct.Data(), ret.Data())
		}

		// unsafe identity
		if ret, err = x.Sub(zero, UseUnsafe()); err != nil{
			t.Errorf("Failed unsafe identity test: %v", err)
			return false
		}
		if ret != ret {
			t.Error("Failed unsafe identity test. Expected the return *Dense to be the same as the left")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()){
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// subtractions are NOT commutative
		zero = newDense({{asType . | title | strip}}, x.len())
		ret, _ = x.Sub(zero)
		ret2, _ := zero.Sub(x.Dense)
		if reflect.DeepEqual(ret.Data(), ret2.Data()) {
			t.Error("Subtraction should NOT be commutative")
			return false
		}

		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Identity test: %v", err)
	}

	f = func(x *QCDense{{short .}}) bool {
		var ret1, ret2, reuse1, reuse2 *Dense
		var zero {{asType .}}
		var err error
		zero = 0

		// safe TransInv/TransInvR tests
		if ret1, err = x.TransInv(zero); err != nil {
			t.Errorf("Error while safe TransInv: %v", err)
			return false
		}
		if ret2, err = x.TransInvR(zero); err != nil {
			t.Errorf("Error while safe TransInvR: %v", err)
			return false
		}
		if reflect.DeepEqual(ret1.Data(), ret2.Data()){
			t.Error("Expected subtraction to be non commutative")
			return false
		}

		// reuse - wrong length
		reuse1 = newDense({{asType . | title | strip}}, x.len()+1)
		if _, err = x.TransInv(zero, WithReuse(reuse1)); err == nil {
			t.Error("Expected an error when reuse size does not match")
			return false
		}

		// reuse - correct
		reuse1 = newDense({{asType . | title | strip}}, x.len())
		reuse2 = newDense({{asType . | title | strip}}, x.len())
		if ret1, err = x.TransInv(zero, WithReuse(reuse1)); err != nil {
			t.Errorf("Error while reuse TransInv: %v", err)
			return false
		}
		if ret2, err = x.TransInvR(zero, WithReuse(reuse2)); err != nil {
			t.Errorf("Error while reuse TransInvR: %v", err)
			return false
		}
		if ret1 != reuse1 {
			t.Errorf("Expected ret1 to be the same as reuse1")
			return false
		}
		if ret2 != reuse2 {
			t.Errorf("Expected ret2 to be the same as reuse2")
			return false
		}
		if reflect.DeepEqual(ret1.Data(), ret2.Data()){
			t.Error("Expected subtraction to be non commutative")
			return false
		}

		// unsafe 
		x2 := newDense({{asType . | title | strip}}, x.len())
		copyDense(x2, x.Dense)
		if ret1, err = x.TransInv(zero, UseUnsafe()); err != nil {
			t.Errorf("Error while reuse TransInv: %v", err)
			return false
		}
		if ret2, err = x2.TransInvR(zero, UseUnsafe()); err != nil {
			t.Errorf("Error while reuse TransInvR: %v", err)
			return false
		}
		if ret1 != x.Dense {
			t.Errorf("Expected ret1 to be the same as x")
			return false
		}
		if ret2 != x2 {
			t.Errorf("Expected ret2 to be the same as x2")
			return false
		}
		if reflect.DeepEqual(ret1.Data(), ret2.Data()){
			t.Error("Expected subtraction to be non commutative")
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed TransInv test: %v", err)
	}
}
`

const testMultiplicationRaw = `func TestMultiplication{{short .}}(t *testing.T){
	var f func(*QCDense{{short .}}) bool
	var err error

	f = func(x *QCDense{{short .}}) bool {
		var ret, correct, one, reuse *Dense
		var err error
		var oneScalar {{asType .}} = 1
		one = newDense({{asType . | title | strip}}, x.len()+1)
		one.Memset(oneScalar)

		// basic length test 
		if _, err = one.Mul(x.Dense); err == nil {
			t.Error("Failed length test")
			return false
		}

		// safe identity
		one = newDense({{asType . | title | strip}}, x.len())
		one.Memset(oneScalar)
		correct = newDense({{asType . | title | strip}}, x.len())
		copyDense(correct, x.Dense)

		if ret, err = one.Mul(x.Dense); err != nil {
			t.Errorf("Failed safe identity test: %v", err)
			return false
		}

		if ret == one || ret == x.Dense {
			t.Error("Failed safe identity test: safe op should not return same value")
			return false
		}

		if !reflect.DeepEqual(correct.Data(), ret.Data()){
			t.Errorf("Failed safe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// reuse identity - wrong length
		reuse = newDense({{asType . | title | strip}}, x.len()+1)
		if _, err = x.Mul(one, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when reuse is of a differing size")
			return false
		}

		// reuse identity
		reuse = newDense({{asType . | title | strip}}, x.len())
		if ret, err = one.Mul(x.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Failed reuse identity test: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Failed reuse identity test. Expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()){
			t.Errorf("Failed reuse identity test: Operation incorrect: \nWant: %v \nGot: %v", correct.Data(), ret.Data())
		}

		// unsafe identity
		if ret, err = one.Mul(x.Dense, UseUnsafe()); err != nil{
			t.Errorf("Failed unsafe identity test: %v", err)
			return false
		}
		if ret != one {
			t.Error("Failed unsafe identity test. Expected the return *Dense to be the same as the left")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()){
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// test commutativity and associativity
		one = newDense({{asType . | title | strip}}, x.len())
		one.Memset(oneScalar)
		if ret, err = x.Mul(one); err != nil{
			t.Errorf("Failed commutativity/associativity test: %v", err)
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()){
			t.Errorf("Failed commutativity/associativity  test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}		

		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	f = func(x *QCDense{{short .}}) bool {
		var ret, correct, reuse *Dense
		var one {{asType .}}
		var err error
		one = 1
		correct = newDense({{asType . | title | strip}}, x.len())
		copyDense(correct, x.Dense)

		// Safe Scale
		if ret, err = x.Scale(one); err != nil {
			t.Errorf("Failed Safe Scale test %v", err)
			return false
		}
		if ret == x.Dense {
			t.Error("Failed Safe Scale: ret == x")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()){
			t.Errorf("Failed Safe Scale: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Reuse Scale
		reuse = newDense({{asType . | title | strip}}, x.len())
		if ret, err = x.Scale(one, WithReuse(reuse)); err != nil {
			t.Errorf("Failed Reuse Scale test %v", err)
			return false
		}
		if ret != reuse {
			t.Error("Failed Reuse Scale: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()){
			t.Errorf("Failed Reuse Scale: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Unsafe Scale
		if ret, err = x.Scale(one, UseUnsafe()); err != nil {
			t.Errorf("Unsafe trans test failed:  %v", err)
			return false
		}
		if ret != x.Dense {
			t.Error("Failed Unsafe Scale: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()){
			t.Errorf("Failed Unsafe Scale: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false	
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Scale tests: %v", err)
	}
}
`

var (
	testAdditionBasicProperties *template.Template
	testAdditionFuncOpt         *template.Template
	testSubtraction             *template.Template
	testMultiplication          *template.Template
)

func init() {
	testAdditionBasicProperties = template.Must(template.New("testAdditionBasicProp").Funcs(funcs).Parse(testAdditionBasicPropertiesRaw))
	testAdditionFuncOpt = template.Must(template.New("testAdditionFuncOpt").Funcs(funcs).Parse(testAdditionFuncOptRaw))
	testSubtraction = template.Must(template.New("testSubtraction").Funcs(funcs).Parse(testSubtractionRaw))
	testMultiplication = template.Must(template.New("testMul").Funcs(funcs).Parse(testMultiplicationRaw))
}

func denseArithTests(f io.Writer, generic *ManyKinds) {
	numbers := filter(generic.Kinds, isNumber)
	mk := &ManyKinds{numbers}
	testAdditionBasicProperties.Execute(f, mk)
	testAdditionFuncOpt.Execute(f, mk)
	for _, k := range generic.Kinds {
		if isNumber(k) {
			fmt.Fprintf(f, "/* %s */\n\n", k)

			testSubtraction.Execute(f, k)
			testMultiplication.Execute(f, k)
		}
	}
}
