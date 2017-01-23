package main

import (
	"fmt"
	"io"
	"text/template"
)

const testAdditionRaw = `func TestAddition{{short .}}(t *testing.T){
	var f func(*QCDense{{short .}}) bool
	var err error

	f = func(x *QCDense{{short .}}) bool {
		var ret, correct, zero, reuse *Dense
		var err error
		zero = newDense({{asType . | title | strip}}, x.len()+1)

		// basic length test 
		if _, err = zero.Add(x.Dense); err == nil {
			t.Error("Failed length test")
			return false
		}

		// safe identity
		zero = newDense({{asType . | title | strip}}, x.len())
		correct = newDense({{asType . | title | strip}}, x.len())
		copyDense(correct, x.Dense)

		if ret, err = zero.Add(x.Dense); err != nil {
			t.Errorf("Failed safe identity test: %v", err)
			return false
		}

		if ret == zero || ret == x.Dense {
			t.Error("Failed safe identity test: safe op should not return same value")
			return false
		}

		if !reflect.DeepEqual(correct.Data(), ret.Data()){
			t.Errorf("Failed safe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// reuse identity - wrong length
		reuse = newDense({{asType . | title | strip}}, x.len()+1)
		if _, err = x.Add(zero, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when reuse is of a differing size")
			return false
		}

		// reuse identity
		reuse = newDense({{asType . | title | strip}}, x.len())
		if ret, err = zero.Add(x.Dense, WithReuse(reuse)); err != nil {
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
		if ret, err = zero.Add(x.Dense, UseUnsafe()); err != nil{
			t.Errorf("Failed unsafe identity test: %v", err)
			return false
		}
		if ret != zero {
			t.Error("Failed unsafe identity test. Expected the return *Dense to be the same as the left")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()){
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// test commutativity and associativity
		zero = newDense({{asType . | title | strip}}, x.len())
		if ret, err = x.Add(zero); err != nil{
			t.Errorf("Failed commutativity/associativity test: %v", err)
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()){
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}		

		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	f = func(x *QCDense{{short .}}) bool {
		var ret, correct, reuse *Dense
		var zero {{asType .}}
		var err error
		zero = 0
		correct = newDense({{asType . | title | strip}}, x.len())
		copyDense(correct, x.Dense)

		// Safe Trans
		if ret, err = x.Trans(zero); err != nil {
			t.Errorf("Failed Safe Trans test %v", err)
			return false
		}
		if ret == x.Dense {
			t.Error("Failed Safe Trans: ret == x")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()){
			t.Errorf("Failed Safe Trans: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Reuse Trans
		reuse = newDense({{asType . | title | strip}}, x.len())
		if ret, err = x.Trans(zero, WithReuse(reuse)); err != nil {
			t.Errorf("Failed Reuse Trans test %v", err)
			return false
		}
		if ret != reuse {
			t.Error("Failed Reuse Trans: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()){
			t.Errorf("Failed Reuse Trans: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Unsafe Trans
		if ret, err = x.Trans(zero, UseUnsafe()); err != nil {
			t.Errorf("Unsafe trans test failed:  %v", err)
			return false
		}
		if ret != x.Dense {
			t.Error("Failed Unsafe Trans: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()){
			t.Errorf("Failed Unsafe Trans: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false	
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Trans tests: %v", err)
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
	testAddition       *template.Template
	testSubtraction    *template.Template
	testMultiplication *template.Template
)

func init() {
	testAddition = template.Must(template.New("testAddition").Funcs(funcs).Parse(testAdditionRaw))
	testSubtraction = template.Must(template.New("testSubtraction").Funcs(funcs).Parse(testSubtractionRaw))
	testMultiplication = template.Must(template.New("testMul").Funcs(funcs).Parse(testMultiplicationRaw))
}

func denseArithTests(f io.Writer, generic *ManyKinds) {
	for _, k := range generic.Kinds {
		if isNumber(k) {
			fmt.Fprintf(f, "/* %s */\n\n", k)
			testAddition.Execute(f, k)
			testSubtraction.Execute(f, k)
			testMultiplication.Execute(f, k)
		}
	}
}
