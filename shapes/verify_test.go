package shapes_test

import (
	"fmt"

	. "gorgonia.org/gorgonia/shapes"
)

type T int

func (t T) Shape() Shape {
	switch t {
	case 0:
		return Shape{5, 4, 2}
	case 1:
		return Shape{4, 2}
	case 2:
		return Shape{2, 10}
	default:
		return Shape{10, 10, 10, 10} // bad shape
	}

}

func ExampleVerify() {
	// the actual field type doesn't matter for now
	type A struct {
		A T `shape:"(a, b, c)"`
		B T `shape:"(b, c)"`
	}

	type B struct {
		A
		C T `shape:"(c, d)"`
	}

	a1 := A{0, 1}
	a2 := A{0, 100}

	if err := Verify(a1); err != nil {
		fmt.Printf("a1 is a correct value. No errors expected. Got %v instead\n", err)
	}
	err := Verify(a2)
	if err == nil {
		fmt.Printf("a2 is an incorrect value. Errors expected but none was returned\n")
	}
	fmt.Printf("Cannot Verify a2: %v\n", err)

	// Output:
	// Cannot Verify a2: Unification Fail. (4, 2) ~ (10, 10, 10, 10) cannot proceed as they do not contain the same amount of sub-expressions. (4, 2) has 2 subexpressions while (10, 10, 10, 10) has 4 subexpressions

}
