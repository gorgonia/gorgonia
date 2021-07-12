package shapes

import (
	"fmt"
	"testing"
	"testing/quick"
)

func TestShapes_Clone(t *testing.T) {
	f := func(s Shape) bool {
		s2 := s.Clone().(Shape)
		if !s.Eq(s2) {
			return false
		}
		return true
	}

	if err := quick.Check(f, nil); err != nil {
		t.Error(err)
	}
}

var shapesEqCases = []struct {
	a, b Shape
}{
	{Shape{}, Shape{}},      // scalar is always the same
	{Shape{2}, Shape{2, 1}}, // vector "soft equality"
	{Shape{1, 2}, Shape{2}}, // vector "soft equality"
	{Shape{1, 2, 3}, Shape{1, 2, 3}},
}

var shapesNeCases = []struct {
	a, b Shape
}{
	{Shape{}, Shape{1}}, // differing lengths
	{Shape{2}, Shape{1, 3}},
	{Shape{1, 2, 3}, Shape{1, 2, 4}},
}

func TestShapes_Eq(t *testing.T) {
	for _, c := range shapesEqCases {
		if !c.a.Eq(c.b) {
			t.Errorf("Expected %v = %v", c.a, c.b)
		}
		if !c.b.Eq(c.a) {
			t.Errorf("Expected %v = %v", c.b, c.a)
		}
	}

	for _, c := range shapesNeCases {
		if c.a.Eq(c.b) {
			t.Errorf("Expected %v != %v", c.a, c.b)
		}
		if c.b.Eq(c.a) {
			t.Errorf("Expected %v != %v", c.b, c.a)
		}
	}
}

func ExampleShape_IsScalarEquiv() {
	s := Shape{1, 1, 1, 1, 1, 1}
	fmt.Printf("%v is scalar equiv: %t\n", s, s.IsScalarEquiv())

	s = Shape{}
	fmt.Printf("%v is scalar equiv: %t\n", s, s.IsScalarEquiv())

	s = Shape{2, 3}
	fmt.Printf("%v is scalar equiv: %t\n", s, s.IsScalarEquiv())

	s = Shape{0, 0, 0}
	fmt.Printf("%v is scalar equiv: %t\n", s, s.IsScalarEquiv())

	s = Shape{1, 2, 0, 3}
	fmt.Printf("%v is scalar equiv: %t\n", s, s.IsScalarEquiv())

	// Output:
	// (1, 1, 1, 1, 1, 1) is scalar equiv: true
	// () is scalar equiv: true
	// (2, 3) is scalar equiv: false
	// (0, 0, 0) is scalar equiv: true
	// (1, 2, 0, 3) is scalar equiv: false

}
