package shapes

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

func Example_unsafePermute() {
	pattern := []int{2, 1, 3, 4, 0}
	x1 := []int{1, 2, 3, 4, 5}
	x2 := []int{5, 4, 3, 2, 1}
	fmt.Printf("Before:\nx1: %v\nx2: %v\n", x1, x2)

	err := UnsafePermute(pattern, x1, x2)
	if err != nil {
		fmt.Println(err)
	}
	fmt.Printf("After:\nx1: %v\nx2: %v\n\n", x1, x2)

	// when patterns are monotonic and increasing, it is noop

	pattern = []int{0, 1, 2, 3, 4}
	err = UnsafePermute(pattern, x1, x2)
	if _, ok := err.(NoOpError); ok {
		fmt.Printf("NoOp with %v:\nx1: %v\nx2: %v\n\n", pattern, x1, x2)
	}

	// special cases for 2 dimensions

	x1 = x1[:2]
	x2 = x2[:2]
	fmt.Printf("Before:\nx1: %v\nx2: %v\n", x1, x2)
	pattern = []int{1, 0} // the only valid pattern with 2 dimensions
	if err := UnsafePermute(pattern, x1, x2); err != nil {
		fmt.Println(err)
	}
	fmt.Printf("After:\nx1: %v\nx2: %v\n\n", x1, x2)

	// Bad patterns

	// invalid axis
	pattern = []int{2, 1}
	err = UnsafePermute(pattern, x1, x2)
	fmt.Printf("Invalid axis in pattern %v: %v\n", pattern, err)

	// repeated axes
	pattern = []int{1, 1}
	err = UnsafePermute(pattern, x1, x2)
	fmt.Printf("Repeated axes in pattern %v: %v\n", pattern, err)

	// dimension mismatches
	pattern = []int{1}
	err = UnsafePermute(pattern, x1, x2)
	fmt.Printf("Pattern %v has a smaller dimension than xs: %v\n", pattern, err)

	// Output:
	// Before:
	// x1: [1 2 3 4 5]
	// x2: [5 4 3 2 1]
	// After:
	// x1: [3 2 4 5 1]
	// x2: [3 4 2 1 5]
	//
	// NoOp with [0 1 2 3 4]:
	// x1: [3 2 4 5 1]
	// x2: [3 4 2 1 5]
	//
	// Before:
	// x1: [3 2]
	// x2: [3 4]
	// After:
	// x1: [2 3]
	// x2: [4 3]
	//
	// Invalid axis in pattern [2 1]: Invalid axis 2 for ndarray with 2 dimensions
	// Repeated axes in pattern [1 1]: repeated axis 1 in permutation pattern
	// Pattern [1] has a smaller dimension than xs: Dimension mismatch. Expected 2. Got  1 instead.

}

func TestGenericUnsafePermute(t *testing.T) {
	// most generic permutation
	assert := assert.New(t)
	pattern := []int{2, 1, 3, 4, 0}
	x1 := []string{"hello", "world", "a", "b", "c"}
	x2 := []interface{}{1, "hello", 2, "world", 5.0}
	x3 := []int{1, 2, 3, 4, 5}
	x4 := []byte{1, 2, 3, 4, 5}

	if err := genericUnsafePermute(pattern, x1, x2, x3, x4); err != nil {
		t.Fatal(err)
	}

	correctX1 := []string{"a", "world", "b", "c", "hello"}
	correctX2 := []interface{}{2, "hello", "world", 5.0, 1}
	correctX3 := []int{3, 2, 4, 5, 1}
	correctX4 := []byte{3, 2, 4, 5, 1}
	assert.Equal(correctX1, x1)
	assert.Equal(correctX2, x2)
	assert.Equal(correctX3, x3)
	assert.Equal(correctX4, x4)

	// all ints
	x1Is := []int{1, 2, 3, 4, 5}
	x2Is := []Size{5, 4, 3, 2, 1}
	if err := genericUnsafePermute(pattern, x1Is, x2Is); err != nil {
		t.Fatal(err)
	}

	correctX1Is := []int{3, 2, 4, 5, 1}
	correctX2Is := []Size{3, 4, 2, 1, 5}
	assert.Equal(correctX1Is, x1Is)
	assert.Equal(correctX2Is, x2Is)

	// 2 dimensions:
	x1 = x1[:2]
	x2 = x2[:2]
	x3 = x3[:2]
	pattern = []int{1, 0} // this is the only valid pattern for two dimensions. Otherwise it'd be noop
	if err := genericUnsafePermute(pattern, x1, x2, x3); err != nil {
		t.Fatal(err)
	}
	correctX1 = []string{"world", "a"}
	correctX2 = []interface{}{"hello", 2}
	correctX3 = []int{2, 3}
	assert.Equal(correctX1, x1)
	assert.Equal(correctX2, x2)
	assert.Equal(correctX3, x3)

	// noop
	pattern = []int{0, 1}
	err := genericUnsafePermute(pattern, x1, x2, x3)
	if err == nil {
		t.Errorf("Expected an error")
	} else if _, ok := err.(NoOpError); !ok {
		t.Fatal(err)
	}
	assert.Equal(correctX1, x1)
	assert.Equal(correctX2, x2)
	assert.Equal(correctX3, x3)

	// bad patterns

	// imposible axis
	pattern = []int{2, 1}
	if err := genericUnsafePermute(pattern, x1, x2, x3); err == nil {
		t.Errorf("Expected an error")
	}

	// repeated axes
	pattern = []int{0, 0}
	if err := genericUnsafePermute(pattern, x1, x2, x3); err == nil {
		t.Errorf("Expected an error")
	}

	// mismatching dims 1 - pattern is shorter than xs
	x1 = x1[:3]
	x2 = x2[:3]
	pattern = []int{1, 0}
	if err := genericUnsafePermute(pattern, x1, x2); err == nil {
		t.Errorf("Expected an error")
	}

	// mismatching dims 2 - not all x in xs have the same dims
	pattern = []int{2, 1, 0}
	if err := genericUnsafePermute(pattern, x1, x2, x3); err == nil {
		t.Errorf("Expected an error")
	}

	// mismatching dims 2 - pattern is longer than xs
	pattern = []int{0, 1, 3, 2}
	if err := genericUnsafePermute(pattern, x1, x2, x3); err == nil {
		t.Errorf("Expected an error")
	}

	// idiots who will use this wrong (i.e internal use error)

	if err := genericUnsafePermute(pattern); err == nil {
		t.Errorf("Expected an error when len(xs) = 0")
	}

	if err := genericUnsafePermute(pattern, 1, 2); err == nil {
		t.Errorf("Expected an error when xs is not a slice")
	}

}
