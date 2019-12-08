package gorgonia

import (
	"fmt"
	"log"
	"testing"

	"gorgonia.org/tensor"
)

func Example_batchedMatMul() {
	g := NewGraph()
	a := NewTensor(g, Float64, 3, WithShape(2, 2, 3), WithInit(RangedFrom(1)), WithName("a"))
	b := NewTensor(g, Float64, 3, WithShape(2, 3, 2), WithInit(RangedFrom(13)), WithName("b"))
	c, err := BatchedMatMul(a, b)
	if err != nil {
		log.Fatal(err)
	}
	x := NewTensor(g, Float64, 4, WithShape(3, 2, 2, 3), WithInit(RangedFrom(1)), WithName("x"))
	y := NewTensor(g, Float64, 4, WithShape(3, 2, 3, 2), WithInit(RangedFrom(37)), WithName("y"))
	z, err := BatchedMatMul(x, y)
	if err != nil {
		log.Fatal(err)
	}

	m := NewTapeMachine(g)
	if err := m.RunAll(); err != nil {
		log.Fatal(err)
	}

	fmt.Printf("a: %v\n%v\n", a.Value().Shape(), a.Value().Data())
	fmt.Printf("b: %v\n%v\n", b.Value().Shape(), b.Value().Data())
	fmt.Printf("c: %v\n%v\n", c.Value().Shape(), c.Value().Data())
	fmt.Printf("x: %v\n%v\n", x.Value().Shape(), x.Value().Data())
	fmt.Printf("y: %v\n%v\n", y.Value().Shape(), y.Value().Data())
	fmt.Printf("z: %v\n%v\n", z.Value().Shape(), z.Value().Data())

	// Output:
	// a: (2, 2, 3)
	// [1 2 3 4 5 6 7 8 9 10 11 12]
	// b: (2, 3, 2)
	// [13 14 15 16 17 18 19 20 21 22 23 24]
	// c: (2, 2, 2)
	// [94 100 229 244 508 532 697 730]
	// x: (3, 2, 2, 3)
	// [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36]
	// y: (3, 2, 3, 2)
	// [37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72]
	// z: (3, 2, 2, 2)
	// [238 244 589 604 1084 1108 1489 1522 2146 2188 2605 2656 3424 3484 3937 4006 4918 4996 5485 5572 6628 6724 7249 7354]

}

func TestIncrSlices(t *testing.T) {
	// validSlices to see if the slice and shape matches
	validSlices := func(a []sli, shp tensor.Shape) bool {
		for i := range a {
			if a[i].start < shp[i] {
				return true
			}
		}
		return false
	}

	shp := tensor.Shape{2, 3, 4}
	slices := make([]sli, len(shp))
	for i := range slices {
		slices[i].end = 1
	}

	for halt := false; !halt; halt = incrSlices(slices, shp) {
		if !validSlices(slices, shp) {
			t.Errorf("Generated invalid slice %v", slices)
		}
	}
}
