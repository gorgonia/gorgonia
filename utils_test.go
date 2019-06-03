package gorgonia

import "testing"

var divIntegerTests = []struct {
	a     int
	b     int
	floor int
	ceil  int
}{
	{
		a:     -2,
		b:     2,
		floor: -1,
		ceil:  -1,
	},
	{
		a:     2,
		b:     2,
		floor: 1,
		ceil:  1,
	},
	{
		a:     5,
		b:     2,
		floor: 2,
		ceil:  3,
	},
	{
		a:     -21,
		b:     10,
		floor: -3,
		ceil:  -2,
	},
	{
		a:     29,
		b:     10,
		floor: 2,
		ceil:  3,
	},
}

func TestCeilDivInt(t *testing.T) {
	for _, tst := range divIntegerTests {
		if ceilDivInt(tst.a, tst.b) != tst.ceil {
			t.Fatalf("test %v failed for ceil division (result %v)", tst, ceilDivInt(tst.a, tst.b))
		}
	}
}
func TestFloorDivInt(t *testing.T) {
	for _, tst := range divIntegerTests {
		if floorDivInt(tst.a, tst.b) != tst.floor {
			t.Fatalf("test %v failed for floor division (result %v)", tst, floorDivInt(tst.a, tst.b))
		}
	}
}
