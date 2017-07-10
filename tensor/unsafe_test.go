package tensor

import "testing"

func Test_Iface(t *testing.T) {
	var a interface{}
	a = complex(float64(3.14), float64(3.14))
	ptr := extractPointer(a)
	b := *(*complex128)(ptr)
	if a != b {
		t.Fatalf("Expected %v. Got %v", a, b)
	}

	a = float64(3.14)
	ptr = extractPointer(a)
	c := *(*float64)(ptr)
	if a != c {
		t.Fatalf("Expected %v. Got %v", a, c)
	}
}
