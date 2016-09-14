package gorgonia

import "testing"

func Test_anyToNode(t *testing.T) {
	n := anyToNode(1)
	t.Logf("%T, %v", n, n)
}
