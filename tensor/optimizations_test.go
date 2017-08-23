package tensor
import (
	"testing"
)
// this file contains tests to make sure certain algorithms/optimizations aren't crazy

func TestRequiresIterator(t *testing.T) {
	T := New(Of(Int), WithBacking([]int{1,2,3,4}))
	sliced, _ := T.Slice(makeRS(1, 3))
	if requiresIterator(sliced) {
		t.Errorf("Slicing on rows should not require Iterator")
	}
}