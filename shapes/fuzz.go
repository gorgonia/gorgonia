// +build gofuzz

package shapes

import (
	fuzz "github.com/google/gofuzz"
)

func Fuzz(data []byte) int {
	if len(data) == 0 {
		return -1
	}
	// fuzz ops
	var bo BinOp
	fuzz.NewFromGoFuzz(data).Fuzz(&bo)
	if bo.isValid() {
		return 1
	}

	sz, err := bo.resolveSize()
	if err == nil {
		return 1
	}
	if sz == 0 {
		panic("XXX")
	}
	return 0
}
