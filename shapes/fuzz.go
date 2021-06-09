// +build gofuzz

package shapes

import (
	fuzz "github.com/google/gofuzz"
)

func Fuzz(data []byte) int {
	if len(data) < 5 {
		return -1
	}
	t := true
	for _, v := range data {
		if v != 0 {
			t = false
			break
		}
	}
	if t {
		return -1
	}

	var expr Expr
	fuzz.New().Fuzz(&expr)

	// fuzz ops
	var bo BinOp
	fuzz.NewFromGoFuzz(data).Fuzz(&bo)
	if bo.isValid() {
		return 1
	}

	_, err := bo.resolveSize()
	if err == nil {
		return 1
	}
	_, err = bo.resolveBool()
	if err == nil {
		return 1
	}
	_, _, err = bo.resolveAB()
	if err == nil {
		return 1
	}

	return 0
}
