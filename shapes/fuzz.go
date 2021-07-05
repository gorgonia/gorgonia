// +build gofuzz

package shapes

import (
	fuzz "github.com/google/gofuzz"
)

func Fuzz(data []byte) int {
	expr, err := Parse(string(data))
	if err != nil {
		return -1
	}
	switch expr.(type) {
	case Shape:
		return -1
	case Abstract:
		return -1
	case Size:
		return -1
	case Var:
		return -1
	}
	var b Shape
	fuzz.New().Fuzz(&b)
	_, err = InferApp(expr, b)
	if err != nil {
		return 0
	}
	return 1
}
