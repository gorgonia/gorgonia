package gorgonnx

import "fmt"

type errOp struct {
	op  string
	err error
}

func (e *errOp) Error() string {
	return fmt.Sprintf("%s: %v", e.op, e.err)
}
