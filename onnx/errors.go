package onnx

// ErrNotImplemented is fired when trying to call an operator that is not yet implemented
type ErrNotImplemented struct {
	Operator string
}

func (e *ErrNotImplemented) Error() string {
	return "operato " + e.Operator + " not implemented"
}
