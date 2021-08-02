package cuda

import "gorgonia.org/tensor"

func (e *Engine) FMA(a tensor.Tensor, x tensor.Tensor, y tensor.Tensor) (tensor.Tensor, error) {
	panic("not implemented") // TODO: Implement
}

func (e *Engine) FMAScalar(a tensor.Tensor, x interface{}, y tensor.Tensor) (tensor.Tensor, error) {
	panic("not implemented") // TODO: Implement
}

func (e *Engine) MatMul(a tensor.Tensor, b tensor.Tensor, preallocated tensor.Tensor) error {
	panic("not implemented") // TODO: Implement
}

func (e *Engine) MatVecMul(a tensor.Tensor, b tensor.Tensor, preallocated tensor.Tensor) error {
	panic("not implemented") // TODO: Implement
}

func (e *Engine) Outer(a tensor.Tensor, b tensor.Tensor, preallocated tensor.Tensor) error {
	panic("not implemented") // TODO: Implement
}

func (e *Engine) Dot(a tensor.Tensor, b tensor.Tensor, opts ...tensor.FuncOpt) (tensor.Tensor, error) {
	panic("not implemented") // TODO: Implement
}

func (e *Engine) SVD(a tensor.Tensor, uv bool, full bool) (s tensor.Tensor, u tensor.Tensor, v tensor.Tensor, err error) {
	panic("not implemented") // TODO: Implement
}
