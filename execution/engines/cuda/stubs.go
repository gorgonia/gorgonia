package cuda

import (
	"context"

	"gorgonia.org/tensor"
)

func (e *Engine) FMA(ctx context.Context, a tensor.Tensor, x tensor.Tensor, y tensor.Tensor) (tensor.Tensor, error) {
	panic("not implemented") // TODO: Implement
}

func (e *Engine) FMAScalar(ctx context.Context, a tensor.Tensor, x interface{}, y tensor.Tensor) (tensor.Tensor, error) {
	panic("not implemented") // TODO: Implement
}

func (e *Engine) Dot(a tensor.Tensor, b tensor.Tensor, opts ...tensor.FuncOpt) (tensor.Tensor, error) {
	panic("not implemented") // TODO: Implement
}

func (e *Engine) SVD(ctx context.Context, a tensor.Tensor, uv bool, full bool) (s tensor.Tensor, u tensor.Tensor, v tensor.Tensor, err error) {
	panic("not implemented") // TODO: Implement
}
