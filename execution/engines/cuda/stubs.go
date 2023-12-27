package cuda

import (
	"context"

	"gorgonia.org/tensor"
)

func (e *Engine[DT, T]) Inner(ctx context.Context, a, b T) (DT, error) {
	panic("not implemented") // TODO: Implement
}

func (e *Engine[DT, T]) FMA(ctx context.Context, a T, x T, y T) (err error) {
	panic("not implemented") // TODO: Implement
}

func (e *Engine[DT, T]) FMAScalar(ctx context.Context, a T, x DT, y T) (err error) {
	panic("not implemented") // TODO: Implement
}

func (e *Engine[DT, T]) Dot(a T, b T, opts ...tensor.FuncOpt) (T, error) {
	panic("not implemented") // TODO: Implement
}

func (e *Engine[DT, T]) SVD(ctx context.Context, a T, uv bool, full bool) (s T, u T, v T, err error) {
	panic("not implemented") // TODO: Implement
}
