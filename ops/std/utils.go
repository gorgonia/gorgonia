package stdops

import (
	"context"
	gerrors "gorgonia.org/gorgonia/internal/errors"
)

func handleCtx(ctx context.Context) error {
	if ctx == nil {
		return nil
	}
	select {
	case <-ctx.Done():
		return gerrors.NoOp{}
	default:
	}
	return nil
}
