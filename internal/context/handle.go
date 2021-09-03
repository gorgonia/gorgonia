package gctx // import "gorgonia.org/gorgonia/internal/context"

import (
	"context"
	gerrors "gorgonia.org/gorgonia/internal/errors"
)

// Handle is the default handler for contexts
func Handle(ctx context.Context) error {
	select {
	case <-ctx.Done():
		return gerrors.NoOp{}
	default:
	}
	return nil
}
