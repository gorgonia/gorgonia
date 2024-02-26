package internal

import (
	"context"
	"gorgonia.org/gorgonia/internal/errors"
)

// HandleCtx is the default handler for contexts.
func HandleCtx(ctx context.Context) error {
	select {
	case <-ctx.Done():
		return errors.NoOp{}
	default:
	}
	return nil
}

// HandleNoOp is the default handler for no op errors.
func HandleNoOp(err error) error {
	if _, ok := err.(errors.NoOpError); ok {
		return nil
	}
	if err == nil {
		return nil
	}
	return err
}
