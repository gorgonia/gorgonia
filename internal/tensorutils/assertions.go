package gtu

import (
	"context"

	"github.com/pkg/errors"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/dense"
)

// contexter is any engine (or type) that returns the current context.
type contexter interface {
	Context() context.Context
}

// GetDenseTensor tries to extract a tensor.DenseTensor from a tensor.Tensor.
func GetDenseTensor[DT any, T tensor.Tensor[DT, T]](t tensor.DescWithStorage) (dense.DenseTensor[DT, T], error) {
	switch tt := t.(type) {
	case dense.DenseTensor[DT, T]:
		return tt, nil
	// case dense.Densor[DT]:
	// 	var z T

	// 	return z.FromDense(tt.GetDense()), nil
	default:
		_ = tt
		return nil, errors.Errorf("Tensor %T is not a DenseTensor", t)
	}
}

// CtxFromEngine returns a context from an engine if the engine contains a context.
// Otherwise context.Background() is returned.
func CtxFromEngine(e tensor.Engine) context.Context {
	if c, ok := e.(contexter); ok {
		return c.Context()
	}
	return context.Background()
}

// CtxFromEngines returns the first context from an engine that has a context.
// If none is found, context.Background() is returned.
func CtxFromEngines(es ...tensor.Engine) context.Context {
	for _, e := range es {
		if c, ok := e.(contexter); ok {
			return c.Context()
		}
	}
	return context.Background()
}
