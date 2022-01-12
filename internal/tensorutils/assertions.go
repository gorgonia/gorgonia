package gtu

import (
	"context"

	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

// contexter is any engine (or type) that returns the current context.
type contexter interface {
	Context() context.Context
}

// GetDenseTensor tries to extract a tensor.DenseTensor from a tensor.Tensor.
func GetDenseTensor(t tensor.Tensor) (tensor.DenseTensor, error) {
	switch tt := t.(type) {
	case tensor.DenseTensor:
		return tt, nil
	case tensor.Densor:
		return tt.Dense(), nil
	default:
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
