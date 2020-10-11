package gorgonia

import "github.com/pkg/errors"

// DiagFlat takes the flattened value and creates a diagonal matrix from it.
//
// It is non-differentiable.
func DiagFlat(a *Node) (*Node, error) {
	if a.Shape().IsScalarEquiv() {
		return nil, errors.Errorf("Cannot perform DiagFlat on a scalar equivalent node")
	}
	return ApplyOp(diagFlatOp{}, a)
}
