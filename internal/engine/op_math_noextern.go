// +build !cuda

package engine

import "gorgonia.org/gorgonia/internal/execution"

func (op elemUnaryOp) CallsExtern() bool { return false }
func (op elemBinOp) CallsExtern() bool   { return false }
func (op linAlgBinOp) CallsExtern() bool {
	if op.āBinaryOperator != vecDotOperator {
		return true
	}
	return false
}

func NewAddOp(a, b *Node, ctx execution.Context) *ExternalOp {
	add := newElemBinOp(addOpType, a, b)
	op := NewExternalOp(add, ctx, nil)
	op.Device = execution.CPU
	return op
}

// NewSubOp creates a new *ExternalOp that wraps a sub op
func NewSubOp(a, b *Node, ctx execution.Context) *ExternalOp {
	sub := newEBOByType(subOpType, a.t, b.t)
	op := NewExternalOp(sub, ctx, nil)
	op.Device = execution.CPU
	return op
}

func NewHadamardProdOp(a, b *Node, ctx execution.Context) *ExternalOp {
	mul := newEBOByType(mulOpType, a.t, b.t)
	op := NewExternalOp(mul, ctx, nil)
	op.Device = execution.CPU
	return op
}
