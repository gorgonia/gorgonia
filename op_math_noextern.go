// +build !cuda

package gorgonia

func (op elemUnaryOp) CallsExtern() bool { return false }
func (op elemBinOp) CallsExtern() bool   { return false }

func NewAddOp(a, b *Node, ctx ExecutionContext) *ExternalOp {
	add := newElemBinOp(addOpType, a, b)
	op := NewExternalOp(add, ctx, nil)
	op.Device = CPU
	op.UseCPU = true
	return op
}

// NewSubOp creates a new *ExternalOp that wraps a sub op
func NewSubOp(a, b *Node, ctx ExecutionContext) *ExternalOp {
	sub := newEBOByType(subOpType, a.t, b.t)
	op := NewExternalOp(sub, ctx, nil)
	op.Device = CPU
	op.UseCPU = true
	return op
}

func NewHadamardProdOp(a, b *Node, ctx ExecutionContext) *ExternalOp {
	mul := newEBOByType(mulOpType, a.t, b.t)
	op := NewExternalOp(mul, ctx, nil)
	op.Device = CPU
	op.UseCPU = true
	return op
}
