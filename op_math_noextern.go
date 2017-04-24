// +build !cuda

package gorgonia

func (op elemUnaryOp) CallsExtern() bool { return false }
func (op elemBinOp) CallsExtern() bool   { return false }

func NewAddOp(a, b *Node, ctx ExecutionContext) *ExternalOp {
	add := newElemBinOp(addOpType, a, b)
	op := NewExternalOp(add, ctx, nil)
	op.Device = CPU
	return op
}
