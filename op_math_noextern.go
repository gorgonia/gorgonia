// +build !cuda

package gorgonia

func (op elemUnaryOp) CallsExtern() bool { return false }
func (op elemBinOp) CallsExtern() bool   { return false }
