// +build !cuda

package gorgonia

func (op elemUnaryOp) CallsExtern() bool { return false }
