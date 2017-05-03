// +build !cuda

package gorgonia

func (op repeatOp) CallsExtern() bool { return false }
