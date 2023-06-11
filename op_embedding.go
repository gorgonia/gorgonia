package gorgonia

import (
	"fmt"
	"github.com/chewxy/hm"
	"gorgonia.org/tensor"
	"hash"
)

type embeddingOp struct {
}

func (op embeddingOp) Arity() int {
	return 2
}

func (op embeddingOp) Type() hm.Type {
	//TODO implement me
	panic("implement me")
}

func (op embeddingOp) InferShape(sizer ...DimSizer) (tensor.Shape, error) {
	//TODO implement me
	panic("implement me")
}

func (op embeddingOp) Do(value ...Value) (Value, error) {
	//TODO implement me
	panic("implement me")
}

func (op embeddingOp) ReturnsPtr() bool {
	return false
}

func (op embeddingOp) CallsExtern() bool {
	return false
}

func (op embeddingOp) OverwritesInput() int {
	return -1
}

func (op embeddingOp) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, op.String())
}

func (op embeddingOp) Hashcode() uint32 {
	return simpleHash(op)
}

func (op embeddingOp) String() string {
	return "Embedding"
}

type embeddingDiffOp struct {
	*embeddingDiffOp
}
