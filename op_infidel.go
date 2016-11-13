package gorgonia

import (
	"hash"
	"hash/fnv"

	"github.com/chewxy/gorgonia/tensor/types"
)

/*
This file contains code for Ops that aren't really functions in the sense that they aren't pure.

Since they're not adherents to the Church of Lambda, they are INFIDELS! A fatwa will be issued on them shortly

*/

type stmtOp interface {
	Op
	isStmt() bool
}

// letOp is not really a function. It's more of a binding statement.
// However, it's implemented as a Op so that it can be counted for register allocation and liveness
type letOp struct{}

func (op letOp) Arity() int                                                      { return 0 }
func (op letOp) Type() Type                                                      { return nil }
func (op letOp) ReturnsPtr() bool                                                { return true }
func (op letOp) OverwritesInput() int                                            { return 0 }
func (op letOp) CallsExtern() bool                                               { return false }
func (op letOp) InferShape(...DimSizer) (types.Shape, error)                     { return nil, nil }
func (op letOp) DiffWRT(int) []bool                                              { return nil }
func (op letOp) SymDiff(inputs Nodes, outputNode, gradNode *Node) (Nodes, error) { return nil, nil }
func (op letOp) Do(vals ...Value) (Value, error)                                 { return nil, nil }
func (op letOp) String() string                                                  { return "=" }
func (op letOp) WriteHash(h hash.Hash)                                           { h.Write([]byte("let")) }
func (op letOp) Hashcode() uint32 {
	h := fnv.New32a()
	op.WriteHash(h)
	return h.Sum32()
}

func (op letOp) isStmt() bool { return true }

// readOp reads a value off the input. This op ensures that a value used, and hence codegen'd out
type readOp struct {
	into *Value // no, it's not a mistake. It's a pointer to a Value (which is an interface{} type)
}

func (op readOp) Arity() int                                                      { return 0 }
func (op readOp) Type() Type                                                      { return nil }
func (op readOp) ReturnsPtr() bool                                                { return true }
func (op readOp) OverwritesInput() int                                            { return 0 }
func (op readOp) CallsExtern() bool                                               { return false }
func (op readOp) InferShape(...DimSizer) (types.Shape, error)                     { return nil, nil }
func (op readOp) DiffWRT(int) []bool                                              { return nil }
func (op readOp) SymDiff(inputs Nodes, outputNode, gradNode *Node) (Nodes, error) { return nil, nil }
func (op readOp) Do(vals ...Value) (Value, error)                                 { return nil, nil }
func (op readOp) String() string                                                  { return "print" }
func (op readOp) WriteHash(h hash.Hash)                                           { h.Write([]byte("print")) }
func (op readOp) Hashcode() uint32 {
	h := fnv.New32a()
	op.WriteHash(h)
	return h.Sum32()
}

func (op readOp) isStmt() bool { return true }
