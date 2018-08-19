package gorgonia

import (
	"fmt"
	"hash"

	"github.com/chewxy/hm"
	"gorgonia.org/tensor"
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
func (op letOp) Type() hm.Type                                                   { return nil }
func (op letOp) ReturnsPtr() bool                                                { return true }
func (op letOp) OverwritesInput() int                                            { return 0 }
func (op letOp) CallsExtern() bool                                               { return false }
func (op letOp) InferShape(...DimSizer) (tensor.Shape, error)                    { return nil, nil }
func (op letOp) DiffWRT(int) []bool                                              { return nil }
func (op letOp) SymDiff(inputs Nodes, outputNode, gradNode *Node) (Nodes, error) { return nil, nil }
func (op letOp) Do(vals ...Value) (Value, error)                                 { return nil, nil }
func (op letOp) String() string                                                  { return "=" }
func (op letOp) WriteHash(h hash.Hash)                                           { h.Write([]byte("let")) }
func (op letOp) Hashcode() uint32                                                { return simpleHash(op) }

func (op letOp) isStmt() bool { return true }

// readOp reads a value off the input. This op ensures that a value used, and hence codegen'd out
type readOp struct {
	into *Value // no, it's not a mistake. It's a pointer to a Value (which is an interface{} type)
}

func (op readOp) Arity() int                                                      { return 0 }
func (op readOp) Type() hm.Type                                                   { return nil }
func (op readOp) ReturnsPtr() bool                                                { return true }
func (op readOp) OverwritesInput() int                                            { return 0 }
func (op readOp) CallsExtern() bool                                               { return false }
func (op readOp) InferShape(...DimSizer) (tensor.Shape, error)                    { return nil, nil }
func (op readOp) DiffWRT(int) []bool                                              { return nil }
func (op readOp) SymDiff(inputs Nodes, outputNode, gradNode *Node) (Nodes, error) { return nil, nil }
func (op readOp) Do(vals ...Value) (Value, error)                                 { return nil, nil }
func (op readOp) String() string                                                  { return "print" }
func (op readOp) WriteHash(h hash.Hash)                                           { h.Write([]byte("print")) }
func (op readOp) Hashcode() uint32                                                { return simpleHash(op) }

func (op readOp) isStmt() bool { return true }

// devTrans is a dummy Op, used to aid in creating the program that is run in a *tapeMachine. It is inserted not into the graph, but into a slice of sorted nodes, and will not show up in thegraph.
type devTrans struct {
	from, to Device
	toNode   *Node
}

func (op devTrans) Arity() int                                   { panic("not implemented") }
func (op devTrans) Type() hm.Type                                { panic("not implemented") }
func (op devTrans) InferShape(...DimSizer) (tensor.Shape, error) { panic("not implemented") }
func (op devTrans) Do(...Value) (Value, error)                   { panic("not implemented") }
func (op devTrans) ReturnsPtr() bool                             { return false }
func (op devTrans) CallsExtern() bool                            { return true }
func (op devTrans) OverwritesInput() int                         { return -1 }
func (op devTrans) WriteHash(h hash.Hash)                        { fmt.Fprintf(h, "from:%vto%v", op.from, op.to) }
func (op devTrans) Hashcode() uint32                             { return simpleHash(op) }

func (op devTrans) String() string { return fmt.Sprintf("[CP %v %v]", op.from, op.to) }
func (op devTrans) isStmt() bool   { return true }

func (op devTrans) CUDADo(extern External, dev Device, prealloc Value, inputs ...Value) (retVal Value, err error) {
	return nil, nil
}
func (op devTrans) CUDAFuncName() string { return op.String() }
