package main

import "reflect"

type Op interface {
	Name() string
	Arity() int
	SymbolTemplate() string
	TypeClass() TypeClass
}

type BinOp interface {
	Op
	IsFunc() bool
	IsBinOp()
}

type UnaryOp interface {
	Op
	IsUnaryOp()
}

type RetSamer interface {
	RetSame() bool
}

type basicBinOp struct {
	symbol string
	name   string
	isFunc bool
	is     TypeClass
}

func (op basicBinOp) Name() string           { return op.name }
func (op basicBinOp) Arity() int             { return 2 }
func (op basicBinOp) SymbolTemplate() string { return op.symbol }
func (op basicBinOp) TypeClass() TypeClass   { return op.is }
func (op basicBinOp) IsFunc() bool           { return op.isFunc }
func (op basicBinOp) IsBinOp()               {}

type TypedBinOp struct {
	BinOp
	Kind reflect.Kind
}

// IsFunc contains special conditions
func (op TypedBinOp) IsFunc() bool {
	if op.Name() == "Mod" && isFloatCmplx(op.Kind) {
		return true
	}
	return op.BinOp.IsFunc()
}

type TypedCmpBinOp struct {
	TypedBinOp
	retSame bool
}

func (op TypedCmpBinOp) RetSame() bool { return op.retSame }

type unaryOp struct {
	symbol string
	name   string
	is     TypeClass
}

func (op unaryOp) Name() string           { return op.name }
func (op unaryOp) Arity() int             { return 1 }
func (op unaryOp) SymbolTemplate() string { return op.symbol }
func (op unaryOp) TypeClass() TypeClass   { return op.is }
func (op unaryOp) IsUnaryOp()             {}
