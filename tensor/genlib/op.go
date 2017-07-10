package main

import "reflect"

type BinOp struct {
	Name        string
	Symbol      string
	MixedType   string // mixed type name
	Inverse     string
	IsFunc      bool
	Commutative bool
	Identity    int                     // yes it's an int
	Is          func(reflect.Kind) bool // type class
}

type UnaryOp struct {
	Name   string
	Symbol string
	Is     func(reflect.Kind) bool // type class
}

var arithBinOps = [...]BinOp{
	{"Add", "+", "Trans", "Sub", false, true, 0, isAddable},
	{"Sub", "-", "TransInv", "Add", false, false, 0, isAddable},
	{"Mul", "*", "Scale", "Div", false, true, 1, isNumber},
	{"Div", "/", "ScaleInv", "Mul", false, false, 1, isNumber},
	{"Pow", "Pow", "PowOf", "", true, false, 1, isFloatCmplx},
}

var unaryBinOps = [...]UnaryOp{
	{"Neg", "-", isNumber},
	{"Abs", "Abs", isSignedNumber},
}
