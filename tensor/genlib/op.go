package main

import (
	"fmt"
	"reflect"
)

type BinOp struct {
	Name   string
	Symbol string
	IsFunc bool

	Template    string
	Commutative bool
	Identity    int                     // yes it's an int
	Is          func(reflect.Kind) bool // type class
	Check       func(reflect.Kind) string
}

func (op BinOp) BuildVV(t reflect.Kind) string {
	if op.Name == "Mod" {
		goto def
	}

	if isFloat(t) {
		return fmt.Printf("return %s.%s(a, b)", vecPkg(t), op.Name)
	}
def:
	if op.IsFunc {
		return fmt.Sprintf(funcLoop, op.RangeableVV(), op.RangeableVV(), op.Symbol, short(t), op.LeftVV(), op.RightVV())
	}
	return fmt.Sprintf(opLoop, op.RangeableVV(), op.RangeableVV(), op.Symbol, op.RightVV())
}

func (op BinOp) RangeableVV() string { return "at" }
func (op BinOp) LeftVV() string      { return "at[i]" }
func (op BinOp) RightVV() string     { return "bt[i]" }
func (op BinOp) RangeableSV() string { return "bt" }
func (op BinOp) LeftSV() string      { return "bt[i]" }
func (op BinOp) RightSV() string     { return "at" }
func (op BinOp) RangeableVS() string { return "bt" }
func (op BinOp) LeftVS() string      { return "at[i]" }
func (op BinOp) RightVS() string     { return "bt" }

type UnaryOp struct {
	Name     string
	Template string
	Is       func(reflect.Kind) bool // type class
}

var arithBinOps = [...]BinOp{
	{"Add", "+",
		"{{$left}} += {{$right}}", false, true, 0, isAddable},
	{"Sub", "-",
		"{{$left}} -= {{$right}}", false, false, 0, isAddable},
	{"Mul", "*",
		"{{$left}} *= {{$right}}", false, true, 1, isNumber},
	{"Div", "/",
		"{{$left}} /= {{$right}}", false, false, 1, isNumber},
	{"Pow", "{{.mathPkg}}Pow",
		"{{$left}} = {{.mathPkg}}.Pow({{$right}})", true, false, 1, isFloatCmplx},

	{"Mod", "%", "{{$left}} %= {{$right}}", false, false, 0, isNumber},
}

var unaryBinOps = [...]UnaryOp{
	{"Abs", "Abs", isSignedNumber},
	{"Sign", "Sign", isSignedNumber},
	{"Exp", "Exp", isFloat},
	{"Log", "Log", isFloat},
	{"Log2", "Log2", isFloat},
	{"Log10", "Log10", isFloat},
	{"Sqrt", "Sqrt", isFloat},
	{"Cbrt", "Cbrt", isFloat},

	// {"Neg", "-", isNumber},
	// {"Inv", "1/", isNumber},
	// {"Square", "x*x", isNumber},
}

const opLoop = `for i := range %s {
	%s[i] %s= %s
}
`

const opIncrLoop = `for i := range %s {
	%s[i] += %s[i] %s %s
}
`

const funcLoop = `for i := range %s {
	%s[i] = %s%s(%s, %s)
}
`
