package main

import "reflect"

type BinOp struct {
	Name      string
	MixedName string
	Symbol    string

	VecTemplate   string
	MixedTemplate string
	Commutative   bool
	Identity      int                     // yes it's an int
	Is            func(reflect.Kind) bool // type class

	Specialized    []func(reflect.Kind) bool // generate code using specialized
	Specialization []string                  // templates
}

type UnaryOp struct {
	Name     string
	Template string
	Is       func(reflect.Kind) bool // type class
}

var arithBinOps = [...]BinOp{
	{"Add", "Trans", "+",
		"{{$left}} += {{$right}}", "{{$left}} += {{$right}}", false, true, 0, isAddable},
	{"Sub", "TransInv", "-",
		"{{$left}} -= {{$right}}", "{{$left}} -= {{$right}}", false, false, 0, isAddable},
	{"Mul", "Scale", "*",
		"{{$left}} *= {{$right}}", "{{$left}} *= {{$right}}", false, true, 1, isNumber},
	{"Div", "ScaleInv", "/",
		"{{$left}} /= {{$right}}", "{{$left}} /= {{$right}}", false, false, 1, isNumber},
	{"Pow", "PowOf", "{{.mathPkg}}Pow",
		"{{$left}} = math.Pow({{$right0}}, {{$right1}})", "{{$left}} = math.Pow({{$right0}}, {{$right1}})", true, false, 1, isFloatCmplx},

	// {"Mod", "%", "Mod_", "", false, false, 0, isNumber},
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
