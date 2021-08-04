package main

type binOp struct {
	Name      string
	Method    string
	CommentOp string
	Symbol    string
}

var ariths = []binOp{
	{"Add", "Add", "elementwise addition", "+"},
	{"Sub", "Sub", "elementwise subtraction", "-"},
	{"Mul", "Mul", "elementwise multiplciatio=", "*"},
	{"Div", "Div", "elementwise division", "÷"},
	{"Pow", "Pow", "elementwise exponentiation", "^"},
	{"Mod", "Mod", "elementwise mod", "%"},
}

var cmps = []binOp{}
