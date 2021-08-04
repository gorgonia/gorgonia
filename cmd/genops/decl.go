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

var arithTest = []struct {
	binOp
	Correct       string
	CorrectVS     string
	CorrectSV     string
	CorrectScalar string
}{
	{binOp{"Add", "Add", "elementwise addition", "+"},
		"[]float64{11, 22, 33, 44, 55, 66}",
		"[]float64{101, 102, 103, 104, 105, 106}",
		"[]float64{101, 102, 103, 104, 105, 106}",
		"3.0"},
	{binOp{"Sub", "Sub", "elementwise subtraction", "-"},
		"[]float64{-9, -18, -27, -36, -45, -54}",
		"[]float64{-99, -98, -97, -96, -95, -94}",
		"[]float64{99, 98, 97, 96, 95, 94}",
		"-1.0"},
	{binOp{"Mul", "Mul", "elementwise multiplciatio=", "*"},
		"[]float64{10, 40, 90, 160, 250, 360}",
		"[]float64{100, 200, 300, 400, 500, 600}",
		"[]float64{100, 200, 300, 400, 500, 600}",
		"2.0"},
	{binOp{"Div", "Div", "elementwise division", "÷"},
		"[]float64{0.1, 0.1, 0.1, 0.1, 0.1, 0.1}",
		"[]float64{0.01, 0.02, 0.03, 0.04, 0.05, 0.06}",
		"[]float64{100, 50, 100.0/3.0, 25, 20, 100.0/6.0}",
		"0.5"},
	{binOp{"Pow", "Pow", "elementwise exponentiation", "^"},
		"[]float64{1,math.Pow(2, 20), math.Pow(3, 30), math.Pow(4, 40), math.Pow(5,50), math.Pow(6,60)}",
		"[]float64{1,math.Pow(2,100), math.Pow(3, 100), math.Pow(4, 100), math.Pow(5,100), math.Pow(6, 100)}",
		"[]float64{math.Pow(100,1), math.Pow(100, 2), math.Pow(100, 3), math.Pow(100, 4), math.Pow(100,5), math.Pow(100, 6)}",
		"1.0"},
	{binOp{"Mod", "Mod", "elementwise mod", "%"},
		"[]float64{1,2,3,4, 5, 6}",
		"[]float64{1,2,3,4,5, 6}",
		"[]float64{0, 0, 1, 0, 0, 4}",
		"1.0"},
}

var cmps = []binOp{}
