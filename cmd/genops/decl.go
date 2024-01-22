package main

type op struct {
	// Name of the Op. It will be suffixed with -VV, -VS, -SV.
	Name string

	// Name of the tensor.XXX function to be called.
	Method string

	// Comment on what the binop actually does.
	CommentOp string

	// Symbol for the bin op.
	Symbol string

	// IsDiff denotes whether the operation is differentiable.
	IsDiff bool
}

var ariths = []op{
	{"add", "Add", "elementwise addition", "+", true},
	{"sub", "Sub", "elementwise subtraction", "-", true},
	{"mul", "Mul", "elementwise multiplciatio=", "*", true},
	{"div", "Div", "elementwise division", "÷", true},
	{"pow", "Pow", "elementwise exponentiation", "^", true},
	{"mod", "Mod", "elementwise mod", "%", false},
}

type binopTest struct {
	op
	binopTestInput
	binopTestResult
	IsCmpRetTrue bool // to generate tests for AsSameType()
}
type binopTestInput struct {
	AVV, BVV        string
	AVV2, BVV2, CVV string
	AVS, BVS, CVS   string
	ASV, BSV, CSV   string
}
type binopTestResult struct {
	Correct       string
	CorrectVS     string
	CorrectSV     string
	CorrectScalar string
}

var arithTestInput = binopTestInput{
	AVV:  "dense.New[float64](tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))",
	BVV:  "dense.New[float64](tensor.WithShape(2, 3), tensor.WithBacking([]float64{10, 20, 30, 40, 50, 60}))",
	AVV2: "dense.New[float64](tensor.WithShape(), tensor.WithBacking([]float64{1}))",
	BVV2: "dense.New[float64](tensor.WithShape(), tensor.WithBacking([]float64{2}))",
	CVV:  "dense.New[float64](tensor.WithShape(), tensor.WithBacking([]float64{-1}))",

	AVS: "dense.New[float64](tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))",
	BVS: "dense.New[float64](tensor.WithShape(), tensor.WithBacking([]float64{100}))",
	CVS: "dense.New[float64](tensor.WithShape(2, 3), tensor.WithBacking([]float64{-1, -1, -1, -1, -1, -1}))",

	ASV: "dense.New[float64](tensor.WithShape(), tensor.WithBacking([]float64{100}))",
	BSV: "dense.New[float64](tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))",
	CSV: "dense.New[float64](tensor.WithShape(2, 3), tensor.WithBacking([]float64{-1, -1, -1, -1, -1, -1}))",
}

var arithTestResults = []binopTestResult{
	// add
	{
		"[]float64{11, 22, 33, 44, 55, 66}",
		"[]float64{101, 102, 103, 104, 105, 106}",
		"[]float64{101, 102, 103, 104, 105, 106}",
		"3.0",
	},
	// sub
	{
		"[]float64{-9, -18, -27, -36, -45, -54}",
		"[]float64{-99, -98, -97, -96, -95, -94}",
		"[]float64{99, 98, 97, 96, 95, 94}",
		"-1.0",
	},
	// mul
	{
		"[]float64{10, 40, 90, 160, 250, 360}",
		"[]float64{100, 200, 300, 400, 500, 600}",
		"[]float64{100, 200, 300, 400, 500, 600}",
		"2.0",
	},
	// div
	{
		"[]float64{0.1, 0.1, 0.1, 0.1, 0.1, 0.1}",
		"[]float64{0.01, 0.02, 0.03, 0.04, 0.05, 0.06}",
		"[]float64{100, 50, 100.0/3.0, 25, 20, 100.0/6.0}",
		"0.5",
	},
	// pow
	{
		"[]float64{1,math.Pow(2, 20), math.Pow(3, 30), math.Pow(4, 40), math.Pow(5,50), math.Pow(6,60)}",
		"[]float64{1,math.Pow(2,100), math.Pow(3, 100), math.Pow(4, 100), math.Pow(5,100), math.Pow(6, 100)}",
		"[]float64{math.Pow(100,1), math.Pow(100, 2), math.Pow(100, 3), math.Pow(100, 4), math.Pow(100,5), math.Pow(100, 6)}",
		"1.0",
	},
	// mod
	{
		"[]float64{1,2,3,4, 5, 6}",
		"[]float64{1,2,3,4,5, 6}",
		"[]float64{0, 0, 1, 0, 0, 4}",
		"1.0",
	},
}

var cmps = []op{
	{"lt", "Lt", "elementwise less-than", "<", false},
	{"lte", "Lte", "elementwise less-than-or-equal-to", "≤", false},
	{"gt", "Gt", "elementwise greater-than", ">", false},
	{"gte", "Gte", "elementwise greater-than-or-equal-to", "≥", false},
	{"elEq", "ElEq", "elementwise equal-to", "=", false},
	{"elNe", "ElNe", "elementwise not-equal-to", "≠", false},
}

var cmpTestResultsBool = []binopTestResult{
	// lt
	{
		"[]bool{false, false, false, true, true, true}",
		"[]bool{true, true, true, true, true, true}",
		"[]bool{false, false, false, false, false, false}",
		"true",
	},
	// lte
	{
		"[]bool{true, true, true, true, true, true}",
		"[]bool{true, true, true, true, true, true}",
		"[]bool{false, false, false, false, false, false}",
		"true",
	},
	// gt
	{
		"[]bool{false, false, false, false, false, false}",
		"[]bool{false, false, false, false, false, false}",
		"[]bool{true, true, true, true, true, true}",
		"false",
	},

	// gte
	{
		"[]bool{true, true, true, false, false, false}",
		"[]bool{false, false, false, false, false, false}",
		"[]bool{true, true, true, true, true, true}",
		"false",
	},

	// eq
	{
		"[]bool{true, true, true, false, false, false}",
		"[]bool{false, false, false, false, false, false}",
		"[]bool{false, false, false, false, false, false}",
		"false",
	},
	// ne
	{
		"[]bool{false, false, false, true, true, true}",
		"[]bool{true, true, true, true, true, true}",
		"[]bool{true, true, true, true, true, true}",
		"true",
	},
}

var cmpTestInputBool = binopTestInput{
	AVV:  "dense.New[float64](tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))",
	BVV:  "dense.New[float64](tensor.WithShape(2, 3), tensor.WithBacking([]float64{1,2, 3, 40, 50, 60}))",
	AVV2: "dense.New[float64](tensor.WithShape(), tensor.WithBacking([]float64{1}))",
	BVV2: "dense.New[float64](tensor.WithShape(), tensor.WithBacking([]float64{2}))",
	CVV:  "dense.New[float64](tensor.WithShape(), tensor.WithBacking([]bool{false}))",

	AVS: "dense.New[float64](tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))",
	BVS: "dense.New[float64](tensor.WithShape(), tensor.WithBacking([]float64{100}))",
	CVS: "dense.New[float64](tensor.WithShape(2, 3), tensor.WithBacking([]bool{false, false, false, false, false, false}))",

	ASV: "dense.New[float64](tensor.WithShape(), tensor.WithBacking([]float64{100}))",
	BSV: "dense.New[float64](tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))",
	CSV: "dense.New[float64](tensor.WithShape(2, 3), tensor.WithBacking([]bool{false, false, false, false, false, false}))",
}

var cmpTestResultsSame = []binopTestResult{
	// lt
	{
		"[]float64{0, 0, 0, 1, 1, 1}",
		"[]float64{1, 1, 1, 1, 1, 1}",
		"[]float64{0, 0, 0, 0, 0, 0}",
		"1.0",
	},
	// lte
	{
		"[]float64{1, 1, 1, 1, 1, 1}",
		"[]float64{1, 1, 1, 1, 1, 1}",
		"[]float64{0, 0, 0, 0, 0, 0}",
		"1.0",
	},
	// gt
	{
		"[]float64{0, 0, 0, 0, 0, 0}",
		"[]float64{0, 0, 0, 0, 0, 0}",
		"[]float64{1, 1, 1, 1, 1, 1}",
		"0.0",
	},

	// gte
	{
		"[]float64{1, 1, 1, 0, 0, 0}",
		"[]float64{0, 0, 0, 0, 0, 0}",
		"[]float64{1, 1, 1, 1, 1, 1}",
		"0.0",
	},

	// eq
	{
		"[]float64{1, 1, 1, 0, 0, 0}",
		"[]float64{0, 0, 0, 0, 0, 0}",
		"[]float64{0, 0, 0, 0, 0, 0}",
		"0.0",
	},
	// ne
	{
		"[]float64{0, 0, 0, 1, 1, 1}",
		"[]float64{1, 1, 1, 1, 1, 1}",
		"[]float64{1, 1, 1, 1, 1, 1}",
		"1.0",
	},
}

var cmpTestInputSame = binopTestInput{
	AVV:  "dense.New[float64](tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))",
	BVV:  "dense.New[float64](tensor.WithShape(2, 3), tensor.WithBacking([]float64{1,2, 3, 40, 50, 60}))",
	AVV2: "dense.New[float64](tensor.WithShape(), tensor.WithBacking([]float64{1}))",
	BVV2: "dense.New[float64](tensor.WithShape(), tensor.WithBacking([]float64{2}))",
	CVV:  "dense.New[float64](tensor.WithShape(), tensor.WithBacking([]float64{0}))",

	AVS: "dense.New[float64](tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))",
	BVS: "dense.New[float64](tensor.WithShape(), tensor.WithBacking([]float64{100}))",
	CVS: "dense.New[float64](tensor.WithShape(2, 3), tensor.WithBacking([]float64{0, 0, 0, 0, 0, 0}))",

	ASV: "dense.New[float64](tensor.WithShape(), tensor.WithBacking([]float64{100}))",
	BSV: "dense.New[float64](tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))",
	CSV: "dense.New[float64](tensor.WithShape(2, 3), tensor.WithBacking([]float64{0, 0, 0, 0, 0, 0}))",
}

var unops = []op{
	{"abs", "Abs", "elementwise absolute value", "|·|", false},
	{"sign", "Sign", "elementwise sign", "Sign", false},
	//{"ceil", "Ceil", "elementwise ceil", "⌈·⌉", false},
	//{"floor", "Floor", "elementwise floor", "⌊·⌋", false},

	//{"sin", "Sin", "elementwise sine", "Sin", true},
	//{"cos", "Cos", "elementwise cos", "Cos", true},
	{"exp", "Exp", "elementwise exp", "Exp", true},
	{"ln", "Log", "elementwise ln", "Ln", true},
	{"log2", "Log2", "elementwise log2", "Log2", true},
	{"neg", "Neg", "elementwise negation", "Neg", true},
	{"square", "Square", "elementwise square", "²", true},
	{"sqrt", "Sqrt", "elementwise square root", "√", true},
	{"inv", "Inv", "elementwise 1/x", "1/·", true},
	{"invSqrt", "InvSqrt", "elementwise 1/√x", "1/√·", true},

	// numerical stabilization
	//{"log1p", "Log1p", "elementwise log1p", "Log1p", true},
	//{"expm1", "Expm1", "elementwise expm1", "Expm1", true},

	// activation functions... perhaps move them to NN
	{"cube", "Cube", "elementwise cube", "³", true},
	{"tanh", "Tanh", "elementwise tanh", "tanh", true},
}

type unoptest struct {
	Input   string
	Correct string
}

type unoptestWithOp struct {
	op
	unoptest
}

var unopTests = []unoptest{
	// abs
	{"[]float64{-1,-2,-3,4,5,6}", "[]float64{1,2,3,4,5,6}"},

	//sign
	{"[]float64{-1,-2,-3,4,5,6}", "[]float64{-1,-1,-1,1,1,1}"},

	// ceil
	// floor
	// sin
	// cos

	// exp
	{"[]float64{1,2,3,4,5,6}", "[]float64{math.Exp(1), math.Exp(2), math.Exp(3), math.Exp(4), math.Exp(5), math.Exp(6)}"},

	// ln
	{"[]float64{1,2,3,4,5,6}", "[]float64{math.Log(1), math.Log(2), math.Log(3), math.Log(4), math.Log(5), math.Log(6)}"},

	// log2
	{"[]float64{1,2,3,4,5,6}", "[]float64{math.Log2(1), math.Log2(2), math.Log2(3), math.Log2(4), math.Log2(5), math.Log2(6)}"},

	// neg
	{"[]float64{-1,-2,-3,4,5,6}", "[]float64{1,2,3,-4,-5,-6}"},

	// square
	{"[]float64{-1,-2,-3,4,5,6}", "[]float64{1*1,2*2,3*3,4*4,5*5,6*6}"},

	// sqrt
	{"[]float64{1,2,3,4,5,6}", "[]float64{math.Sqrt(1), math.Sqrt(2), math.Sqrt(3), math.Sqrt(4), math.Sqrt(5), math.Sqrt(6)}"},

	// inv
	{"[]float64{-1,-2,-3,4,5,6}", "[]float64{1/-1,1.0/-2.0,1.0/-3.0,1.0/4.0,1.0/5.0,1.0/6.0}"},

	// invsqrt
	{"[]float64{1,2,3,4,5,6}", "[]float64{1/math.Sqrt(1),1/math.Sqrt(2),1/math.Sqrt(3),1/math.Sqrt(4),1/math.Sqrt(5),1/math.Sqrt(6)}"},

	// cube
	{"[]float64{-1,-2,-3,4,5,6}", "[]float64{-1,-2*-2*-2,-3*-3*-3,4*4*4,5*5*5,6*6*6}"},

	// tanh
	{"[]float64{-1,-2,-3,4,5,6}", "[]float64{math.Tanh(-1),math.Tanh(-2),math.Tanh(-3),math.Tanh(4),math.Tanh(5),math.Tanh(6)}"},
}
