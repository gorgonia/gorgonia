package main

type binOp struct {
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

var ariths = []binOp{
	{"add", "Add", "elementwise addition", "+", true},
	{"sub", "Sub", "elementwise subtraction", "-", true},
	{"mul", "Mul", "elementwise multiplciatio=", "*", true},
	{"div", "Div", "elementwise division", "÷", true},
	{"pow", "Pow", "elementwise exponentiation", "^", true},
	{"mod", "Mod", "elementwise mod", "%", false},
}

type binopTest struct {
	binOp
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
	AVV:  "tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))",
	BVV:  "tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{10, 20, 30, 40, 50, 60}))",
	AVV2: "tensor.New(tensor.WithShape(), tensor.WithBacking([]float64{1}))",
	BVV2: "tensor.New(tensor.WithShape(), tensor.WithBacking([]float64{2}))",
	CVV:  "tensor.New(tensor.WithShape(), tensor.WithBacking([]float64{-1}))",

	AVS: "tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))",
	BVS: "tensor.New(tensor.WithShape(), tensor.WithBacking([]float64{100}))",
	CVS: "tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{-1, -1, -1, -1, -1, -1}))",

	ASV: "tensor.New(tensor.WithShape(), tensor.WithBacking([]float64{100}))",
	BSV: "tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))",
	CSV: "tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{-1, -1, -1, -1, -1, -1}))",
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

var cmps = []binOp{
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
	AVV:  "tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))",
	BVV:  "tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{1,2, 3, 40, 50, 60}))",
	AVV2: "tensor.New(tensor.WithShape(), tensor.WithBacking([]float64{1}))",
	BVV2: "tensor.New(tensor.WithShape(), tensor.WithBacking([]float64{2}))",
	CVV:  "tensor.New(tensor.WithShape(), tensor.WithBacking([]bool{false}))",

	AVS: "tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))",
	BVS: "tensor.New(tensor.WithShape(), tensor.WithBacking([]float64{100}))",
	CVS: "tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]bool{false, false, false, false, false, false}))",

	ASV: "tensor.New(tensor.WithShape(), tensor.WithBacking([]float64{100}))",
	BSV: "tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))",
	CSV: "tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]bool{false, false, false, false, false, false}))",
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
	AVV:  "tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))",
	BVV:  "tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{1,2, 3, 40, 50, 60}))",
	AVV2: "tensor.New(tensor.WithShape(), tensor.WithBacking([]float64{1}))",
	BVV2: "tensor.New(tensor.WithShape(), tensor.WithBacking([]float64{2}))",
	CVV:  "tensor.New(tensor.WithShape(), tensor.WithBacking([]float64{0}))",

	AVS: "tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))",
	BVS: "tensor.New(tensor.WithShape(), tensor.WithBacking([]float64{100}))",
	CVS: "tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{0, 0, 0, 0, 0, 0}))",

	ASV: "tensor.New(tensor.WithShape(), tensor.WithBacking([]float64{100}))",
	BSV: "tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))",
	CSV: "tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{0, 0, 0, 0, 0, 0}))",
}

// unary operators
type unop struct {
	// name of the op.
	Name string

	// The tensor.XXX function to call.
	Method string

	// IsDiff indicates if the operation is differentiable.
	IsDiff bool
}
