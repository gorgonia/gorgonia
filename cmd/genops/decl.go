package main

import "slices"

type Op struct {
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

	// InterfaceName is the name of the interface the method fulfils
	InterfaceName string
}

var ariths = []Op{
	{"add", "Add", "elementwise addition", "+", true, "Adder"},
	{"sub", "Sub", "elementwise subtraction", "-", true, "BasicArither"},
	{"mul", "Mul", "elementwise multiplciatio=", "*", true, "BasicArither"},
	{"div", "Div", "elementwise division", "÷", true, "BasicArither"},
	{"pow", "Pow", "elementwise exponentiation", "^", true, "Arither"},
	{"mod", "Mod", "elementwise mod", "%", false, "Arither"},
}

type binopTest struct {
	Op
	binopTestInput
	binopTestResult
	IsCmp        bool
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
		"[]float64{3.0}",
	},
	// sub
	{
		"[]float64{-9, -18, -27, -36, -45, -54}",
		"[]float64{-99, -98, -97, -96, -95, -94}",
		"[]float64{99, 98, 97, 96, 95, 94}",
		"[]float64{-1.0}",
	},
	// mul
	{
		"[]float64{10, 40, 90, 160, 250, 360}",
		"[]float64{100, 200, 300, 400, 500, 600}",
		"[]float64{100, 200, 300, 400, 500, 600}",
		"[]float64{2.0}",
	},
	// div
	{
		"[]float64{0.1, 0.1, 0.1, 0.1, 0.1, 0.1}",
		"[]float64{0.01, 0.02, 0.03, 0.04, 0.05, 0.06}",
		"[]float64{100, 50, 100.0/3.0, 25, 20, 100.0/6.0}",
		"[]float64{0.5}",
	},
	// pow
	{
		"[]float64{1,math.Pow(2, 20), math.Pow(3, 30), math.Pow(4, 40), math.Pow(5,50), math.Pow(6,60)}",
		"[]float64{1,math.Pow(2,100), math.Pow(3, 100), math.Pow(4, 100), math.Pow(5,100), math.Pow(6, 100)}",
		"[]float64{math.Pow(100,1), math.Pow(100, 2), math.Pow(100, 3), math.Pow(100, 4), math.Pow(100,5), math.Pow(100, 6)}",
		"[]float64{1.0}",
	},
	// mod
	{
		"[]float64{1,2,3,4, 5, 6}",
		"[]float64{1,2,3,4,5, 6}",
		"[]float64{0, 0, 1, 0, 0, 4}",
		"[]float64{1.0}",
	},
}

var cmps = []Op{
	{"lt", "Lt", "elementwise less-than", "<", false, "Ord"},
	{"lte", "Lte", "elementwise less-than-or-equal-to", "≤", false, "Ord"},
	{"gt", "Gt", "elementwise greater-than", ">", false, "FullOrd"},
	{"gte", "Gte", "elementwise greater-than-or-equal-to", "≥", false, "FullOrd"},
	{"elEq", "ElEq", "elementwise equal-to", "=", false, "Comparer"},
	{"elNe", "ElNe", "elementwise not-equal-to", "≠", false, "Comparer"},
}

var cmpTestResultsBool = []binopTestResult{
	// lt
	{
		"[]bool{false, false, false, true, true, true}",
		"[]bool{true, true, true, true, true, true}",
		"[]bool{false, false, false, false, false, false}",
		"[]bool{true}",
	},
	// lte
	{
		"[]bool{true, true, true, true, true, true}",
		"[]bool{true, true, true, true, true, true}",
		"[]bool{false, false, false, false, false, false}",
		"[]bool{true}",
	},
	// gt
	{
		"[]bool{false, false, false, false, false, false}",
		"[]bool{false, false, false, false, false, false}",
		"[]bool{true, true, true, true, true, true}",
		"[]bool{false}",
	},

	// gte
	{
		"[]bool{true, true, true, false, false, false}",
		"[]bool{false, false, false, false, false, false}",
		"[]bool{true, true, true, true, true, true}",
		"[]bool{false}",
	},

	// eq
	{
		"[]bool{true, true, true, false, false, false}",
		"[]bool{false, false, false, false, false, false}",
		"[]bool{false, false, false, false, false, false}",
		"[]bool{false}",
	},
	// ne
	{
		"[]bool{false, false, false, true, true, true}",
		"[]bool{true, true, true, true, true, true}",
		"[]bool{true, true, true, true, true, true}",
		"[]bool{true}",
	},
}

var cmpTestInputBool = binopTestInput{
	AVV:  "dense.New[float64](tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))",
	BVV:  "dense.New[float64](tensor.WithShape(2, 3), tensor.WithBacking([]float64{1,2, 3, 40, 50, 60}))",
	AVV2: "dense.New[float64](tensor.WithShape(), tensor.WithBacking([]float64{1}))",
	BVV2: "dense.New[float64](tensor.WithShape(), tensor.WithBacking([]float64{2}))",
	CVV:  "dense.New[bool](tensor.WithShape(), tensor.WithBacking([]bool{false}))",

	AVS: "dense.New[float64](tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))",
	BVS: "dense.New[float64](tensor.WithShape(), tensor.WithBacking([]float64{100}))",
	CVS: "dense.New[bool](tensor.WithShape(2, 3), tensor.WithBacking([]bool{false, false, false, false, false, false}))",

	ASV: "dense.New[float64](tensor.WithShape(), tensor.WithBacking([]float64{100}))",
	BSV: "dense.New[float64](tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))",
	CSV: "dense.New[bool](tensor.WithShape(2, 3), tensor.WithBacking([]bool{false, false, false, false, false, false}))",
}

var cmpTestResultsSame = []binopTestResult{
	// lt
	{
		"[]float64{0, 0, 0, 1, 1, 1}",
		"[]float64{1, 1, 1, 1, 1, 1}",
		"[]float64{0, 0, 0, 0, 0, 0}",
		"[]float64{1.0}",
	},
	// lte
	{
		"[]float64{1, 1, 1, 1, 1, 1}",
		"[]float64{1, 1, 1, 1, 1, 1}",
		"[]float64{0, 0, 0, 0, 0, 0}",
		"[]float64{1.0}",
	},
	// gt
	{
		"[]float64{0, 0, 0, 0, 0, 0}",
		"[]float64{0, 0, 0, 0, 0, 0}",
		"[]float64{1, 1, 1, 1, 1, 1}",
		"[]float64{0.0}",
	},

	// gte
	{
		"[]float64{1, 1, 1, 0, 0, 0}",
		"[]float64{0, 0, 0, 0, 0, 0}",
		"[]float64{1, 1, 1, 1, 1, 1}",
		"[]float64{0.0}",
	},

	// eq
	{
		"[]float64{1, 1, 1, 0, 0, 0}",
		"[]float64{0, 0, 0, 0, 0, 0}",
		"[]float64{0, 0, 0, 0, 0, 0}",
		"[]float64{0.0}",
	},
	// ne
	{
		"[]float64{0, 0, 0, 1, 1, 1}",
		"[]float64{1, 1, 1, 1, 1, 1}",
		"[]float64{1, 1, 1, 1, 1, 1}",
		"[]float64{1.0}",
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

var unops = []Op{
	{"abs", "Abs", "elementwise absolute value", "|·|", false, "Abser"},
	{"sign", "Sign", "elementwise sign", "Sign", false, "Signer"},
	{"ceil", "Ceil", "elementwise ceil", "⌈·⌉", false, "IntRepr"},
	{"floor", "Floor", "elementwise floor", "⌊·⌋", false, "IntRepr"},

	{"sin", "Sin", "elementwise sine", "Sin", true, "Trig"},
	{"cos", "Cos", "elementwise cos", "Cos", true, "Trig"},
	{"exp", "Exp", "elementwise exp", "Exp", true, "ExpLoger"},
	{"ln", "Log", "elementwise ln", "Ln", true, "ExpLoger"},
	{"log2", "Log2", "elementwise log2", "Log2", true, "ExpLoger"},
	{"neg", "Neg", "elementwise negation", "Neg", true, "ExpLoger"},
	{"square", "Square", "elementwise square", "²", true, "Squarer"},
	{"sqrt", "Sqrt", "elementwise square root", "√", true, "Squarter"},
	{"inv", "Inv", "elementwise 1/x", "1/·", true, "Inver"},
	{"invSqrt", "InvSqrt", "elementwise 1/√x", "1/√·", true, "InvSqrter"},

	// numerical stabilization
	{"log1p", "Log1p", "elementwise log1p", "Log1p", true, "ExpLoger"},
	{"expm1", "Expm1", "elementwise expm1", "Expm1", true, "ExpLoger"},

	// activation functions... perhaps move them to NN
	{"cube", "Cube", "elementwise cube", "³", true, "Cuber"},
	{"tanh", "Tanh", "elementwise tanh", "tanh", true, "Tanher"},
}

type unopInterface struct {
	InterfaceName string
	Ops           []Op
}

var unopInterfaces []unopInterface

func init() {
	m := make(map[string]unopInterface)
	for _, op := range unops {
		iface := m[op.InterfaceName]
		iface.InterfaceName = op.InterfaceName
		iface.Ops = append(iface.Ops, op)
		m[op.InterfaceName] = iface
	}

	// let's get them in a sorted order
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	slices.Sort(keys)

	for _, k := range keys {
		unopInterfaces = append(unopInterfaces, m[k])
	}
}

type unoptest struct {
	Input   string
	Correct string
}

type unoptestWithOp struct {
	Op
	unoptest
}

var unopTests = map[string]unoptest{
	// abs
	"Abs": {"[]float64{-1,-2,-3,4,5,6}", "[]float64{1,2,3,4,5,6}"},

	//sign
	"Sign": {"[]float64{-1,-2,-3,4,5,6}", "[]float64{-1,-1,-1,1,1,1}"},

	// ceil
	// floor
	// sin
	// cos

	// exp
	"Exp": {"[]float64{1,2,3,4,5,6}", "[]float64{math.Exp(1), math.Exp(2), math.Exp(3), math.Exp(4), math.Exp(5), math.Exp(6)}"},

	// ln
	"Ln": {"[]float64{1,2,3,4,5,6}", "[]float64{math.Log(1), math.Log(2), math.Log(3), math.Log(4), math.Log(5), math.Log(6)}"},

	// log2
	"Log2": {"[]float64{1,2,3,4,5,6}", "[]float64{math.Log2(1), math.Log2(2), math.Log2(3), math.Log2(4), math.Log2(5), math.Log2(6)}"},

	// neg
	"Neg": {"[]float64{-1,-2,-3,4,5,6}", "[]float64{1,2,3,-4,-5,-6}"},

	// square
	"Square": {"[]float64{-1,-2,-3,4,5,6}", "[]float64{1*1,2*2,3*3,4*4,5*5,6*6}"},

	// sqrt
	"Sqrt": {"[]float64{1,2,3,4,5,6}", "[]float64{math.Sqrt(1), math.Sqrt(2), math.Sqrt(3), math.Sqrt(4), math.Sqrt(5), math.Sqrt(6)}"},

	// inv
	"Inv": {"[]float64{-1,-2,-3,4,5,6}", "[]float64{1/-1,1.0/-2.0,1.0/-3.0,1.0/4.0,1.0/5.0,1.0/6.0}"},

	// invsqrt
	"InvSqrt": {"[]float64{1,2,3,4,5,6}", "[]float64{1/math.Sqrt(1),1/math.Sqrt(2),1/math.Sqrt(3),1/math.Sqrt(4),1/math.Sqrt(5),1/math.Sqrt(6)}"},

	// cube
	"Cube": {"[]float64{-1,-2,-3,4,5,6}", "[]float64{-1,-2*-2*-2,-3*-3*-3,4*4*4,5*5*5,6*6*6}"},

	// tanh
	"Tanh": {"[]float64{-1,-2,-3,4,5,6}", "[]float64{math.Tanh(-1),math.Tanh(-2),math.Tanh(-3),math.Tanh(4),math.Tanh(5),math.Tanh(6)}"},
}
