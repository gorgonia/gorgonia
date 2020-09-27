package shapes

import "fmt"

var ()

type Expr interface {
	// Shape | Abstract | Var | E->E | (E->E)@E | Slice | Axis | Axes | Size
	// IndexOf | SliceOf | TransposeOf | RepeatOf | ConcatOf
	isExpr()
}

type Var rune

func (v Var) isSizelike()                {}
func (v Var) isExpr()                    {}
func (v Var) Format(s fmt.State, r rune) { fmt.Fprintf(s, "%c", rune(v)) }

type Axis int

func (a Axis) isExpr() {}

type Axes []Axis

func (a Axes) isExpr()                    {}
func (a Axes) Format(s fmt.State, r rune) { fmt.Fprintf(s, "X%v", axesToInts(a)) }

type Size int

func (s Size) isExpr()     {}
func (s Size) isSizelike() {}

type BinOp struct {
	A  Expr
	B  Expr
	Op OpType // CHECK!
}

func (op BinOp) isSizelike() {}

func (op BinOp) Format(s fmt.State, r rune) { fmt.Fprintf(s, "%v%v%v", op.A, op.Op, op.B) }

// UnaryOp represetns a unary operation on a shape expression
type UnaryOp struct {
	Op OpType
	A  Expr
}

func (op UnaryOp) isExpr()                    {}
func (op UnaryOp) Format(s fmt.State, r rune) { fmt.Fprintf(s, "%v %v", op.Op, op.A) }

// complex expressions

type Arrow struct {
	A, B Expr
}

func (a Arrow) isExpr() {}

func (a Arrow) Format(s fmt.State, r rune) { fmt.Fprintf(s, "%v → %v", a.A, a.B) }

type App struct {
	A Arrow
	B Expr
}

func (a App) isExpr() {}

func (a App) Format(s fmt.State, r rune) { fmt.Fprintf(s, "%v @ %v", a.A, a.B) }

type IndexOf struct {
	I Size
	A Expr
}

func (i IndexOf) isExpr()                    {}
func (i IndexOf) Format(s fmt.State, r rune) { fmt.Fprintf(s, "%v[%d]", i.A, i.I) }

type TransposeOf struct {
	Axes Axes
	A    Expr
}

func (t TransposeOf) isExpr()                    {}
func (t TransposeOf) Format(s fmt.State, r rune) { fmt.Fprintf(s, "Tr %v %v", t.Axes, t.A) }

type SliceOf struct {
	Slice Slice
	A     Expr
}

func (s SliceOf) isExpr()                     {}
func (s SliceOf) Format(st fmt.State, r rune) { fmt.Fprintf(st, "%v%v", s.A, s.Slice) }

// Sli is a expression representation of a slice.
type Sli struct {
	Start, Stop, Step int
}

func (s Sli) isExpr() {}
func (s Sli) Format(st fmt.State, r rune) {
	fmt.Fprintf(st, "[%d", s.Start)
	if s.Stop-s.Start > 1 {
		fmt.Fprintf(st, ":%d", s.Stop)
	}
	if s.Step > 1 {
		fmt.Fprintf(st, "~:%d", s.Step)
	}
	st.Write([]byte("]"))
}

type ConcatOf struct {
	Along Axis
	A, B  Expr
}

func (c ConcatOf) isExpr()                    {}
func (c ConcatOf) Format(s fmt.State, r rune) { fmt.Fprintf(s, "%v :{%d}: %v", c.A, c.Along, c.B) }

type RepeatOf struct {
	Along   Axis
	Repeats []Size
	A       Expr
}

func (r RepeatOf) isExpr() {}
func (r RepeatOf) Format(s fmt.State, ru rune) {
	fmt.Fprintf(s, "Repeat{%d}{%v} %v", r.Along, r.Repeats, r.A)
}

type SubjectTo struct {
	OpType
	A, B Expr
}

func (s SubjectTo) Format(st fmt.State, r rune) {
	fmt.Fprintf(st, "(%v %v %v)", s.A, s.OpType, s.B)
}

type Compound struct {
	Expr
	SubjectTo
}

func (c Compound) Format(s fmt.State, r rune) {
	fmt.Fprintf(s, "%v s.t. %v", c.Expr, c.SubjectTo)
}

type OpType byte

const (
	// Unary
	Const OpType = iota // K
	Dims
	Prod
	Sum

	// Binary
	Add // +
	Sub // -
	Mul // ×
	Div // ÷

	// Cmp
	Eq
	Ne
	Lt
	Gt
	Lte
	Gte
)

func (o OpType) String() string {
	switch o {
	case Const:
		return "K"
	case Dims:
		return "D"
	case Prod:
		return "Π"
	case Sum:
		return "Σ"
	case Add:
		return "+"
	case Sub:
		return "-"
	case Mul:
		return "×"
	case Div:
		return "÷"
	case Eq:
		return "="
	case Ne:
		return "≠"
	case Lt:
		return "<"
	case Gt:
		return ">"
	case Lte:
		return "≤"
	case Gte:
		return "≥"
	default:
		return "UNFORMATTED OPTYPE"
	}
}

/* Example

MatMul:
(a, b) → (b, c) → (a, c)

is represented as:

Arrow {
	Arrow {
		Abstract{Var('a'), Var('b')},
		Abstract{Var('b'), Var('c')},
	},
	Abstract{Var('a'), Var('c')},
}

Add:

a → a → a

is represented as:

Arrow {
	Arrow {
		Var('a'),
		Var('a'),
	},
	Var('a'),
}


Flatten/Ravel:
a → Πa

is represented as:

Arrow {
	Var('a'),
	UnaryOp {Prod, Var('a')},
}


Sum:

a → ()

is represented as:

Arrow {
	Var('a'),
	Shape{},
}

Transpose:
a → Axes → Tr Axes a

is represented as:

Arrow {
	Arrow{
		Var('a'),
		axes,
	},
	TransposeOf {
		axes,
		Var('a'),
	},
}

Slice:
a → Sli → SliceOf Sli a

if represented as:

Arrow {
	Arrow {
		Var('a'),
		sli,
	},
	SliceOf {
		sli,
		Var('a'),
	}
}


More complicated examples:

Reshape:

a → b → b s.t. (Πa = Πb)

is represented as

Compound {
	Arrow {
		Arrow {
			Var('a'),
			Var('b'),
		},
		Var('b'),
	},
	SubjectTo {
		Eq,
		UnaryOp{Prod, Var('a')},
		UnaryOp{Prod, Var('b')},
	}
}


Sum a matrix columnwise (along axis 1):

(a → Axes→ Tr Axes a) → b s.t. (D b = D a - 1)

is represented as:

Compound {
	Arrow {
		Arrow {
			Arrow {
				Var('a'),
				Axes{1,0},
			},
			TransposeOf {
				Axes{1, 0},
				Var('a')
			},
		},
		Var('b'),
	},
	SubjectTo {
		Eq,
		UnaryOp{Dim, Var('b')},
		BinaryOp {
			Sub,
			UnaryOp {Dim, Var('a')},
			Size(1),
		}
	},
}

*/
