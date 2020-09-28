package shapes

import "fmt"

var ()

type Expr interface {
	// Shape | Abstract | Var | E->E | (E->E)@E | Slice | Axis | Axes | Size
	// IndexOf | SliceOf | TransposeOf | RepeatOf | ConcatOf
	isExpr()

	// Exprs returns a slice of expressions if it is a container expression
	// (Arrow, App, XXXOf)
	// Otherwise it returns nil
	Exprs() []Expr

	substitutable
}

// Var represents a variable
type Var rune

func (v Var) isSizelike()                {}
func (v Var) isExpr()                    {}
func (v Var) Format(s fmt.State, r rune) { fmt.Fprintf(s, "%c", rune(v)) }
func (v Var) apply(ss substitutions) substitutable {
	if len(ss) == 0 {
		return v
	}
	for _, s := range ss {
		if s.For == v {
			return s.Sub
		}
	}
	return v
}
func (v Var) freevars() varset { return varset{v} }
func (v Var) Exprs() []Expr    { return nil }

// Axis represents an axis in doing shape stuff.
type Axis int

func (a Axis) isExpr()                              {}
func (a Axis) apply(ss substitutions) substitutable { return a }
func (a Axis) freevars() varset                     { return nil }
func (a Axis) Exprs() []Expr                        { return nil }

// Axes represents a list of axes.
// Despite being a container type (i.e. an Axis is an Expr),
// it returns nil for Exprs(). This is because we want to treat Axes as a monolithic entity.
type Axes []Axis

func (a Axes) isExpr()                              {}
func (a Axes) Format(s fmt.State, r rune)           { fmt.Fprintf(s, "X%v", axesToInts(a)) }
func (a Axes) apply(ss substitutions) substitutable { return a }
func (a Axes) freevars() varset                     { return nil }
func (a Axes) Exprs() []Expr                        { return nil }

// Size represents a size of a dimension/axis
type Size int

func (s Size) isExpr()                              {}
func (s Size) isSizelike()                          {}
func (s Size) apply(ss substitutions) substitutable { return s }
func (s Size) freevars() varset                     { return nil }
func (s Size) Exprs() []Expr                        { return nil }

// BinOp represents a binary operation. It is not an expression
type BinOp struct {
	Op OpType // CHECK!
	A  Expr
	B  Expr
}

func (op BinOp) isSizelike() {}
func (op BinOp) isExpr()     {}
func (op BinOp) apply(ss substitutions) substitutable {
	return BinOp{
		Op: op.Op,
		A:  op.A.apply(ss).(Expr),
		B:  op.B.apply(ss).(Expr),
	}
}
func (op BinOp) freevars() varset {
	retVal := op.A.freevars()
	retVal = append(retVal, op.B.freevars()...)
	return unique(retVal)
}

func (op BinOp) Format(s fmt.State, r rune) { fmt.Fprintf(s, "%v %v %v", op.A, op.Op, op.B) }

// UnaryOp represetns a unary operation on a shape expression
type UnaryOp struct {
	Op OpType
	A  Expr
}

func (op UnaryOp) isExpr()                    {}
func (op UnaryOp) Format(s fmt.State, r rune) { fmt.Fprintf(s, "%v %v", op.Op, op.A) }
func (op UnaryOp) apply(ss substitutions) substitutable {
	return UnaryOp{
		Op: op.Op,
		A:  op.A.apply(ss).(Expr),
	}
}
func (op UnaryOp) freevars() varset { return op.A.freevars() }

// complex expressions

// Arrow represents a function of shapes, from A → B.
type Arrow struct {
	A, B Expr
}

func (a Arrow) isExpr() {}

func (a Arrow) Format(s fmt.State, r rune) { fmt.Fprintf(s, "%v → %v", a.A, a.B) }

func (a Arrow) apply(ss substitutions) substitutable {
	return Arrow{
		A: a.A.apply(ss).(Expr),
		B: a.B.apply(ss).(Expr),
	}
}
func (a Arrow) freevars() varset {
	retVal := a.A.freevars()
	retVal = append(retVal, a.B.freevars()...)
	return unique(retVal)
}

func (a Arrow) Exprs() []Expr { return []Expr{a.A, a.B} }

// App represents an application of a Arrow to another expression.
type App struct {
	A Arrow
	B Expr
}

func (a App) isExpr() {}

func (a App) Format(s fmt.State, r rune) { fmt.Fprintf(s, "%v @ %v", a.A, a.B) }

func (a App) apply(ss substitutions) substitutable {
	return App{
		A: a.A.apply(ss).(Arrow),
		B: a.B.apply(ss).(Expr),
	}
}
func (a App) freevars() varset {
	retVal := a.A.freevars()
	retVal = append(retVal, a.B.freevars()...)
	return unique(retVal)
}
func (a App) Exprs() []Expr { return []Expr{a.A, a.B} }

// IndexOf represents a slice operation where the range is 1
type IndexOf struct {
	I Size
	A Expr
}

func (i IndexOf) isExpr()                    {}
func (i IndexOf) Format(s fmt.State, r rune) { fmt.Fprintf(s, "%v[%d]", i.A, i.I) }
func (i IndexOf) apply(ss substitutions) substitutable {
	return IndexOf{
		I: i.I,
		A: i.A.apply(ss).(Expr),
	}
}
func (i IndexOf) freevars() varset { return i.A.freevars() }
func (i IndexOf) Exprs() []Expr    { return []Expr{i.I, i.A} }

// TransposeOf
type TransposeOf struct {
	Axes Axes
	A    Expr
}

func (t TransposeOf) isExpr()                    {}
func (t TransposeOf) Format(s fmt.State, r rune) { fmt.Fprintf(s, "Tr %v %v", t.Axes, t.A) }
func (t TransposeOf) apply(ss substitutions) substitutable {
	return TransposeOf{
		Axes: t.Axes,
		A:    t.A.apply(ss).(Expr),
	}
}
func (t TransposeOf) freevars() varset { return t.A.freevars() }
func (t TransposeOf) Exprs() []Expr    { return []Expr{t.Axes, t.A} }

type SliceOf struct {
	Slice Slice
	A     Expr
}

func (s SliceOf) isExpr()                     {}
func (s SliceOf) Format(st fmt.State, r rune) { fmt.Fprintf(st, "%v%v", s.A, s.Slice) }
func (s SliceOf) apply(ss substitutions) substitutable {
	return SliceOf{
		Slice: s.Slice,
		A:     s.A.apply(ss).(Expr),
	}
}
func (s SliceOf) freevars() varset { return s.A.freevars() }
func (s SliceOf) Exprs() []Expr    { return []Expr{toSli(s.Slice), s.A} }

// Sli is a expression representation of a slice.
//
// We are treating it as a monolithic shape expression
type Sli struct {
	start, end, step int
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
func (s Sli) apply(ss substitutions) substitutable { return s }
func (s Sli) freevars() varset                     { return nil }
func (s Sli) Exprs() []Expr                        { return nil }

func S(start int, others ...int) Sli {
	var end, step int
	if len(opt) > 0 {
		end = opt[0]
	} else {
		end = start + 1
	}

	step = 1
	if len(opt) > 1 {
		step = opt[1]
	} else if end == start+1 {
		step = 0
	}
	return Sli{
		start: start,
		end:   end,
		step:  step,
	}
}
func toSli(s Slice) Sli {
	if ss, ok := s.(Sli); ok {
		return ss
	}
	return Sli{s.Start(), s.End(), s.Step()}
}
func (s Sli) Start() int { return s.start }
func (s Sli) End() int   { return s.end }
func (s Sli) Step() int  { return s.step }

type ConcatOf struct {
	Along Axis
	A, B  Expr
}

func (c ConcatOf) isExpr()                    {}
func (c ConcatOf) Format(s fmt.State, r rune) { fmt.Fprintf(s, "%v :{%d}: %v", c.A, c.Along, c.B) }
func (c ConcatOf) apply(ss substitutions) substitutable {
	return ConcatOf{
		Along: c.Along,
		A:     c.A.apply(ss).(Expr),
		B:     c.B.apply(ss).(Expr),
	}
}
func (c ConcatOf) freevars() varset {
	retVal := c.A.freevars()
	retVal = append(retVal, c.B.freevars()...)
	return unique(retVal)
}

type RepeatOf struct {
	Along   Axis
	Repeats []Size
	A       Expr
}

func (r RepeatOf) isExpr() {}
func (r RepeatOf) Format(s fmt.State, ru rune) {
	fmt.Fprintf(s, "Repeat{%d}{%v} %v", r.Along, r.Repeats, r.A)
}
func (r RepeatOf) apply(ss substitutions) substitutable {
	return RepeatOf{
		Along:   r.Along,
		Repeats: r.Repeats,
		A:       r.A.apply(ss).(Expr),
	}
}
func (r RepeatOf) freevars() varset { return r.A.freevars() }

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
func (c Compound) apply(ss substitutions) substitutable {
	return Compound{
		Expr:      c.Expr.apply(ss).(Expr),
		SubjectTo: c.SubjectTo,
	}
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
