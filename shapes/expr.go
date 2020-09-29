package shapes

import "fmt"

var ()

type Expr interface {
	// Shape | Abstract | Var | E->E | Slice | Axis | Axes | Size
	// IndexOf | SliceOf | TransposeOf | RepeatOf | ConcatOf
	isExpr()

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
func (v Var) freevars() varset              { return varset{v} }
func (v Var) subExprs() []substitutableExpr { return nil }

// Axis represents an axis in doing shape stuff.
type Axis int

func (a Axis) isExpr()                              {}
func (a Axis) apply(ss substitutions) substitutable { return a }
func (a Axis) freevars() varset                     { return nil }
func (a Axis) subExprs() []substitutableExpr        { return nil }

// Axes represents a list of axes.
// Despite being a container type (i.e. an Axis is an Expr),
// it returns nil for Exprs(). This is because we want to treat Axes as a monolithic entity.
type Axes []Axis

func (a Axes) isExpr()                              {}
func (a Axes) Format(s fmt.State, r rune)           { fmt.Fprintf(s, "X%v", axesToInts(a)) }
func (a Axes) apply(ss substitutions) substitutable { return a }
func (a Axes) freevars() varset                     { return nil }
func (a Axes) subExprs() []substitutableExpr        { return nil }

// Size represents a size of a dimension/axis
type Size int

func (s Size) isExpr()                              {}
func (s Size) isSizelike()                          {}
func (s Size) apply(ss substitutions) substitutable { return s }
func (s Size) freevars() varset                     { return nil }
func (s Size) subExprs() []substitutableExpr        { return nil }

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
func (a Arrow) freevars() varset { return arrowToTup(&a).freevars() }

func (a Arrow) subExprs() []substitutableExpr {
	return []substitutableExpr{a.A.(substitutableExpr), a.B.(substitutableExpr)}
}

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
func (i IndexOf) subExprs() []substitutableExpr {
	return []substitutableExpr{i.I, i.A.(substitutableExpr)}
}

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
func (t TransposeOf) subExprs() []substitutableExpr {
	return []substitutableExpr{t.Axes, t.A.(substitutableExpr)}
}

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
func (s SliceOf) subExprs() []substitutableExpr {
	return []substitutableExpr{toSli(s.Slice), s.A.(substitutableExpr)}
}

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
func (c ConcatOf) freevars() varset { return (exprtup{c.A, c.B}).freevars() }
func (c ConcatOf) subExprs() []substitutableExpr {
	return []substitutableExpr{c.Along, c.A.(substitutableExpr), c.B.(substitutableExpr)}
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
func (r RepeatOf) subExprs() []substitutableExpr {
	return []substitutableExpr{r.Along, r.A.(substitutableExpr)}
}

type SubjectTo struct {
	OpType
	A, B Operation
}

func (s SubjectTo) Format(st fmt.State, r rune) {
	fmt.Fprintf(st, "(%v %v %v)", s.A, s.OpType, s.B)
}

func (s SubjectTo) apply(ss substitutions) substitutable {
	return SubjectTo{
		OpType: s.OpType,
		A:      s.A.apply(ss).(Operation),
		B:      s.B.apply(ss).(Operation),
	}
}

func (s SubjectTo) freevars() varset { return append(s.A.freevars(), s.B.freevars()...) }

func (s SubjectTo) subExprs() []substitutableExpr { return []substitutableExpr{s.A, s.B} }

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
func (c Compound) freevars() varset {
	retVal := c.Expr.freevars()
	retVal = append(retVal, c.SubjectTo.freevars()...)
	return retVal
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
