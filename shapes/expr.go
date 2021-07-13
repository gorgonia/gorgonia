package shapes

import (
	"fmt"
)

//go-sumtype:decl Expr

// Expr represents an expression. The following types are Expr:
// 	Shape | Abstract | Arrow | Compound
// 	Var | Size | UnaryOp
// 	IndexOf | TransposeOf | SliceOf | RepeatOf | ConcatOf
//	Sli | Axis | Axes
//
// A compact BNF is as follows:
// 	E := S | A | E → E | (E s.t. X)
//	a | Sz | Π E | Σ E | D E
// 	I n E | T []Ax E | L : E | R Ax n E | C Ax E E
//	: | Ax | []Ax
type Expr interface {
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
func (a Axes) Dims() int                            { return len(a) }
func (a Axes) AsInts() []int                        { return axesToInts(a) }

// Size represents a size of a dimension/axis
type Size int

func (s Size) isExpr()                              {}
func (s Size) isSizelike()                          {}
func (s Size) apply(ss substitutions) substitutable { return s }
func (s Size) freevars() varset                     { return nil }
func (s Size) subExprs() []substitutableExpr        { return nil }

// Size also implements Operation (i.e. it's a Const)

func (s Size) isValid() bool              { return true }
func (s Size) resolveSize() (Size, error) { return s, nil }

// Sizes are a list of sizes.
type Sizes []Size

func (s Sizes) isExpr()                              {}
func (s Sizes) Format(f fmt.State, r rune)           { fmt.Fprintf(f, "Sz%v", sizesToInts(s)) }
func (s Sizes) apply(ss substitutions) substitutable { return s }
func (s Sizes) freevars() varset                     { return nil }
func (s Sizes) subExprs() []substitutableExpr        { return nil }
func (s Sizes) AsInts() []int                        { return sizesToInts(s) }

// complex expressions

// Arrow represents a function of shapes, from A → B.
type Arrow struct {
	A, B Expr
}

func MakeArrow(exprs ...Expr) Arrow {
	if len(exprs) < 2 {
		panic("Expect at least two expressions to make an Arrow")
	}
	a := Arrow{A: exprs[0]}
	if len(exprs) > 2 {
		a.B = MakeArrow(exprs[1:]...)
	} else {
		a.B = exprs[1]
	}
	return a
}

func (a Arrow) isExpr() {}

func (a Arrow) Format(s fmt.State, r rune) {
	if _, ok := a.A.(Arrow); ok {
		fmt.Fprintf(s, "(%v) → %v", a.A, a.B)
		return
	}
	fmt.Fprintf(s, "%v → %v", a.A, a.B)
}

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
