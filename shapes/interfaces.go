package shapes

var (
	_ Shapelike = Abstract{}
	_ Shapelike = Shape{}
)

//go-sumtype:decl Sizelike

type Shapelike interface {
	Dims() int
	TotalSize() int // needed?
	DimSize(dim int) (Sizelike, error)
	T(axes ...Axis) (newShape Shapelike, err error)
	S(slices ...Slice) (newShape Shapelike, err error)
	Repeat(axis Axis, repeats ...int) (newShape Shapelike, finalRepeats []int, size int, err error)
	Concat(axis Axis, others ...Shapelike) (newShape Shapelike, err error)
}

type intslike interface {
	AsInts() []int
}

// Shaper is anything that can return a Shape.
type Shaper interface {
	Shape() Shape
}

type Exprer interface {
	Shape() Expr
}

var (
	_ Sizelike = Size(0)
	_ Sizelike = Var('a')
	_ Sizelike = BinOp{}
	_ Sizelike = UnaryOp{}
)

// Sizelike represents something that can go into a Abstract. The following types are Sizelike:
// 	Size | Var | BinOp | UnaryOp
type Sizelike interface {
	// Size, Var, BinOp
	isSizelike()
}

type Conser interface {
	Cons(Conser) Conser
	isConser()
}

// substitutable is anything that can apply a list of subsitution and then return a substitutable.
//
// The following implements substitutable:
//
// Exprs:
//	Shape | Abstract | Arrow | Compound
//	Var | Size | UnaryOp
//	IndexOf | TransposeOf | SliceOf | RepeaOf | ConcatOf
//	Sli | Axis | Axes
//
// Constraints:
//	exprConstraint | constraints | SubjectTo
//
// Operations:
//	BinOp
//
// Compound:
//	Compound
type substitutable interface {
	apply(substitutions) substitutable
	freevars() varset // set of free variables
}

// same as substitutable, except doesn't apply to internal constraints (exprConstraint and constraints)
type substitutableExpr interface {
	substitutable
	subExprs() []substitutableExpr
}

// Operation represents an operation (BinOp or UnaryOp)
type Operation interface {
	isValid() bool
	substitutableExpr
}

type boolOp interface {
	Operation
	resolveBool() (bool, error)
}

type sizeOp interface {
	Operation
	resolveSize() (Size, error)
}

//  resolver is anything that can resolve an expression
//
// e.g. "built-in" unary terms like TransposeOf, ConcatOf, SliceOf, RepeatOf
type resolver interface {
	resolve() (Expr, error)
}
