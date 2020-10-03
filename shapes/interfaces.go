package shapes

var (
	_ Shapelike = Abstract{}
	_ Shapelike = Shape{}
)

type Shapelike interface {
	Dims() int
	TotalSize() int // needed?
	DimSize(dim int) (Sizelike, error)
	T(axes ...Axis) (newShape Shapelike, err error)
	S(slices ...Slice) (newShape Shapelike, err error)
	Repeat(axis Axis, repeats ...int) (newShape Shapelike, finalRepeats []int, size int, err error)
	Concat(axis Axis, others ...Shapelike) (newShape Shapelike, err error)
}

var (
	_ Sizelike = Size(0)
	_ Sizelike = Var('a')
	_ Sizelike = BinOp{}
	_ Sizelike = UnaryOp{}
)

type Sizelike interface {
	// Size, Var, BinOp
	isSizelike()
}

type Conser interface {
	Cons(Conser) Conser
	isConser()
}

type substitutable interface {
	apply(substitutions) substitutable
	freevars() varset // set of free variables
}

// Expr | BinOp
type substitutableExpr interface {
	substitutable
	subExprs() []substitutableExpr
}

// Operation represents an operation (BinOp or UnaryOp)
type Operation interface {
	isValid() bool
	resolveSize() (Size, error)
	substitutableExpr
}

//  resolver is anything that can resolve an expression
//
// e.g. "built-in" unary terms like TransposeOf, ConcatOf, SliceOf, RepeatOf
type resolver interface {
	resolve() (Expr, error)
}
