package stdops

import (
	"github.com/chewxy/hm"
	"gorgonia.org/shapes"
)

// internal.go provides the data structures that are useful for the specific implementations in this package.

var twotrues = []bool{true, true}

// binop is a generic data structure meant to be embedded in any binary operations.
// binop has the basic methods of all binary operations. One may elect to think of
// binop as a "partial" definition of a binary operation, to be completed by embedding
// it into other structs.
type binop struct{}

// Arity returns 2. The operation requires two inputs.
func (op binop) Arity() int { return 2 }

// Type returns the operation type of (·) : a → a → a
func (op binop) Type() hm.Type {
	a := hm.TypeVariable('a')
	return hm.NewFnType(a, a, a)
}

// ShapeExpr returns the shape expression of (·) : a → a → a.
func (op binop) ShapeExpr() shapes.Expr {
	a := shapes.Var('a')
	return shapes.MakeArrow(a, a, a)
}

// DiffWRT returns {true, true} for all binops defined in this library.
func (op binop) DiffWRT(inputs int) []bool { return twotrues }

type binopVS struct{ binop }

// Type returns the operation type of (·) : a → b → a
func (op binopVS) Type() hm.Type {
	a := hm.TypeVariable('a')
	b := hm.TypeVariable('b')
	return hm.NewFnType(a, b, a)
}

// ShapeExpr returns the shape expression of (·) : a → () → a.
func (op binopVS) ShapeExpr() shapes.Expr {
	a := shapes.Var('a')
	return shapes.MakeArrow(a, shapes.ScalarShape(), a)
}

type binopSV struct{ binop }

// Type returns the operation type of (·) : a → b → b
func (op binopSV) Type() hm.Type {
	a := hm.TypeVariable('a')
	b := hm.TypeVariable('b')
	return hm.NewFnType(a, b, b)
}

// ShapeExpr returns the shape expression of (·) : () → a → a.
func (op binopSV) ShapeExpr() shapes.Expr {
	a := shapes.Var('a')
	return shapes.MakeArrow(shapes.ScalarShape(), a, a)
}
