package gorgonia

import (
	"github.com/pkg/errors"
)

var (
	_ Result = (*Node)(nil)
	_ Result = (Nodes)(nil)
	_ Result = Err{}
)

// Result is either a Node or Nodes or error. It's a poor man's sum types and it's not sealed for good reason
type Result interface {
	Node() *Node
	Nodes() Nodes
	Err() error
}

// Input is either a Node or Nodes
type Input interface {
	Node() *Node
	Nodes() Nodes
}

type Errer interface {
	Err() error
}

// Lift1  decorates a function with a prechecl and post function lifting
func Lift1(fn func(a *Node) (*Node, error)) func(a Input) Result {
	return func(a Input) Result {
		if err := CheckOne(a); err != nil {
			return Err{errors.WithStack(err)}
		}
		return LiftResult(fn(a.Node()))
	}
}

// Lift1Axoa;  decorates a function with a prechecl and post function lifting
func Lift1Axial(fn func(a *Node, axes ...int) (*Node, error)) func(a Input, axes ...int) Result {
	return func(a Input, axes ...int) Result {
		if err := CheckOne(a); err != nil {
			return Err{errors.WithStack(err)}
		}
		return LiftResult(fn(a.Node(), axes...))
	}
}

// Lift2  decorates a function with a prechecl and post function lifting
func Lift2(fn func(a, b *Node) (*Node, error)) func(a, b Input) Result {
	return func(a, b Input) Result {
		if err := CheckOne(a); err != nil {
			return Err{errors.WithStack(err)}
		}
		if err := CheckOne(b); err != nil {
			return Err{errors.WithStack(err)}
		}
		return LiftResult(fn(a.Node(), b.Node()))
	}
}

// Lift2Broadcast  decorates a function with a prechecl and post function lifting
func Lift2Broadcast(fn func(a, b *Node, pat1, pat2 []byte) (*Node, error)) func(a, b Input, pat1, pat2 []byte) Result {
	return func(a, b Input, pat1, pat2 []byte) Result {
		if err := CheckOne(a); err != nil {
			return Err{errors.WithStack(err)}
		}
		if err := CheckOne(b); err != nil {
			return Err{errors.WithStack(err)}
		}
		return LiftResult(fn(a.Node(), b.Node(), pat1, pat2))
	}
}

// Do1 runs a precheck before performing a unary operation
func Do1(fn func(a *Node) (*Node, error), a Input) Result {
	if err := CheckOne(a); err != nil {
		return Err{errors.WithStack(err)}
	}
	return LiftResult(fn(a.Node()))
}

// Do1Axial runs a precheck before performing a unary operation with an axial option
func Do1Axial(fn func(a *Node, axes ...int) (*Node, error), a *Node, axes ...int) Result {
	if err := CheckOne(a); err != nil {
		return Err{errors.WithStack(err)}
	}
	return LiftResult(fn(a.Node(), axes...))
}

// Do2 runs a pre-check before performing a binary operation
func Do2(fn func(a, b *Node) (*Node, error), a, b Input) Result {
	if err := CheckOne(a); err != nil {
		return Err{errors.WithStack(err)}
	}
	if err := CheckOne(b); err != nil {
		return Err{errors.WithStack(err)}
	}
	return LiftResult(fn(a.Node(), b.Node()))
}

// Do2Broadcast runs a pre-check before performing a broadcast binop
func Do2Broadcast(fn func(a, b *Node, pat1, pat2 []byte) (*Node, error), a, b Input, pat1, pat2 []byte) Result {
	if err := CheckOne(a); err != nil {
		return Err{errors.WithStack(err)}
	}
	if err := CheckOne(b); err != nil {
		return Err{errors.WithStack(err)}
	}
	return LiftResult(fn(a.Node(), b.Node(), pat1, pat2))
}

// Err implements Result
type Err struct{ E error }

func (err Err) Node() *Node  { return nil }
func (err Err) Nodes() Nodes { return nil }
func (err Err) Err() error   { return err.E }

func LiftResult(a *Node, err error) Result {
	if err != nil {
		return Err{err}
	}
	return a
}

// CheckOne checks whether an input is an error
func CheckOne(in Input) error {
	if errer, ok := in.(Errer); ok && errer.Err() != nil {
		return errer.Err()
	}
	return nil
}
