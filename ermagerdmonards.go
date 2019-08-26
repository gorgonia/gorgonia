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

// Errer is an interface that can return an error.
type Errer interface {
	Err() error
}

// Lift1  decorates a function with a precheck and post function lifting
func Lift1(fn func(a *Node) (*Node, error)) func(a Input) Result {
	return func(a Input) Result {
		if err := CheckOne(a); err != nil {
			return Err{errors.WithStack(err)}
		}
		return LiftResult(fn(a.Node()))
	}
}

// Lift1Axial;  decorates a function with a precheck and post function lifting
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

// Err implements Result
type Err struct{ E error }

func (err Err) Node() *Node  { return nil }
func (err Err) Nodes() Nodes { return nil }
func (err Err) Err() error   { return err.E }

// resultM is a wrapper for Input to create a Result. This is the default Result if an unknown Input was passed in.
type resultM struct{ Input }

func (r resultM) Err() error { return nil }

func LiftResult(a Input, err error) Result {
	if err != nil {
		return Err{err}
	}
	switch at := a.(type) {
	case Result:
		return at
	default:
		return resultM{a}
	}
}

// CheckOne checks whether an input is an error
func CheckOne(in Input) error {
	if errer, ok := in.(Errer); ok && errer.Err() != nil {
		return errer.Err()
	}
	return nil
}

// NodesFromInputs creates a Nodes from a list of Input.
func NodesFromInputs(xs ...Input) (Nodes, error) {
	for i := range xs {
		if err:= CheckOne(xs[i]); err != nil{
			return nil, errors.Wrapf(err, "NodesFromInputs %dth input",i)
		}
		// check if the Input is a *Node
		if xs[i].Node() == nil {
			return nil, errors.Errorf("Input %d is not a *Node", i)
		}
	}

	retVal := make(Nodes, len(xs))
	for i :=range xs {
		retVal[i] = xs[i].Node()
	}
	return retVal, nil
}
