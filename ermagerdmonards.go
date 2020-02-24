package gorgonia

import (
	"github.com/pkg/errors"
)

var (
	_ Result = (*Node)(nil)
	_ Result = (Nodes)(nil)
	_ Result = gErr{}
)

// Result is either a Node or Nodes or error. It's a poor man's sum types and it's not sealed for good reason
type Result interface {
	Input
	Errer
}

// Input is something that can produce both a *Node and Nodes. Returning nil is OK.
type Input interface {
	Node() *Node
	Nodes() Nodes
}

// Errer is an interface that can return an error.
type Errer interface {
	Err() error
}

// Mker is an interface of any Input that can make a new version of itself
type Mker interface {
	Mk(...Input) Input
}

// Lift1  decorates a function with a precheck and post function lifting
func Lift1(fn func(a *Node) (*Node, error)) func(a Input) Result {
	return func(a Input) Result {
		if err := CheckOne(a); err != nil {
			return Err(errors.WithStack(err))
		}
		return TransformResult(a)(fn(a.Node()))
	}
}

// Lift1Axial  decorates a function with a precheck and post function lifting
func Lift1Axial(fn func(a *Node, axes ...int) (*Node, error)) func(a Input, axes ...int) Result {
	return func(a Input, axes ...int) Result {
		if err := CheckOne(a); err != nil {
			return Err(errors.WithStack(err))
		}
		return TransformResult(a)(fn(a.Node(), axes...))
	}
}

// Lift2 decorates a function with a precheck and post function lifting
func Lift2(fn func(a, b *Node) (*Node, error)) func(a, b Input) Result {
	return func(a, b Input) Result {
		if err := CheckOne(a); err != nil {
			return Err(errors.WithStack(err))
		}
		if err := CheckOne(b); err != nil {
			return Err(errors.WithStack(err))
		}
		return TransformResult(a, b)(fn(a.Node(), b.Node()))
	}
}

// Lift2Broadcast decorates a function with a precheck and post function lifting
func Lift2Broadcast(fn func(a, b *Node, pat1, pat2 []byte) (*Node, error)) func(a, b Input, pat1, pat2 []byte) Result {
	return func(a, b Input, pat1, pat2 []byte) Result {
		if err := CheckOne(a); err != nil {
			return Err(errors.WithStack(err))
		}
		if err := CheckOne(b); err != nil {
			return Err(errors.WithStack(err))
		}
		return TransformResult(a, b)(fn(a.Node(), b.Node(), pat1, pat2))
	}
}

// gErr implements Result and error.
type gErr struct{ error }

// Err is a function that returns a gErr. It wraps errors with stack information.
// A gErr implements Result, as well as error.
// This way, the Err() method acts as an unwrapper.
func Err(e error) gErr { return gErr{errors.WithStack(e)} }

func (err gErr) Node() *Node  { return nil }
func (err gErr) Nodes() Nodes { return nil }
func (err gErr) Err() error   { return err.error }

// resultM is a wrapper for Input to create a Result. This is the default Result if an unknown Input was passed in.
type resultM struct{ Input }

func (r resultM) Err() error { return nil }

// LiftResult creates a Result from a Input and error pair.
// If the error is not nil, the Input is discarded.
//
// The usual use case is in a function that returns a `(*Node, error)`.
// e.g LiftResult(Add(a, b))
func LiftResult(a Input, err error) Result {
	if err != nil {
		return Err(err)
	}
	switch at := a.(type) {
	case Result:
		return at
	default:
		return resultM{a}
	}
}

// TransformResult is like LiftResult, but allows for custom data types that fulfil Mker
func TransformResult(ins ...Input) func(a Input, err error) Result {
	return func(a Input, err error) Result {
		if err != nil {
			return Err(err)
		}
		for _, in := range ins {
			if mk, ok := in.(Mker); ok {
				a = mk.Mk(a)
			}
		}
		switch at := a.(type) {
		case Result:
			return at
		default:
			return resultM{a}
		}
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
		if err := CheckOne(xs[i]); err != nil {
			return nil, errors.Wrapf(err, "NodesFromInputs %dth input", i)
		}
		// check if the Input is a *Node
		if xs[i].Node() == nil {
			return nil, errors.Errorf("Input %d is not a *Node", i)
		}
	}

	retVal := make(Nodes, len(xs))
	for i := range xs {
		retVal[i] = xs[i].Node()
	}
	return retVal, nil
}
