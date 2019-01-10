package engine

import (
	"fmt"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia/internal/value"
	"gorgonia.org/gorgonia/ops"
)

// NoOpError is an error returned when an operation does nothing.
type NoOpError interface {
	NoOp() bool
}

type noopError struct{}

func (e noopError) NoOp() bool    { return true }
func (e noopError) Error() string { return "NoOp" }

// errNoStabilization is an error used internally for when there is no stabilization mechanism is found.
type errNoStabilization interface {
	error
	noStabilization() bool
}

// nostabilizationErr is used internally to communicate that there isn't any stabilization possible
type noStabilizationErr struct{}

func (noStabilizationErr) Error() string         { return "No stabilization mechanism found" }
func (noStabilizationErr) noStabilization() bool { return true }

// noIncrErr is an error used internally when a value.Value cannot be incremented
type noIncrErr struct {
	v value.Value
}

func (noIncrErr) Error() string        { return incrErr }
func (e noIncrErr) Value() value.Value { return e.v }

// oomError represents an Out of tensor.Memory error. It is typically used for CUDA related machine work
type oomError struct {
	res       int64
	allocated int64
}

func (e oomError) Reserved() int64  { return e.res }
func (e oomError) Allocated() int64 { return e.allocated }
func (e oomError) Error() string    { return fmt.Sprintf("allocated/reserved: %v/%v", e.allocated, e.res) }

// AutoDiffError is an error which should be passed if the function is not differentiable. This is useful for ops.Op implementations
type AutoDiffError struct{}

func (err AutoDiffError) Error() string { return "AutoDiffError" }

// vmContextualError is an error that is used to wrap errors that arise from the VM
type vmContextualError struct {
	error
	node  *Node // which node was it processing
	instr int   // what instruction ID it was
}

func (err vmContextualError) Node() *Node        { return err.node }
func (err vmContextualError) Value() value.Value { return err.node.Value() }
func (err vmContextualError) InstructionID() int { return err.instr }

func nyi(what string, implFor interface{}) error {
	return errors.Errorf(nyiFail, what, implFor)
}

func nondiffErr(op ops.Op) error {
	return errors.Errorf("%s is a non-differentiable function", op)
}

// checkErrSetDeriv sets the deriv if the error is a value.Valuer. Helper function for linalg operations
func checkErrSetDeriv(err error, dv *value.DualValue) error {
	if ver, ok := err.(value.Valuer); ok {
		return dv.SetDeriv(ver.Value())
	}
	return err
}

// SymDiffError provides the context at which an error occured
type SymDiffError struct {
	nodes   Nodes
	single  *Node
	grad    *Node
	gradMap map[*Node]Nodes
	err     error
}

func (err SymDiffError) Error() string          { return err.err.Error() }
func (err SymDiffError) Nodes() Nodes           { return err.nodes }
func (err SymDiffError) Node() *Node            { return err.single }
func (err SymDiffError) Grads() map[*Node]Nodes { return err.gradMap }
func (err SymDiffError) Grad() *Node            { return err.grad }
