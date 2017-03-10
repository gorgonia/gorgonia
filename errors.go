package gorgonia

import (
	"fmt"
	"runtime"

	"github.com/pkg/errors"
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

// noIncrErr is an error used internally when a Value cannot be incremented
type noIncrErr struct {
	v Value
}

func (noIncrErr) Error() string  { return "increment couldn't be done. Safe op was performed instead" }
func (e noIncrErr) Value() Value { return e.v }

// valueErr is an error used internally for when an error is itself a Valuer
type valueErr struct {
	Valuer

	msg    string
	fnName string
	file   string
	line   int
}

func newValueErr(v Valuer, format string, attrs ...interface{}) error {
	pc, _, _, _ := runtime.Caller(1)
	fn := runtime.FuncForPC(pc)
	file, line := fn.FileLine(pc)

	return valueErr{
		Valuer: v,

		msg:    fmt.Sprintf(format, attrs...),
		fnName: fn.Name(),
		file:   file,
		line:   line,
	}
}

func (err valueErr) Error() string {
	return fmt.Sprintf("ValueError: %v. Happened at %v:%d. Called by: %v", err.msg, err.file, err.line, err.fnName)
}

func (err valueErr) Offender() interface{} { return err.Valuer }

// AutoDiffError is an error which should be passed if the function is not differentiable. This is useful for Op implementations
type AutoDiffError struct{}

func (err AutoDiffError) Error() string { return "AutoDiffError" }

func nyi(what string, implFor interface{}) error {
	return errors.Errorf(nyiFail, what, implFor)
}

func nondiffErr(op Op) error {
	return errors.Errorf("%s is a non-differentiable function", op)
}
