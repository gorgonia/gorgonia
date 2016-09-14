package gorgonia

import (
	"fmt"
	"runtime"
)

type errorTyper interface {
	ErrorType() errorType
}

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

//go:generate stringer -type=errorType
type errorType byte

const (
	typeError         errorType = iota // implementation does not handle this type
	NotYetImplemented                  // not yet implemented

	// PEBKAC errors. User who uses this lib made some boo-boos
	TypeError
	ShapeError
	CompileError
	GraphError
	SymbDiffError
	AutoDiffError
	RuntimeError
)

type Error struct {
	errorType
	msg string

	fnName string
	file   string
	line   int
}

func (e Error) Error() string {
	return fmt.Sprintf("%s : %s | %s:%s:%d", e.errorType, e.msg, e.file, e.fnName, e.line)
}

func (e Error) ErrorType() errorType {
	return e.errorType
}

func NewError(t errorType, format string, attrs ...interface{}) error {
	pc, _, _, _ := runtime.Caller(1)
	fn := runtime.FuncForPC(pc)
	file, line := fn.FileLine(pc)
	e := Error{
		errorType: t,
		msg:       fmt.Sprintf(format, attrs...),

		fnName: fn.Name(),
		file:   file,
		line:   line,
	}
	return e
	// return errors.Wrap(e, wrap)
}

func nyi(what string, implFor interface{}) error {
	pc, _, _, _ := runtime.Caller(1)
	fn := runtime.FuncForPC(pc)
	file, line := fn.FileLine(pc)
	e := Error{
		errorType: NotYetImplemented,
		msg:       fmt.Sprintf("%s not yet implemented for %v of %T", what, implFor, implFor),

		fnName: fn.Name(),
		file:   file,
		line:   line,
	}
	return e
}

func nondiffErr(op Op) error {
	pc, _, _, _ := runtime.Caller(1)
	fn := runtime.FuncForPC(pc)
	file, line := fn.FileLine(pc)
	e := Error{
		errorType: SymbDiffError,
		msg:       fmt.Sprintf("%s not differentiable", op),

		fnName: fn.Name(),
		file:   file,
		line:   line,
	}
	return e
}
