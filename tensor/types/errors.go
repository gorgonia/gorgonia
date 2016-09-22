package types

import (
	"fmt"
	"runtime"
)

type NoOpError interface {
	NoOp() bool
}

type noopError struct{}

func (e noopError) NoOp() bool    { return true }
func (e noopError) Error() string { return "NoOp" }

//go:generate stringer -type=errorType
type errorType byte

const (
	EmptyTensor errorType = iota // this is equivalent to a null pointer exception
	IndexError
	ShapeMismatch
	DtypeMismatch
	DimensionMismatch
	SizeMismatch
	AxisError
	InfNaNError
	IOError

	InvalidCmpOp
	OpError

	NotYetImplemented
)

const emptyTMsg = "Tensor is uninitialized (no shape, no data)"

type Error struct {
	errorType
	msg string

	// this section is added so it's easier to debug code
	fnName string
	line   int
	file   string
}

func (e Error) Error() string {
	return fmt.Sprintf("%v: %v. Happened at: %v:%v", e.errorType, e.msg, e.file, e.line)
}

func NewError(et errorType, format string, attrs ...interface{}) error {
	pc, _, _, _ := runtime.Caller(1)
	fn := runtime.FuncForPC(pc)
	file, line := fn.FileLine(pc)

	return Error{
		errorType: et,
		msg:       fmt.Sprintf(format, attrs...),
		fnName:    fn.Name(),
		file:      file,
		line:      line,
	}

}

func EmptyTensorError() error {
	pc, _, _, _ := runtime.Caller(1)
	fn := runtime.FuncForPC(pc)
	file, line := fn.FileLine(pc)

	return Error{errorType: EmptyTensor, msg: emptyTMsg, fnName: fn.Name(), file: file, line: line}
}

func DtypeMismatchErr(a, b Dtype) error {
	pc, _, _, _ := runtime.Caller(1)
	fn := runtime.FuncForPC(pc)
	file, line := fn.FileLine(pc)
	return Error{
		errorType: DtypeMismatch,
		msg:       fmt.Sprintf("Dtypes mismatched between %v and %v", a, b),
		fnName:    fn.Name(),
		file:      file,
		line:      line,
	}
}

func DimMismatchErr(expected, got int) error {
	pc, _, _, _ := runtime.Caller(1)
	fn := runtime.FuncForPC(pc)
	file, line := fn.FileLine(pc)
	return Error{
		errorType: DimensionMismatch,
		msg:       fmt.Sprintf("Dimension mismatch. Expected %d, got %d", expected, got),
		fnName:    fn.Name(),
		file:      file,
		line:      line,
	}
}

func AxisErr(format string, axes ...interface{}) error {
	pc, _, _, _ := runtime.Caller(1)
	fn := runtime.FuncForPC(pc)
	file, line := fn.FileLine(pc)
	return Error{
		errorType: AxisError,
		msg:       fmt.Sprintf(format, axes...),
		fnName:    fn.Name(),
		file:      file,
		line:      line,
	}
}

func IndexErr(axis, want, has int) error {
	pc, _, _, _ := runtime.Caller(1)
	fn := runtime.FuncForPC(pc)
	file, line := fn.FileLine(pc)
	return Error{
		errorType: IndexError,
		msg:       fmt.Sprintf("Index %d is out of bounds for axis %d which has size %d", want, axis, has),
		fnName:    fn.Name(),
		file:      file,
		line:      line,
	}
}
