package errors

import (
	"fmt"
	"runtime"

	"github.com/pkg/errors"
)

const (
	CloneFail = "Failed to clone Value"
	OpDoFail  = "Failed to carry op.Do"

	TypeMismatch  = "Type Mismatch: a %T and b %T"
	ShapeMismatch = "Shape Mismatch. Expected %v. Got %v"
	ArrayMismatch = "Cannot reuse %v. Length of array: %d. Expected length of at least %d."
	DtypeError    = "Dtype Error. Expected %v. Got %v"

	SymbolicOpFail = "Failed to perform %v symbolically"

	noopMsg = "NoOp"

	// NYI errors
	prmsg        = "Please make a pull request at github.com/gorgonia/gorgoniai if you wish to contribute a solution"
	nyiFail      = "%q not yet implemented. "
	nyiTypeFail  = "%q not yet implemented for interactions with %T. "
	nyiTypeFail2 = "%q (%v) not yet implemented for interactions with %T. "
	nyiFailN     = "%q not yet implemented. %v. "

	FailedFuncOpt = "Unable to handle FuncOpts for %s"
	EngineSupport = "Engine %T does not implement %T, which is needed for %s"
)

type NoOp struct{}

func (err NoOp) Error() string { return noopMsg }
func (err NoOp) NoOp()         {}

// NYI is a convenience function that decorates a NYI error message with additional information.
func NYI(args ...interface{}) error {
	msg := nyiFail
	var fnName string = "UNKNOWN FUNCTION"
	pc, _, _, ok := runtime.Caller(1)
	if ok {
		fnName = runtime.FuncForPC(pc).Name()
	}

	switch len(args) {
	case 0:
		// no args, so only caller name.
		msg = nyiFail
	case 1:
		// 1 arg, it must be a "type"
		msg = nyiTypeFail
	case 2:
		// 2 args. it's a description, followed by a type
		msg = nyiTypeFail2
	default:
		msg = nyiFailN
	}

	// prepend fnName
	args = append(args, fnName)
	copy(args[1:], args[0:])
	args[0] = fnName
	err := fmt.Sprintf(msg, args...) + prmsg
	return errors.New(err)
}
