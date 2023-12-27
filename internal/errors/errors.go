package errors

import (
	"github.com/pkg/errors"
	"runtime"
)

var (
	Errorf = errors.Errorf
	Wrap   = errors.Wrap
	Wrapf  = errors.Wrapf
	New    = errors.New
)

// ThisFn returns the name of the function
func ThisFn(skips ...uint) string {
	c := 1
	if len(skips) > 0 {
		c += int(skips[0])
	}

	pc, _, _, ok := runtime.Caller(c)
	if !ok {
		return "UNKNOWNFUNC"
	}
	return runtime.FuncForPC(pc).Name()
}
