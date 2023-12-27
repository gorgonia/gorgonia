package gerrors

import "runtime"

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
