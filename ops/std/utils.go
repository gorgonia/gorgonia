package stdops

import (
	"reflect"
	"runtime"
)

// funcName returns the name of the function.
func funcName(f interface{}) string {
	val := reflect.ValueOf(f)
	if val.Kind() != reflect.Func {
		return "Not a function"
	}

	//TODO(go1.18) - pc := uintptr(val.UnsafePointer())
	pc := val.Pointer()
	return runtime.FuncForPC(pc).Name()
}
