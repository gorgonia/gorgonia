// +build debug

package cuda

import (
	"runtime"
	"reflect"
)

func logf(format string, args ...interface{}) {}

func getfuncname(a interface{}) string {
	if a == nil {
		return "nil"
	}
	return runtime.FuncForPC(reflect.ValueOf(a).Pointer()).Name()
}
