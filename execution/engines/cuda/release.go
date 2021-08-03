// +build !debug

package cuda

func logf(format string, args ...interface{}) {}

func getfuncname(a interface{}) string {return "NO FUNC NAME IN RELEASE BUILD"}
