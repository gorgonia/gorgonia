// +build debug
// +build linux

package cuda

import (
	"runtime"
	"syscall"
)

func logtid(category string, logcaller int) {
	tid := syscall.Gettid()
	format := category + "- tid %v"
	if logcaller > 0 {
		pc, _, _, _ := runtime.Caller(logcaller + 1)
		format += ", called by %v"
		logf(format, tid, runtime.FuncForPC(pc).Name())
		return
	}
	logf(format, tid)
}
