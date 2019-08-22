// +build cuda

package gorgonia

import "runtime"

// InitCuda runs the main gorgonia service loop.
// The binary's main.main must call gorgonia.InitCuda() to run this loop.
// InitCuda does not return. If the binary needs to do other work, it
// must do it in separate goroutines.
func initCuda(done <-chan bool) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	for {
		select {
		case f := <-mainfunc:
			f()
		case <-done:
			return
		}
	}
}

// queue of work to run in main thread.
var mainfunc = make(chan func())

// do runs f on the main thread.
func mainDo(f func()) {
	done := make(chan bool, 1)
	mainfunc <- func() {
		f()
		done <- true
	}
	<-done
}
