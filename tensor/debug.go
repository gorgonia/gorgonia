// +build debug

package tensor

import (
	"fmt"
	"log"
	"os"
	"reflect"
	"runtime/debug"
	"strings"
	"sync/atomic"
	"unsafe"
)

var TABCOUNT uint32

var TRACK = false

const DEBUG = true

var _logger_ = log.New(os.Stderr, "", 0)
var replacement = "\n"

func tabcount() int {
	return int(atomic.LoadUint32(&TABCOUNT))
}

func enterLoggingContext() {
	atomic.AddUint32(&TABCOUNT, 1)
	tabcount := tabcount()
	_logger_.SetPrefix(strings.Repeat("\t", tabcount))
	replacement = "\n" + strings.Repeat("\t", tabcount)
}

func leaveLoggingContext() {
	tabcount := tabcount()
	tabcount--

	if tabcount < 0 {
		atomic.StoreUint32(&TABCOUNT, 0)
		tabcount = 0
	} else {
		atomic.StoreUint32(&TABCOUNT, uint32(tabcount))
	}
	_logger_.SetPrefix(strings.Repeat("\t", tabcount))
	replacement = "\n" + strings.Repeat("\t", tabcount)
}

func logf(format string, others ...interface{}) {
	if DEBUG {
		// format = strings.Replace(format, "\n", replacement, -1)
		s := fmt.Sprintf(format, others...)
		s = strings.Replace(s, "\n", replacement, -1)
		_logger_.Println(s)
		// _logger_.Printf(format, others...)
	}
}

var stats = new(debug.GCStats)

func loggc() {
	debug.ReadGCStats(stats)
	log.Printf("NUMGC: %v", stats.NumGC)
}

func init() {
	debug.SetPanicOnFault(true)
	debug.SetTraceback("all")
}

type rtype struct {
	size       uintptr
	ptrdata    uintptr // number of bytes in the type that can contain pointers
	hash       uint32  // hash of type; avoids computation in hash tables
	tflag      uint8   // extra type information flags
	align      uint8   // alignment of variable with this type
	fieldAlign uint8   // alignment of struct field with this type
	kind       uint8   // enumeration for C
	alg        uintptr // algorithm table
	gcdata     uintptr // garbage collection data
	str        int32   // string form
	ptrToThis  int32   // type for pointer to this type, may be zero
}

func (t *rtype) Format(s fmt.State, c rune) {
	fmt.Fprintf(s, "size: %d pointers: %d, hash: 0x%x, flag: %d, align: %d, kind: %d", t.size, t.ptrdata, t.hash, t.tflag, t.align, t.kind)
}

func logRtype(t *reflect.Type) {
	iface := *(*[2]uintptr)(unsafe.Pointer(t))
	rt := (*rtype)(unsafe.Pointer(iface[1]))
	log.Printf("TYPE INFO: %v(%p) - %v", *t, t, rt)
}
