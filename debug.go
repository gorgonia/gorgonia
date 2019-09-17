// +build debug

package gorgonia

import (
	"bytes"
	"fmt"
	"log"
	"os"
	"strings"
	"sync/atomic"
)

// DEBUG is a global flag that activates various debugging functions
const DEBUG = true

func init() {
	log.Printf("DEBUG")
}

// these constants are used during development time - mainly on tracing statements to see the values of certain things.
// I use these instead of say, Delve because most of the time, the larger picture has to be known. Delve tends to give small picture views
var (
	compileDev        = false
	shapeInferenceDev = false
	typeSystemDev     = false
	symdiffDev        = false
	autodiffDev       = false
	machineDev        = false
	stabilizationDev  = false
	solverDev         = false
	cudaDev           = false
	allocatorDev      = false
)

// TABCOUNT is a global flag used when debugging
var TABCOUNT uint32

var logger = log.New(os.Stderr, "", 0)
var replacement = "\n"

func tabcount() int {
	return int(atomic.LoadUint32(&TABCOUNT))
}

func enterLogScope() {
	atomic.AddUint32(&TABCOUNT, 1)
	tabcount := tabcount()
	logger.SetPrefix(strings.Repeat("\t", tabcount))
	replacement = "\n" + strings.Repeat("\t", tabcount)
}

func leaveLogScope() {
	tabcount := tabcount()
	tabcount--

	if tabcount < 0 {
		atomic.StoreUint32(&TABCOUNT, 0)
		tabcount = 0
	} else {
		atomic.StoreUint32(&TABCOUNT, uint32(tabcount))
	}
	logger.SetPrefix(strings.Repeat("\t", tabcount))
	replacement = "\n" + strings.Repeat("\t", tabcount)
}

func logf(format string, others ...interface{}) {
	if DEBUG {
		// format = strings.Replace(format, "\n", replacement, -1)
		s := fmt.Sprintf(format, others...)
		s = strings.Replace(s, "\n", replacement, -1)
		logger.Println(s)
		// logger.Printf(format, others...)
	}
}

func compileLogf(format string, attrs ...interface{}) {
	if compileDev {
		logf(format, attrs...)
	}
}

func shapeLogf(format string, attrs ...interface{}) {
	if shapeInferenceDev {
		logf(format, attrs...)
	}
}

func typeSysLogf(format string, attrs ...interface{}) {
	if typeSystemDev {
		logf(format, attrs...)
	}
}

func symdiffLogf(format string, attrs ...interface{}) {
	if symdiffDev {
		logf(format, attrs...)
	}
}

func autodiffLogf(format string, attrs ...interface{}) {
	if autodiffDev {
		logf(format, attrs...)
	}
}

func machineLogf(format string, attrs ...interface{}) {
	if machineDev {
		logf(format, attrs...)
	}
}

func stabLogf(format string, attrs ...interface{}) {
	if stabilizationDev {
		logf(format, attrs...)
	}
}

func solverLogf(format string, attrs ...interface{}) {
	if solverDev {
		logf(format, attrs...)
	}
}

func cudaLogf(format string, attrs ...interface{}) {
	if cudaDev {
		logf(format, attrs...)
	}
}

func allocatorLogf(format string, attrs ...interface{}) {
	if allocatorDev {
		logf(format, attrs...)
	}
}

func recoverFrom(format string, attrs ...interface{}) {
	if r := recover(); r != nil {
		logger.Printf(format, attrs...)
		panic(r)
	}
}

/* Graph Collision related debugging code */
var nnc, cc, ec int64

func incrNN() {
	atomic.AddInt64(&nnc, 1)
}

func incrCC() {
	atomic.AddInt64(&cc, 1)
}

func incrEC() {
	atomic.AddInt64(&ec, 1)
}

// GraphCollisionStats ...
func GraphCollisionStats() (int, int, int) {
	return int(atomic.LoadInt64(&nnc)), int(atomic.LoadInt64(&cc)), int(atomic.LoadInt64(&ec))
}

/* Compilation related debug utility functions/methods*/
func logCompileState(name string, g *ExprGraph, df *dataflow) {
	var fname string
	if name == "" {
		fname = "TotallyRandomName.csv"
	} else {
		fname = name + ".csv"
	}
	f, err := os.OpenFile(fname, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	compileState(f, g, df)
	compileLogf("Written Compile State to %v", fname)
}

/* Analysis Debug Utility Functions/Methods */
func (df *dataflow) debugIntervals(sorted Nodes) {
	var buf bytes.Buffer
	buf.Write([]byte("Intervals:\n"))
	for _, n := range sorted {
		fmt.Fprintf(&buf, "\t%v:\t%v\n", n, df.intervals[n])
	}
	compileLogf(buf.String())
}
