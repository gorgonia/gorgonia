// +build !debug

package gorgonia

// DEBUG indicates if this build is in debug mode. It is not.
const DEBUG = false

const (
	compileDev        = false
	shapeInferenceDev = false
	typeSystemDev     = false
	symdiffDev        = false
	autodiffDev       = false
	machineDev        = false
	stabilizationDev  = false
	cudaDev           = false
	allocatorDev      = false
)

func tabcount() int { return 0 }

func enterLogScope()                                   {}
func leaveLogScope()                                   {}
func logf(format string, others ...interface{})        {}
func compileLogf(format string, attrs ...interface{})  {}
func shapeLogf(format string, attrs ...interface{})    {}
func typeSysLogf(format string, attrs ...interface{})  {}
func symdiffLogf(format string, attrs ...interface{})  {}
func autodiffLogf(format string, attrs ...interface{}) {}
func machineLogf(format string, attrs ...interface{})  {}
func stabLogf(format string, attrs ...interface{})     {}
func solverLogf(format string, attrs ...interface{})   {}
func cudaLogf(format string, attrs ...interface{})     {}
func allocatorLogf(format string, attr ...interface{}) {}
func recoverFrom(format string, attrs ...interface{})  {}

// GraphCollisionStats returns the collisions in the graph only when built with the debug tag, otherwise it's a noop that returns 0
func GraphCollisionStats() (int, int, int) { return 0, 0, 0 }

func incrCC() {}
func incrEC() {}
func incrNN() {}

/* Compilation related debug utility functions/methods*/
func logCompileState(name string, g *ExprGraph, df *dataflow) {}

/* Analysis Debug Utility Functions/Methods */
func (df *dataflow) debugIntervals(sorted Nodes) {}
