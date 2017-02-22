package gorgonia

// global options

var debugDerives = true
var stabilization = true
var optimizationLevel = 0

// UseStabilization sets the global option to invoke stabilization functions when building the graph.
// Numerical stabilization is on by default
func UseStabilization() {
	stabilization = true
}

// UseNonStable turns off the stabilization functions when building graphs.
func UseNonStable() {
	stabilization = false
}

// DebugDerives turns on the derivation debug option when printing a graph
func DebugDerives() {
	debugDerives = true
}

// DontDebugDerives turns off derivation debug option when printing a graph.
// It is off by default
func DontDebugDerives() {
	debugDerives = false
}
