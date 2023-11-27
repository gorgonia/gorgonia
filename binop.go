package gorgonia

import (
	_ "unsafe"
)

// this file provides utility functions for the binary operation APIs

//--go:linkname binopSymbolic gapi.binopSymbolic
//func binopSymbolic(op ops.Op, g *exprgraph.Graph, a, b Tensor) (retVal Tensor, err error)
