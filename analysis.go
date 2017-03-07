package gorgonia

import (
	"bytes"
	"fmt"
)

// dataflow analysis

type dataflow struct {
	uniques map[uint32]*Node

	replacements map[*Node]*Node
	intervals    map[*Node]*interval
}

func newdataflow() *dataflow {
	df := new(dataflow)
	df.uniques = make(map[uint32]*Node)
	return df
}

// equivalent to the value numbering algorithm
// it returns true if it is unique
func (df *dataflow) vn(n *Node) (retVal *Node, unique bool) {
	compileLogf("Value numbering")
	enterLoggingContext()
	defer leaveLoggingContext()

	node, ok := df.uniques[n.Hashcode()]

	if ok {
		return node, false
	}

	compileLogf("adding a new unique")
	// otherwise, add it to uniques, and then return itself
	df.uniques[n.Hashcode()] = n

	return n, true
}

func analyze(g *ExprGraph, sorted Nodes) *dataflow {
	compileLogf("Performing dataflow analysis")
	enterLoggingContext()
	defer leaveLoggingContext()

	compileLogf("Finding unique leaves")
	df := newdataflow()
	for _, n := range g.leaves {
		df.uniques[n.Hashcode()] = n
	}

	compileLogf("Common subexpression elimination")

	// common subexpression elimination
	replacements := make(map[*Node]*Node)
	var buf bytes.Buffer
	for _, n := range sorted {
		// for i := len(sorted) - 1; i >= 0; i-- {
		// 	n := sorted[i]
		fmt.Fprintf(&buf, "%d, ", n.ID())
		r, _ := df.vn(n)
		replacements[n] = r
	}
	df.replacements = replacements

	compileLogf("replacements: %+p", FmtNodeMap(replacements))
	compileLogf("%v", buf.String())

	// TODO
	// constant propagation
	/*
		for _, node := range g.nodes {
			n := node.(*Node)
			if len(n.Children) > 0 {
				allConst := true
				for _, child := range n.Children {
					if _, ok := child.Op.(constant); !ok {
						allConst = false
						break
					}
				}
			}
		}
	*/
	return df
}

func analyzeMem(g *ExprGraph, sorted Nodes) {

	for _, node := range sorted {
		switch {
		case node.isArg():
		case node.op.OverwritesInput() >= 0:
		case node.op.ReturnsPtr():

		}
	}
}
