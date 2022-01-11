package symdiff

import (
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia/exprgraph"
)

// forwardDiffAnalysis returns the nodes that affect the outputs.
func forwardDiffAnalysis(g *exprgraph.Graph, outputs, sorted []*exprgraph.Node) (retVal exprgraph.NodeIDs) {
	retVal = exprgraph.NodeIDsFromNodes(outputs)

	for _, n := range sorted {
		op, ok := n.Op.(Op)
		if retVal.Contains(n.NodeID()) && ok { // op == nil ⇒ n is input
			children := g.ChildrenOf(n)
			diffs := op.DiffWRT(len(children))
			for i, d := range diffs {
				if d {
					retVal = append(retVal, children[i])
				}
			}
		}
	}
	return
}

// backwardDiffAnalysis returns a list of Nodes that are affected by differentiating output.
// Given a list of WRTs, we want to find a list of nodes that will be affected when backpropagating.
func backwardDiffAnalysis(g *exprgraph.Graph, wrt, sorted []*exprgraph.Node) (retVal exprgraph.NodeIDs, err error) {
	retVal = exprgraph.NodeIDsFromNodes(sorted)

	var (
		op    Op
		ok    bool
		diffs []bool
	)

	// traverse the sorted nodes in reverse order
	for i := len(sorted) - 1; i >= 0; i-- {
		n := sorted[i]
		children := g.ChildrenOf(n)
		if op, ok = n.Op.(Op); ok {
			diffs = op.DiffWRT(len(children))
		}

		// check for non differentiable things
		if len(diffs) == 0 {
			// three cases in which we should continue on to the next node in the sorted list:
			// 	- `op` is a statement.
			//	- `n` is an input (usually correlated with being a leaf).
			// 	- `n` is a leaf node.
			switch {
			case isStmt(op):
				// op is statement
				continue
			case isInput(op):
				// op == nil ⇒ n is input
				continue
			case len(children) == 0:
				// leaf
				continue
			}

			for _, child := range children {
				c := g.Node(int64(child)).(*exprgraph.Node)
				parents := g.ParentsOf(c)
				grandKids := g.ChildrenOf(c)
				if len(parents) == 1 && len(grandKids) > 0 {
					return nil, errors.Errorf("%v is undifferentiable. This makes a portion of the graph unreachable")
				}
			}

		}

	inner:
		for j, child := range children {
			d := diffs[j]
			if retVal.Contains(child) && d {
				retVal = append(retVal, n.NodeID())
				break inner
			}
		}
	}
	return

}

// Backporopagate computes the symbolic differentiation of the outputs with regards to the inputs.
//
// The algorithm is as follows:
// 	1. Filter out unreachable nodes.
// 	2. Forwards analysis, where a list of nodes affecting the output is added to `affectsOutput`
// 	3. Backwards analysis, where a list of nodes affected by differentiating the output are added to `affectedByOutput`
//	4. If there is a difference in both sets, then it's an error
// 	5. Walk the graph from output towards input. On visit of each node, perform symbolic differentiation.
func Backporopagate(g *exprgraph.Graph, outputs, gradOutputs, wrt []*exprgraph.Node) (retVal map[*exprgraph.Node]*exprgraph.Node, err error) {
	panic("NYI")
}
