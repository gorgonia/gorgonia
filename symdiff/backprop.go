package symdiff

import (
	"sort"

	"github.com/pkg/errors"
	"github.com/xtgo/set"
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
	return uniq(retVal)
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
	return uniq(retVal), nil
}

// Backporopagate computes the symbolic differentiation of the outputs with regards to the inputs.
//
// The algorithm is as follows:
// 	1. Filter out unreachable nodes.
// 	2. Forwards analysis, where a list of nodes affecting the output is added to `affectsOutput`
// 	3. Backwards analysis, where a list of nodes affected by differentiating the output are added to `affectedByOutput`
//	4. If there is a difference in both sets, then it's an error
// 	5. Walk the graph from output towards input. On visit of each node, perform symbolic differentiation.
func Backporopagate(g *exprgraph.Graph, outputs, gradOutputs, wrt []*exprgraph.Node) (deriv map[exprgraph.NodeID]exprgraph.NodeID, derivOf map[exprgraph.NodeID]exprgraph.NodeIDs, err error) {
	// input check
	if len(outputs) != len(gradOutputs) {
		// error
		return nil, nil, errors.Errorf("Expected `outputs` and `gradOutputs` to have the same length. `outputs` has %d items. `gradOutputs` has %d items`", len(outputs), len(gradOutputs))
	}

	var sorted []*exprgraph.Node
	if sorted, err = exprgraph.Sort(g); err != nil {
		return nil, nil, errors.Wrap(err, "Failed to sort graph in Backpropagate()")
	}

	/* Checks */
	/*
	   1. A forwards and backwards analysis is made
	   2. A set difference between the `wrt` and `affectsOutput` is computed. There should be NO difference.
	   3. A set difference between the `outputs` and `affectedByOutput` is computed. There should be NO difference.
	*/

	var affectsOutput, affectedByOutput exprgraph.NodeIDs
	affectsOutput = forwardDiffAnalysis(g, outputs, sorted)
	if affectedByOutput, err = backwardDiffAnalysis(g, wrt, sorted); err != nil {
		return nil, nil, errors.Wrap(err, "Failed backwards differentiation analysis")
	}

	wrtSet := exprgraph.NodeIDsFromNodes(wrt)
	wrtSet = uniq(wrtSet)
	badWRTs := diff(wrtSet, affectsOutput)
	if len(badWRTs) > 0 {
		// error
	}

	outSet := exprgraph.NodeIDsFromNodes(outputs)
	outSet = uniq(outSet)
	badOuts := diff(outSet, affectedByOutput)
	if len(badOuts) > 0 {
		// error
	}

	/* Do the symbolic differentiation here now that everything has been checked */

	// nodeGrads is a map of a node to a list of its gradient terms
	// These gradient terms will be summed up when we visit the node
	// while iterating thru the nodes in reverse topological order.
	nodeGrads := make(map[exprgraph.NodeID]exprgraph.NodeIDs)
	for i, n := range outputs {
		nodeGrads[n.NodeID()] = exprgraph.NodeIDs{gradOutputs[i].NodeID()}
	}
	deriv = make(map[exprgraph.NodeID]exprgraph.NodeID)
	derivOf = make(map[exprgraph.NodeID]exprgraph.NodeIDs)

	// actives are nodes that are differentially influenced by the inputs
	// and also differentiably influence the outputs.
	// THese are the nodes wehre we need to call the pullback function to backpropagate the derivatives
	actives := inter(affectsOutput, affectedByOutput)

	for _, n := range sorted {
		if in(actives, n.NodeID()) {
			// skip, because it's already in the list of actives.
			continue
		}
		if d, ok := deriv[n.NodeID()]; ok {
			// skip, because it was previously differentiated
			nodeGrads[n.NodeID()] = append(nodeGrads[n.NodeID()], d)
			continue
		}

		// check if there are any grads coming into this node
		grads := nodeGrads[n.NodeID()]
		switch len(grads) {
		case 0:
			err := Error{
				single:    n.NodeID(),
				nodeGrads: nodeGrads,
				err:       errors.New("No gradients found for node"),
			}
			return nil, nil, err
		case 1:
			// TODO

		default:
			// once we've reached a node, we've already backpropagated from its dependents, so we sum up the gradients
			// TODO
		}

	}

	panic("NYI")
}

// uniq computes the unique node IDs in a set. This operation is an inplace operation. `a` will get clobbered.
func uniq(a exprgraph.NodeIDs) exprgraph.NodeIDs {
	sort.Sort(a)
	n := set.Uniq(a)
	return a[:n]
}

// diff performs the diff between sets `a - b`. It performs it inplace, so `a` will get clobbered.
func diff(a, b exprgraph.NodeIDs) exprgraph.NodeIDs {
	piv := len(a)
	a = append(a, b...)
	sz := set.Diff(a, piv)
	return a[:sz]
}

// inter performs a set intersection between two sets `a ∩ b`. This operation is an inplace operation. `a` will be clobbered.
func inter(a, b exprgraph.NodeIDs) exprgraph.NodeIDs {
	piv := len(a)
	a = append(a, b...)
	n := set.Inter(a, piv)
	return a[:n]
}

// in checks if the wanted node is in the set.
func in(a exprgraph.NodeIDs, want exprgraph.NodeID) bool {
	for _, v := range a {
		if v == want {
			return true
		}
	}
	return false
}
