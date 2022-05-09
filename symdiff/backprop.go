package symdiff

import (
	"sort"

	"github.com/pkg/errors"
	"github.com/xtgo/set"
	"gorgonia.org/gorgonia/exprgraph"
	gapi "gorgonia.org/gorgonia/internal/api"
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
	retVal = exprgraph.NodeIDsFromNodes(wrt)

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
			case isInput(n.Op):
				// op == nil ⇒ n is input
				continue
			case len(children) == 0:
				// leaf
				continue
			}

			// non differentiable op
			for j, child := range children {
				c := g.Node(int64(child)).(*exprgraph.Node)
				parents := g.ParentsOf(c)
				grandKids := g.ChildrenOf(c)
				if len(parents) == 1 && len(grandKids) > 0 {
					return nil, errors.Errorf("%v is  the %dth child of %v. It is undifferentiable. This makes a portion of the graph unreachable", c, j, n)
				}
			}
			continue
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
//
// Here's a visual example of what is going on. Given an expression like:
//	(x × w) + b
// The graph looks like this:
// 	        x×w+b
// 	           │
// 	      ┌────┴────┐
// 	      │         │
// 	      │         │
// 	      ▼         ▼
// 	     x×w        b
// 	      │
// 	┌─────┴────┐
// 	│          │
// 	▼          ▼
// 	x          w
// The sorted nodes will be as follows:
// 	[x×w+b x×w b w x]
// Here `x×w+b` is an output. So there will be an associated `gradOutput`. The `deriv` map will
// look like the following at the start:
// 	{x×w+b: grad_x×w+b}
// Now we traverse the graph using the topologically sorted slice, which yields the following:
// 	| Step | Visited |                   Action                    |                  Result                   |                            `deriv`                             |
// 	|------|---------|---------------------------------------------|-------------------------------------------|----------------------------------------------------------------|
// 	|    0 | x×w+b   | Call +.SymDiff({x×w, b}, x×w+b, grad_x×w+b) | Get {grad_x×w, grad_b} then map it.       | {x×w+b:grad_x×w+b, x×w:grad_x×w, b:grad_b}                     |
// 	|    1 | x×w     | Call .SymDiff({x, w}, x×w, grad_x×w)       | Get {grad_x, grad_w} then map it.         | {x×w+b:grad_x×w+b, x×w:grad_x×w, b:grad_b, w:grad_w, x:grad_x} |
// 	|    2 | b       | No Action.                                  | The gradient is already mapped in Step 0. |                                                                |
// 	|    3 | w       | No Action.                                  | The gradient is already mapped in Step 1. |                                                                |
// 	|    4 | x       | No Action.                                  | The gradient is already mapped in Step 1. |                                                                |
//
// This is the simple case. For more complicated graphs where multiple gradients need to accumulate (e.g `w` in the following expression)
// 	x×w²
// The multiple gradients of `w` would need to be summed up beforehand. This is done through mapping in `nodeGrads`, which tracks the
// input gradients
func Backporopagate(g *exprgraph.Graph, outputs, gradOutputs, wrt []*exprgraph.Node) (bpgraph *Graph, err error) {
	// input check
	if len(outputs) != len(gradOutputs) {
		// error
		return nil, errors.Errorf("Expected `outputs` and `gradOutputs` to have the same length. `outputs` has %d items. `gradOutputs` has %d items`", len(outputs), len(gradOutputs))
	}

	var sorted []*exprgraph.Node
	if sorted, err = exprgraph.Sort(g); err != nil {
		return nil, errors.Wrap(err, "Failed to sort graph in Backpropagate()")
	}

	// TODO: filter out unreachable nodes (out of consideration anyway)

	/* Checks */
	/*
	   1. A forwards and backwards analysis is made
	   2. A set difference between the `wrt` and `affectsOutput` is computed. There should be NO difference.
	   3. A set difference between the `outputs` and `affectedByOutput` is computed. There should be NO difference.
	*/

	var affectsOutput, affectedByOutput exprgraph.NodeIDs
	affectsOutput = forwardDiffAnalysis(g, outputs, sorted)
	if affectedByOutput, err = backwardDiffAnalysis(g, wrt, sorted); err != nil {
		return nil, errors.Wrap(err, "Failed backwards differentiation analysis")
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
	deriv := make(map[exprgraph.NodeID]exprgraph.NodeID)
	derivOf := make(map[exprgraph.NodeID]exprgraph.NodeIDs)

	// actives are nodes that are differentially influenced by the inputs
	// and also differentiably influence the outputs.
	// THese are the nodes wehre we need to call the pullback function to backpropagate the derivatives
	actives := inter(affectsOutput, affectedByOutput)

	for _, n := range sorted {
		op, isSDOp := n.Op.(Op)
		nid := n.NodeID()

		if in(actives, nid) {
			// skip, because it's already in the list of actives.
			continue
		}
		if d, ok := deriv[nid]; ok {
			// skip, because it was previously differentiated
			nodeGrads[nid] = append(nodeGrads[nid], d)
			continue
		}

		// check if there are any grads coming into this node
		grads := nodeGrads[nid]
		switch len(grads) {
		case 0:
			return nil, Error{
				g:         g,
				single:    nid,
				nodeGrads: nodeGrads,
				err:       errors.New("No gradients found for node"),
			}
		case 1:
			d := nodeGrads[nid][0]
			deriv[nid] = d
			derivOf[d] = append(derivOf[d], nid)

		default:
			// once we've reached a node, we've already backpropagated from its dependents, so we sum up the gradients
			summed, err := gapi.ReduceAdd(exprgraph.TensorsFromNodeIDs(g, nodeGrads[nid]))
			if err != nil {
				return nil, Error{
					g:         g,
					single:    nid,
					nodeGrads: nodeGrads,
					err:       errors.Wrap(err, "Failed to sum the gradients for the node"),
				}
			}
			d := summed.(*exprgraph.Node).NodeID()
			deriv[nid] = d
			derivOf[d] = append(derivOf[d], nid)
			nodeGrads[nid] = exprgraph.NodeIDs{d}
		}

		gradNode := g.Get(nodeGrads[nid][0])

		// do the symbolic differentiation
		if isInput(n.Op) {
			continue
		}

		if !isSDOp {
			err = Error{
				g:      g,
				single: nid,
				err:    errors.Errorf("%v Not a symdiff.Op", op),
			}
			return nil, err
		}

		children := exprgraph.NodesFromNodeIDs(g, g.ChildrenOf(n))
		childrenGrads, err := op.SymDiff(g, children, n, gradNode)
		if err != nil {
			return nil, Error{
				single:    nid,
				grad:      gradNode.NodeID(),
				nodeGrads: nodeGrads,
				err:       errors.Wrap(err, ".SymDiff() failed"),
			}
		}

		diffs := op.DiffWRT(len(children))
		for i, child := range children {
			cid := child.NodeID()
			differentiable := diffs[i]
			childGrad := childrenGrads[i]
			if differentiable {
				nodeGrads[cid] = append(nodeGrads[cid], childGrad.NodeID())
			}
		}

	}
	return &Graph{
		Graph:   g,
		deriv:   deriv,
		derivOf: derivOf,
	}, nil
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
