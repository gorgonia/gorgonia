package gorgonia

import (
	"github.com/pkg/errors"
	"gonum.org/v1/gonum/graph"
)

/*
This file holds code for symbolic differentiation.
The purpose of the symbolic differentiation is to analyze and prepare the nodes for automatic differentiation.

The main function that does all the magic is in Backpropagate().


see also: http://colah.github.io/posts/2015-08-Backprop/
*/

// forwardDiffAnalysis returns the nodes that affect outputs.
//
// Given a list of outputs, we want to know which nodes will affect the output
func forwardDiffAnalysis(outputs, sortedNodes Nodes) (retVal NodeSet, err error) {
	symdiffLogf("Forward analysis. Already sorted?")
	enterLogScope()
	defer leaveLogScope()

	sane := outputs.AllSameGraph()
	if !sane {
		return nil, errors.New("The supplied output Nodes are not the same graph")
	}

	// diffSet := outputs.Set()
	diffSet := outputs.mapSet()

	symdiffLogf("Diff Set: %d", diffSet)
	symdiffLogf("%d", sortedNodes)
	// for i := len(sortedNodes) - 1; i ⩾ 0; i-- {
	// 	n := sortedNodes[i]
	for _, n := range sortedNodes {
		if diffSet.Contains(n) && !n.isInput() {
			diffs := n.diffWRT()
			for j, child := range n.children {
				d := diffs[j]
				if d {
					symdiffLogf("Adding %x to  differentiable set", child.ID())
					// diffSet = append(diffSet, child)
					diffSet.Add(child)
				}
			}
		}
	}
	return diffSet, nil
}

// backwardDiffAnalysis returns a list of Nodes that are affected by differentiating output.
// Given a list of WRTs, we want to find a list of nodes that will be affected when backpropagating.
func backwardDiffAnalysis(wrt, sortedNodes Nodes) (retVal NodeSet, err error) {
	symdiffLogf("Backwards analysis")
	enterLogScope()
	defer leaveLogScope()

	sane := wrt.AllSameGraph()
	if !sane {
		return nil, errors.New("The supplied output Nodes are not the same graph")
	}

	// diffSet := wrt.Set()
	diffSet := wrt.mapSet()
	// autodiffLogf("Diff Set: %d", diffSet)
	symdiffLogf("wrt:%d diffset: %d", len(wrt), len(diffSet))
	symdiffLogf("%v", diffSet)
	symdiffLogf("sorted: %d", sortedNodes)

	enterLogScope()
	// for _, n := range sortedNodes {
	for i := len(sortedNodes) - 1; i >= 0; i-- {
		n := sortedNodes[i]
		symdiffLogf("working on %v. Has %d children", n, len(n.children))

		var op SDOp
		var ok bool
		var diffs []bool
		if op, ok = n.op.(SDOp); ok {
			diffs = op.DiffWRT(len(n.children))
		}

		symdiffLogf("differentiable WRT: %v", diffs)
		enterLogScope()
		symdiffLogf("Children: %v", n.children)
		if len(diffs) == 0 {
			// check if this makes nodes unreachable. If it does, then error out
			if n.isStmt {
				symdiffLogf("Statement nodes are Non differentiable!")
				leaveLogScope()
				continue
			} else if n.isInput() {
				symdiffLogf("Input nodes are Non differentiable")
				leaveLogScope()
				continue
			} else if len(n.children) == 0 {
				symdiffLogf("Leaf nodes have no children")
				leaveLogScope()
				continue
			}
			g := n.g
			for _, child := range n.children {
				parents := graph.NodesOf(g.To(child.ID()))

				// symdiffLogf("parents of %v: %v", child, graphNodeToNode(parents))
				if len(parents) == 1 && len(child.children) > 0 {
					leaveLogScope()
					return nil, errors.Errorf("Being unable to differentiate %v would leave a portion of the graph unreachable. Unable to continue", n)
				}
			}
			symdiffLogf("SKIPPING... Non differentiable!")
			leaveLogScope()
			continue
		}

	inner:
		for j, child := range n.children {
			d := diffs[j]
			if diffSet.Contains(child) && d {
				symdiffLogf("Adding %x to differentiable set", child.ID())
				// diffSet = append(diffSet, n)
				diffSet.Add(n)
				break inner
			}
		}
		leaveLogScope()
	}
	leaveLogScope()
	return diffSet, nil
}

// Backpropagate backpropagates errors by performing revers-emode symbolic differentiation, starting from the outputs, and working its way towads the inputs.
//
// This is the rough algorithm:
//		1. Filter out nodes that are unreachable
//		2. Forwards analysis, where a list of nodes affecting the output is added to consideration
//		3. Backwards analysis, where a list of nodes affected by differentiating the output are added to the consideration
//		4. If there is a difference in both sets, it will cause an error (both sets should be the same)
//		5. Traverse the graph from output towards input. On each visit, perform the symbolic differentiation
//
// For most cases, Grad() should be used instead of Backpropagate(), as Grad() performs several checks which would be the general use case, before calling Backpropagate()
func Backpropagate(outputs, gradOutputs, wrt Nodes) (retVal Nodes, err error) {
	symdiffLogf("BACKPROP START")
	symdiffLogf("Outputs: %d", outputs)
	symdiffLogf("gradOutputs: %d", gradOutputs)
	symdiffLogf("WRT: %d", wrt)

	enterLogScope()
	defer leaveLogScope()

	g := outputs[0].g

	// this entire section about removing foreveralone nodes need a rethink
	symdiffLogf("removing foreveralone nodes")
	enterLogScope()
	for i := 0; i < len(g.AllNodes()); i++ {
		n := g.AllNodes()[i]

		fr := g.From(n.ID()).Len()
		to := g.To(n.ID()).Len()

		if fr == 0 && to == 0 && !n.isConstant() && !n.isInput() {
			g.RemoveNode(n)
			symdiffLogf("removed %v(%p); %x; %s", n, n, n.ID(), n.Name())
		}
	}
	leaveLogScope()

	var sortedNodes Nodes
	if sortedNodes, err = Sort(g); err != nil {
		return nil, errors.Wrap(err, sortFail)
	}
	symdiffLogf("sorted nodes: %v", sortedNodes)
	symdiffLogf("sorted nodes: %d", sortedNodes)

	var affectsOutput NodeSet
	var affectedByOutput NodeSet
	if affectsOutput, err = forwardDiffAnalysis(outputs, sortedNodes); err != nil {
		return nil, errors.Wrap(err, "Failed during forward differentiation analysis")
	}

	if affectedByOutput, err = backwardDiffAnalysis(wrt, sortedNodes); err != nil {
		return nil, errors.Wrap(err, "Failed during forward differentiation analysis")
	}

	symdiffLogf("affects output: %v", affectsOutput)
	symdiffLogf("affected by output : %v", affectedByOutput)

	wrtSet := wrt.mapSet()
	badWRTs := wrtSet.Difference(affectsOutput)
	if len(badWRTs) > 0 {
		return nil, SymDiffError{nodes: badWRTs.ToSlice(), err: errors.New("Non Differentiable WRTs")}
	}

	outputSet := outputs.mapSet()
	badOutputs := outputSet.Difference(affectedByOutput)
	if len(badOutputs) > 0 {
		symdiffLogf("badOutputs: %#v", badOutputs)
		return nil, SymDiffError{nodes: badOutputs.ToSlice(), err: errors.New("Non-Differentable Outputs")}
	}

	// map a node to a list of gradient terms
	// these  gradient terms will be summed up when we visit the node
	// when iterating through the nondes in reverse topological order
	nodeGradMap := make(map[*Node]Nodes)
	for i, n := range outputs {
		symdiffLogf("Adding outputs for %x", n.ID())
		nodeGradMap[n] = Nodes{gradOutputs[i]}
	}

	// "active" nodes are the ones that are differentially influenced by the inputs
	// and also differentiably influence the outputs. These are the nodes where we need to call the
	// "pullback" function to backpropagate derivatives
	activeNodes := affectsOutput.Intersect(affectedByOutput)

	symdiffLogf("Active: %d", activeNodes)

	symdiffLogf("Sorted: %d", sortedNodes)
	symdiffLogf("nodeGradMap: %+#d", FmtNodeMap(nodeGradMap))
	enterLogScope()

	for _, node := range sortedNodes {
		if _, ok := activeNodes[node]; !ok {
			symdiffLogf("skipping %x", node.ID())
			continue
		}

		if node.deriv != nil {
			symdiffLogf("skipping %x - previously differentiated", node.ID())
			nodeGradMap[node] = append(nodeGradMap[node], node.deriv)
			continue
		}

		symdiffLogf("Working on %x %v", node.ID(), node)
		enterLogScope()

		// Check if there is any grads coming into this node
		if len(nodeGradMap[node]) < 1 {
			leaveLogScope()
			return nil, SymDiffError{
				single:  node,
				gradMap: nodeGradMap,
				err:     errors.New("No gradients found for node"),
			}
		}

		// once we've reached a node, we already backpropagated from its dependents
		// so we sum up the gradients
		symdiffLogf("nodeGradMap[%x]: %d", node.ID(), nodeGradMap[node])
		if len(nodeGradMap[node]) > 1 {

			var n *Node
			symdiffLogf("reduce adding")
			if n, err = ReduceAdd(nodeGradMap[node], WithGroupName(gradClust)); err != nil {
				leaveLogScope()
				return nil, SymDiffError{
					single:  node,
					nodes:   nodeGradMap[node],
					gradMap: nodeGradMap,
					err:     errors.Wrap(err, "ReduceAdd failed during differentiation"),
				}

			}
			symdiffLogf("reduced to... %x", n.ID())
			// node.derives = append(node.derives, n)
			n.derivOf = append(n.derivOf, node)
			node.deriv = n
			nodeGradMap[node] = Nodes{n}
			// }
		} else if len(nodeGradMap[node]) == 1 {
			deriv := nodeGradMap[node][0]
			deriv.derivOf = append(deriv.derivOf, node)
			node.deriv = deriv
		}

		gradNode := nodeGradMap[node][0]
		if !node.isInput() {
			symdiffLogf("differentiating %x (%v)", node.ID(), node.op)
			enterLogScope()

			var op SDOp
			var childrenGrads Nodes
			var ok bool

			if op, ok = node.op.(SDOp); !ok {
				return nil, SymDiffError{
					single: node,
					err:    errors.New("Not a SymDifOp"),
				}
			}

			symdiffLogf("op: %v || optype: %v ||  node: %v || Children: %#Y || Grad: %v", node.op, node.op.Type(), node.t, node.children, gradNode)
			if childrenGrads, err = op.SymDiff(node.children, node, gradNode); err != nil {
				leaveLogScope()
				return nil, SymDiffError{
					single:  node,
					grad:    gradNode,
					gradMap: nodeGradMap,
					err:     errors.Wrapf(err, ".SymDiff() failed"),
				}
			}

			symdiffLogf("Derived(%d): %P", len(childrenGrads), childrenGrads)
			leaveLogScope()

			diffs := node.diffWRT()
			for i, child := range node.children {
				symdiffLogf("child is %v, i: %v", child, i)
				differentiable := diffs[i]
				childGrad := childrenGrads[i]

				if differentiable {
					childGrad.setGroup(gradClust)
					if grads, ok := nodeGradMap[child]; ok {
						grads = append(grads, childGrad)
						nodeGradMap[child] = grads
					} else {
						nodeGradMap[child] = Nodes{childGrad}
					}
				} else {
					symdiffLogf("Child %x is non differentiable", child.ID())
					if childGrad != nil {
						childGrad.setGroup(strayClust)
					}
				}
			}
		} else {
			symdiffLogf("iz input")
			symdiffLogf("%d ", nodeGradMap[node])
		}
		leaveLogScope()

	}
	leaveLogScope()
	// only we already summed up the gradients for the input nodes, so just take
	// 0th element
	for _, n := range wrt {
		symdiffLogf("nodeGradMap wrt: %d", nodeGradMap[n])
		retVal = append(retVal, nodeGradMap[n][0])
	}
	return
}

// SetDerivOf is used to hack around the fundamental limitations of Gorgonia.
//
// Specifically it is used to set a node as the derivative of another node,
// used in the cuDNN version of batch norm.
//
// The cuDNN BatchNorm operation produces the derivatives for the scale and bias as a side effect
// of calculating the derivative of the input. Because Gorgonia's Ops are modelled as pure functions (and no tuples)
// this causes a bit of trouble. With the clever use of scratch space ops multireturn can be simulated.
// But this causes derivatives to not be set correctly.
func SetDerivOf(deriv, of *Node) {
	deriv.derivOf = append(deriv.derivOf, of)
	of.deriv = deriv
}
