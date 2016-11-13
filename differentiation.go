package gorgonia

import "github.com/pkg/errors"

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
	enterLoggingContext()
	defer leaveLoggingContext()

	sane := outputs.AllSameGraph()
	if !sane {
		return nil, errors.New("The supplied output Nodes are not the same graph")
	}

	// diffSet := outputs.Set()
	diffSet := outputs.mapSet()

	symdiffLogf("Diff Set: %d", diffSet)
	symdiffLogf("%d", sortedNodes)
	// for i := len(sortedNodes) - 1; i >= 0; i-- {
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
	enterLoggingContext()
	defer leaveLoggingContext()

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

	enterLoggingContext()
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
		enterLoggingContext()
		symdiffLogf("Children: %v", n.children)
		if len(diffs) == 0 {
			// check if this makes nodes unreachable. If it does, then error out
			if n.isStmt {
				symdiffLogf("Statement nodes are Non differentiable!")
				leaveLoggingContext()
				continue
			} else if n.isInput() {
				symdiffLogf("Input nodes are Non differentiable")
				leaveLoggingContext()
				continue
			} else if len(n.children) == 0 {
				symdiffLogf("Leaf nodes have no children")
				continue
			}
			g := n.g
			for _, child := range n.children {
				parents := g.To(child)

				symdiffLogf("parents of %v: %v", child, graphNodeToNode(parents))
				if len(parents) == 1 && len(child.children) > 0 {
					return nil, errors.Errorf("Being unable to differentiate %v would leave a portion of the graph unreachable. Unable to continue", n)
				}
			}
			symdiffLogf("SKIPPING... Non differentiable!")
			continue
		}

	inner:
		for j, child := range n.children {
			d := diffs[j]

			// if _, ok := diffSet[child]; ok && d {
			// 	autodiffLogf("Adding %d to differentiable set", child.ID())
			// 	diffSet[n] = empty
			// 	break inner
			// }

			if diffSet.Contains(child) && d {
				symdiffLogf("Adding %x to differentiable set", child.ID())
				// diffSet = append(diffSet, n)
				diffSet.Add(n)
				break inner
			}
		}
		leaveLoggingContext()
	}
	leaveLoggingContext()
	// retVal = diffSet.Set()
	// for n := range diffSet {
	// 	retVal = append(retVal, n)
	// }
	// sort.Sort(retVal)
	// autodiffLogf("RetVal: %d", retVal)
	return diffSet, nil
}

func Backpropagate(outputs, gradOutputs, wrt Nodes) (retVal Nodes, err error) {
	symdiffLogf("BACKPROP START")
	symdiffLogf("Outputs: %d", outputs)
	symdiffLogf("gradOutputs: %d", gradOutputs)
	symdiffLogf("WRT: %d", wrt)

	enterLoggingContext()
	defer leaveLoggingContext()

	g := outputs[0].g

	// this entire section about removing foreveralone nodes need a rethink
	symdiffLogf("removing foreveralone nodes")
	enterLoggingContext()
	for i := 0; i < len(g.AllNodes()); i++ {
		n := g.AllNodes()[i]

		fr := len(g.From(n))
		to := len(g.To(n))

		if fr == 0 && to == 0 && !n.isConstant() && !n.isInput() {
			g.RemoveNode(n)
			symdiffLogf("removed %v(%p); %x; %s", n, n, n.ID(), n.Name())
		}
	}

	leaveLoggingContext()

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
		return nil, errors.Errorf("Non differentiable WRTs: %v", badWRTs)
	}

	outputSet := outputs.mapSet()
	badOutputs := outputSet.Difference(affectedByOutput)
	if len(badOutputs) > 0 {
		symdiffLogf("badOutputs: %#v", badOutputs)
		return nil, errors.Errorf("Non differentiable outputs: %v", badOutputs)
	}

	// map a node to a list of gradient terms
	// these  gradient terms will be summed up when we visit the node
	// when iterating through the nondes in reverse topological order
	nodeGradMap := make(map[*Node]Nodes)
	for i, n := range outputs {
		nodeGradMap[n] = Nodes{gradOutputs[i]}
	}

	// "active" nodes are the ones that are differentially influenced by the inputs
	// and also differentiably influence the outputs. These are the nodes where we need to call the
	// "pullback" function to backpropagate derivatives
	activeNodes := affectsOutput.Intersect(affectedByOutput)

	symdiffLogf("Active: %d", activeNodes)

	symdiffLogf("Sorted: %d", sortedNodes)
	symdiffLogf("nodeGradMap: %+#d", FmtNodeMap(nodeGradMap))
	enterLoggingContext()
	// for i := len(sortedNodes) - 1; i >= 0; i-- {
	// 	node := sortedNodes[i]
	for _, node := range sortedNodes {
		// if !activeNodes.Contains(node) {
		// 	autodiffLogf("skipping %d", node.ID())
		// 	continue
		// }
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
		enterLoggingContext()

		// Check if there is any grads coming into this node
		if len(nodeGradMap[node]) < 1 {
			return nil, errors.Errorf("No gradient node found for Node ID %x - %v", node.ID(), node)
		}

		// once we've reached a node, we already backpropagated from its dependents
		// so we sum up the gradients
		symdiffLogf("nodeGradMap[node]: %d", nodeGradMap[node])
		if len(nodeGradMap[node]) > 1 {

			var n *Node
			symdiffLogf("reduce adding")
			if n, err = ReduceAdd(nodeGradMap[node], WithGroupName(gradClust)); err != nil {
				return nil, errors.Wrap(err, "ReduceAdd failed during differentiation")
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
			enterLoggingContext()

			var op SDOp
			var childrenGrads Nodes
			var ok bool

			if op, ok = node.op.(SDOp); !ok {
				// error
			}

			symdiffLogf("op: %v || optype: %v ||  node: %v || Children: %#Y || Grad: %v", node.op, node.op.Type(), node.t, node.children, gradNode)
			if childrenGrads, err = op.SymDiff(node.children, node, gradNode); err != nil {
				return nil, errors.Wrapf(err, "SymDiff for %v. OpType: %v. Node Type: %v. Children: %#v. Grad: %v", node.op, node.op.Type(), node.t, node.children, gradNode)

			}
			symdiffLogf("Derived(%d): %d", len(childrenGrads), childrenGrads)
			leaveLoggingContext()

			diffs := node.diffWRT()
			for i, child := range node.children {
				symdiffLogf("child is %v, i: %v", child, i)
				differentiable := diffs[i]
				childGrad := childrenGrads[i]

				if differentiable {
					// node.derives = append(node.derives, childGrad)
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
		leaveLoggingContext()

	}
	leaveLoggingContext()
	// only we already summed up the gradients for the input nodes, so just take
	// 0th element
	for _, n := range wrt {
		symdiffLogf("nodeGradMap wrt: %d", nodeGradMap[n])
		retVal = append(retVal, nodeGradMap[n][0])
	}
	return
}
