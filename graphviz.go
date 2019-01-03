// This file holds all the mess relative to graphviz

package gorgonia

import (
	"bytes"
	"fmt"

	"github.com/awalterschulze/gographviz"
)

// ToDot returns the graph as a graphviz compatible string
func (n *Node) ToDot() string {
	graphName := exprgraphClust

	g := gographviz.NewEscape()
	g.SetName(graphName)
	g.SetDir(true)

	g.AddAttr(exprgraphClust, "splines", "spline")
	g.AddAttr(exprgraphClust, "nodesep", "0.5")
	g.AddAttr(exprgraphClust, "ranksep", "1.2 equally")

	seen := make(map[*Node]string)
	n.dot(g, graphName, seen)

	return g.String()
}

// RestrictedToDot prints the graphviz compatible string but does not print the entire tree
// up and down indicates how many levels to look up, and how many levels to look down
func (n *Node) RestrictedToDot(up, down int) string {
	if n.g == nil {
		return n.ToDot()
	}

	g := n.g
	var ns, upQ, downQ Nodes

	//	up
	ns = Nodes{n}
	upQ = Nodes{n}
	for l := 0; l < up; l++ {
		origLen := len(upQ)
		for i := 0; i < origLen; i++ {
			qn := upQ[i]
			toQN := graphNodeToNode(g.To(qn.ID()))
			upQ = append(upQ, toQN...)
			ns = append(ns, toQN...)
		}
		upQ = upQ[origLen:]
	}

	// down
	downQ = Nodes{n}
	for d := 0; d < down; d++ {
		origLen := len(downQ)
		for i := 0; i < origLen; i++ {
			qn := downQ[i]
			downQ = append(downQ, qn.children...)
			ns = append(ns, qn.children...)
		}
		downQ = downQ[origLen:]
	}

	sg := g.subgraph(ns, false)

	n.ofInterest = true
	defer func() {
		n.ofInterest = false
	}()
	return sg.ToDot()
}

// dotString returns the ID of the node.
func (n *Node) dotString(g *gographviz.Escape, graphName string) string {
	var buf bytes.Buffer
	if err := exprNodeTempl.ExecuteTemplate(&buf, "node", n); err != nil {
		panic(err)
	}

	id := fmt.Sprintf("Node_%p", n)
	label := buf.String()
	attrs := map[string]string{
		"fontname": "monospace",
		"shape":    "none",
		"label":    label,
	}

	g.AddNode(graphName, id, attrs)
	return id
}

func (n *Node) dotCluster() string {
	var group string
	var isConst bool
	var isInput = n.isInput()

	if n.op != nil {
		_, isConst = n.op.(constant)
	}

	switch {
	case isConst:
		group = constantsClust
	case isInput:
		group = inputsClust
	case n.group == "":
		group = exprgraphClust
	default:
		group = n.group
	}
	return group
}

func (n *Node) dot(g *gographviz.Escape, graphName string, seen map[*Node]string) string {
	var id string
	var ok bool
	if id, ok = seen[n]; !ok {
		id = n.dotString(g, graphName)
		seen[n] = id
	} else {
		return id
	}

	for i, child := range n.children {
		childID := child.dot(g, graphName, seen)
		edgeAttrs := map[string]string{
			"taillabel":  fmt.Sprintf(" %d ", i+1),
			"labelfloat": "false",
		}

		g.AddPortEdge(id, id+":anchor:s", childID, childID+":anchor:n", true, edgeAttrs)
	}
	return id
}

// ToDot generates the graph in graphviz format. The use of this is to generate for the entire graph
// which may have multiple trees with different roots
// TODO: This is getting unwieldy. Perhaps refactor out into a ToDot(...Opt)?
func (g *ExprGraph) ToDot() string {
	gv := gographviz.NewEscape()
	gv.SetName(fullGraphName)
	gv.SetDir(true)

	gv.AddAttr(fullGraphName, "nodesep", "1")
	gv.AddAttr(fullGraphName, "ranksep", "1.5 equally")
	gv.AddAttr(fullGraphName, "rankdir", "TB")
	if len(g.byHash) > 100 {
		gv.AddAttr(fullGraphName, "nslimit", "3") // numiter=3*len(nodes)
		// gv.AddAttr(fullGraphName, "splines", "line") // ugly as sin.
	}

	groups := make(map[string]struct{})
	for h, n := range g.byHash {
		if n != nil {
			group := n.dotCluster()
			groups[group] = struct{}{}
			continue
		}
		// other wise it'se a clash of hash
		for _, n := range g.evac[h] {
			group := n.dotCluster()
			groups[group] = struct{}{}

		}
	}

	for grp := range groups {
		attrs := map[string]string{"label": grp}

		parentGraph := fullGraphName
		if grp == inputsClust || grp == constantsClust {
			parentGraph = inputConsts
			if !gv.IsSubGraph(inputConsts) {
				groupAttrs := map[string]string{"rank": "max"}
				gv.AddSubGraph(fullGraphName, inputConsts, groupAttrs)
			}
		}
		gv.AddSubGraph(parentGraph, "cluster_"+grp, attrs)
	}

	// for _, n := range g.byHash {
	for _, n := range g.all {
		group := n.dotCluster()
		n.dotString(gv, "cluster_"+group)
	}

	// for _, from := range g.byHash {
	for _, from := range g.all {
		for i, child := range from.children {
			if ok := g.all.Contains(child); !ok {
				// not in graph, so ignore it...
				continue
			}
			fromID := fmt.Sprintf("Node_%p", from)
			toID := fmt.Sprintf("Node_%p", child)

			edgeAttrs := map[string]string{
				"taillabel":  fmt.Sprintf(" %d ", i),
				"labelfloat": "false",
			}

			// we invert the from and to nodes for gradients, As the expressionGraph builds upwards from bottom, the gradient builds downwards.
			if from.group == gradClust && child.group == gradClust {
				edgeAttrs["dir"] = "back"
				gv.AddPortEdge(toID, toID+":anchor:s", fromID, fromID+":anchor:n", true, edgeAttrs)
			} else {
				gv.AddPortEdge(fromID, fromID+":anchor:s", toID, toID+":anchor:n", true, edgeAttrs)
			}
		}
	}

	// draw deriv lines
	if debugDerives {
		edgeAttrs := map[string]string{
			"style":      "dashed",
			"constraint": "false",
			"weight":     "999",
		}

		for _, n := range g.byHash {
			if n == nil {
				// collision found... what to do?
				continue
			}
			if n.derivOf != nil {
				id := fmt.Sprintf("Node_%p", n)
				for _, derivOf := range n.derivOf {
					if _, ok := g.to[derivOf]; !ok {
						continue
					}
					ofID := fmt.Sprintf("Node_%p", derivOf)
					// gv.AddPortEdge(id, ":anchor:w", ofID, ofID+":anchor:e", true, edgeAttrs)
					gv.AddEdge(id, ofID, true, edgeAttrs)
				}
			}
		}
	}

	// stupid invisible nodes to keep expressiongraph on the left
	subGAttrs := make(map[string]string)
	// subGAttrs.Add("rank", "max")
	gv.AddSubGraph(fullGraphName, outsideSubG, subGAttrs)

	attrs := map[string]string{
		"style": "invis",
	}
	gv.AddNode(outsideSubG, outsideRoot, attrs)

	outsides := []string{outsideRoot}
	var insides []string

	// build the inside and outside list
	if _, hasInputs := groups[inputsClust]; hasInputs {
		insides = append(insides, insideInputs)
		gv.AddNode("cluster_inputs", insideInputs, attrs)
	}

	if _, hasConst := groups[constantsClust]; hasConst {
		if len(insides) > 0 {
			outsides = append(outsides, outsideConsts)
			gv.AddNode(outsideSubG, outsideConsts, attrs)
		}
		insides = append(insides, insideConsts)
		gv.AddNode("cluster_constants", insideConsts, attrs)
	}

	if len(insides) > 0 {
		outsides = append(outsides, outsideExprG)
		gv.AddNode(outsideSubG, outsideExprG, attrs)
	}
	insides = append(insides, insideExprG)
	gv.AddNode("cluster_expressionGraph", insideExprG, attrs)

	for group := range groups {
		if group == exprgraphClust || group == constantsClust || group == inputsClust {
			continue
		}
		inside := "inside_" + group
		outside := "outside_" + group
		insides = append(insides, inside)
		outsides = append(outsides, outside)

		gv.AddNode(outsideSubG, outside, attrs)
		gv.AddNode("cluster_"+group, inside, attrs)
	}

	edgeAttrs := map[string]string{
		"style":      "invis",
		"weight":     "999",
		"constraint": "false",
	}
	for i, o := range outsides {
		// outside-inside
		gv.AddEdge(o, insides[i], true, edgeAttrs)

		if i > 0 {
			// outside-outside
			gv.AddEdge(outsides[i-1], o, true, edgeAttrs)

			// inside-inside
			gv.AddEdge(insides[i-1], insides[i], true, edgeAttrs)
		}
	}
	return gv.String()
}
