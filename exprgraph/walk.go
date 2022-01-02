package exprgraph

type nodeset map[NodeID]struct{}

func (ns nodeset) Add(n int64) { ns[NodeID(n)] = struct{}{} }

// Walk returns a channel of Nodes. The nodes are populated from walking the graph.
// The walk is done concurrently in a DFS-ish fashion ("ish" is due to concurrency).
func Walk(g *Graph, start, end *Node) <-chan *Node {
	retVal := make(chan *Node)
	walked := make(nodeset)
	ch := make(chan *Node)
	go func() {
		walk(g, start.ID(), end.ID(), ch, walked)
	}()
	return retVal
}

func walk(g *Graph, start, end int64, ch chan *Node, walked nodeset) {
	if _, ok := walked[NodeID(start)]; ok {
		return // walked before
	}

	ch <- g.nodes[start]
	walked.Add(start)
	if start == end {
		return
	}

	for _, child := range g.from[start] {
		walk(g, child, end, ch, walked)
	}
}
