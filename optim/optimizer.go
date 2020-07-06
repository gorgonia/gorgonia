// optimizer is a package that provides intrinsics for graph optimization

package optimizer

import "gorgonia.org/gorgonia/exprgraph"

type Option interface {
	Pattern() Pattern
	Replacement() expr.Node
}

// Pattern represents a pattern in the graph.
// These conditions are AND'd.
//
// e.g. Let us consider the following pattern:
//
// 	p := Pattern{
//		Pair: {Node: exprgraph.Cons(nil, "x+y", nil), MatchFlag: MatchName},
// 		Children: []Pair{
//			{Node: exprgraph.Cons(nil, "x", T), MatchFlag: MatchName&MatchValue},
//		}
// 	}
//
// When passed into `Optimize`, the optimizer will look for each node
// to see if the following are true:
//
// 	- There is a node with "x+y" as a name
// 	- AND (the node has a child whose name is "x" and has the value "T")
//
// If it is, then it will be returned as a Match.
//
// As in all things in life, products are a lot simpler than unions. So there are no facilities for OR'd patterns.
type Pattern struct {
	Pair
	Children []Pair
	Parents  []Pair
}

func (p Pattern) Matches(node exprgraph.Node, in *exprgraph.Graph) (Match, bool) {
	panic("NYI")
}

// Pair represents a Node with a matchflag.
type Pair struct {
	exprgraph.Node
	MatchFlag
}

// Match represents a match found.
type Match struct {
	exprgraph.Node
	Children []exprgraph.Node
	Parents  []exprgraph.Node
}

type MatchFlag byte

const (
	MatchOp MatchFlag = iota
	MatchName
	MatchValue
)

func Optimize(g *exprgraph.Graph, opts ...Option) *exprgraph.Graph {
	for _, n := range g.AllNodes() {
		for _, o := range opts {
			p := o.Pattern()
			if m, ok := p.Matches(n, g); ok {
				replace(g, m, o.Replacement())
			}
		}
	}
	return g // TODO BAD
}

func replace(g *exprgraph.Graph, match Match, replacement Node) {
}
