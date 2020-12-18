package exprgraph

import "gonum.org/v1/gonum/graph"

type edge struct{ from, to NodeID }

func (e edge) From() graph.Node         { return e.from }
func (e edge) To() graph.Node           { return e.to }
func (e edge) ReversedEdge() graph.Edge { e.from, e.to = e.to, e.from; return e }
func (e edge) Weight() float64          { return 0 }
