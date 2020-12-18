package exprgraph

import "gonum.org/v1/gonum/graph"

// WeightedEdge is a simple weighted graph edge.
type WeightedEdge struct {
	F, T graph.Node
	W    float64
}

// From returns the from-node of the edge.
func (e WeightedEdge) From() graph.Node { return e.F }

// To returns the to-node of the edge.
func (e WeightedEdge) To() graph.Node { return e.T }

// ReversedEdge returns a new Edge with the F and T fields
// swapped. The weight of the new Edge is the same as
// the weight of the receiver.
func (e WeightedEdge) ReversedEdge() graph.Edge { return WeightedEdge{F: e.T, T: e.F, W: e.W} }

// Weight returns the weight of the edge.
func (e WeightedEdge) Weight() float64 { return e.W }
