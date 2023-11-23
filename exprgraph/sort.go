package exprgraph

import (
	"sort"

	"github.com/pkg/errors"
	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/topo"
)

// Sort topologically sorts *Graph: a root of graph will be listed first
// nodes are sorted using gonum's SortStabilized function.
//
// see https://godoc.org/gonum.org/v1/gonum/graph/topo#SortStabilized for more info
func Sort(g *Graph) (sorted []Node, err error) {
	var sortedN []graph.Node

	if sortedN, err = topo.SortStabilized(g, reverseLexical); err != nil {
		return nil, errors.Wrap(err, "sort failed")
	}

	for _, n := range sortedN {
		sorted = append(sorted, n.(Node))
	}
	return sorted, nil
}

// byID is a sort.Interface for the reverseLexical function.
// It's important to observe that the Less() method is not the usual Less function.
// Larger IDs are ordered before smaller IDs.
// This is so that a reverse is achieved (without having to call sort.Reverse).
type byID []graph.Node

func (ns byID) Len() int           { return len(ns) }
func (ns byID) Less(i, j int) bool { return ns[i].ID() > ns[j].ID() }
func (ns byID) Swap(i, j int)      { ns[i], ns[j] = ns[j], ns[i] }

func reverseLexical(a []graph.Node) { sort.Sort(byID(a)) }
