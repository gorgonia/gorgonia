package xvm

import (
	"context"

	"gorgonia.org/gorgonia"
)

// Machine is a top-level struture that will coordinate the execution of a graph
type Machine struct {
	nodes   []*node
	pubsubs []*pubsub
}

// NewMachine creates an exeuction machine from an exprgraph
func NewMachine(g *gorgonia.ExprGraph) *Machine {
	if g == nil {
		return nil
	}
	nodes := make([]*node, 0)
	nodesIte := g.Nodes()
	for nodesIte.Next() {
		n := nodesIte.Node().(*gorgonia.Node)
		if n.Op() != nil {
			nodes = append(nodes, &node{
				op: n.Op(),
			})
		}
	}
	return &Machine{
		nodes: nodes,
	}
}

// Run performs the computation
func (m *Machine) runAllNodes(ctx context.Context) error {
	ctx, cancel := context.WithCancel(ctx)
	errC := make(chan error, 0)
	total := len(m.nodes)
	for i := range m.nodes {
		go func(n *node) {
			err := n.Compute(ctx)
			errC <- err
		}(m.nodes[i])
	}
	var err error
	for err = range errC {
		total--
		if err != nil || total == 0 {
			break
		}
	}
	cancel()
	close(errC)
	return err
}

func (m *Machine) runAllPubSub(ctx context.Context) {
	for _, p := range m.pubsubs {
		go p.run(ctx)
	}
}
