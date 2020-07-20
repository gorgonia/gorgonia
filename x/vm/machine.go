package xvm

import (
	"context"

	"gorgonia.org/gorgonia"
)

// Machine is a top-level struture that will coordinate the execution of a graph
type Machine struct {
	nodes   []*node
	pubsubs *pubsub
}

// NewMachine creates an exeuction machine from an exprgraph
func NewMachine(g *gorgonia.ExprGraph) *Machine {
	if g == nil {
		return nil
	}
	nodesIte := g.Nodes()
	nodes := make([]*node, 0, nodesIte.Len())
	for nodesIte.Next() {
		n := nodesIte.Node().(*gorgonia.Node)
		var nn *node
		topNode := g.To(n.ID()).Len() == 0
		op := n.Op()
		switch {
		case op == nil:
			nn = newInput(n)
		case op.Arity() == 0:
			nn = newInput(n)
		default:
			nn = newOp(n, topNode)
		}
		nodes = append(nodes, nn)
	}
	m := &Machine{
		nodes: nodes,
	}
	m.pubsubs = createNetwork(nodes, g)
	return m
}

// createNetwork instantiate all the channels and create the pubsubs
func createNetwork(ns []*node, g *gorgonia.ExprGraph) *pubsub {
	ids := make(map[int64]*node, len(ns))
	for i := range ns {
		ids[ns[i].id] = ns[i]
	}
	ps := &pubsub{
		publishers:  make([]publisher, 0),
		subscribers: make([]subscriber, 0),
	}
	// Deal with publishers
	publishers := make(map[int64]publisher, len(ns))
	for i := range ns {
		currNode := ns[i]
		if currNode.outputC == nil {
			continue
		}
		to := g.To(currNode.id)
		publisher := publisher{
			id:          currNode.id,
			publisher:   currNode.outputC,
			subscribers: make([]chan gorgonia.Value, to.Len()),
		}
		for to.Next() {
			publisher.subscribers = append(publisher.subscribers, ids[to.Node().ID()].outputC)
		}
		publishers[currNode.id] = publisher
		ps.publishers = append(ps.publishers, publisher)
	}
	// Deal with subscribers
	for i := range ns {
		currNode := ns[i]
		if currNode.inputC == nil {
			continue
		}
		from := g.From(currNode.id)
		subscriber := subscriber{
			id:         currNode.id,
			subscriber: currNode.inputC,
			publishers: make([]chan gorgonia.Value, from.Len()),
		}
		for from.Next() {
			subscriber.publishers = append(subscriber.publishers, publishers[from.Node().ID()].publisher)
		}
		ps.subscribers = append(ps.subscribers, subscriber)
	}
	return ps
}

// Run the computation
func (m *Machine) Run(ctx context.Context) error {
	cancel := m.pubsubs.run(ctx)
	err := m.runAllNodes(ctx)
	cancel()
	return err
}

// Close all the plumbing
func (m *Machine) Close() {
	for i := range m.pubsubs.publishers {
		pub := m.pubsubs.publishers[i]
		if pub.publisher != nil {
			close(pub.publisher)
		}
		for j := range pub.subscribers {
			sub := pub.subscribers[j]
			if sub != nil {
				close(sub)
			}
		}
	}
	for i := range m.pubsubs.subscribers {
		sub := m.pubsubs.subscribers[i]
		if sub.subscriber != nil {
			close(sub.subscriber)
		}
		for j := range sub.publishers {
			pub := sub.publishers[j]
			if pub != nil {
				close(pub)
			}
		}
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
