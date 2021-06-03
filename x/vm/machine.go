package xvm

import (
	"context"
	"strconv"
	"strings"
	"time"

	"gorgonia.org/gorgonia"
)

// Machine is a top-level struture that will coordinate the execution of a graph
type Machine struct {
	nodes  []*node
	pubsub *pubsub
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
			nn = newOp(n, !topNode)
		}
		nodes = append(nodes, nn)
	}
	m := &Machine{
		nodes: nodes,
	}
	m.pubsub = createNetwork(nodes, g)
	return m
}

// createNetwork instantiate all the channels and create the pubsubs
func createNetwork(ns []*node, g *gorgonia.ExprGraph) *pubsub {
	ids := make(map[int64]*node, len(ns))
	for i := range ns {
		ids[ns[i].id] = ns[i]
	}
	ps := &pubsub{
		publishers:  make([]*publisher, 0),
		subscribers: make([]*subscriber, 0),
	}
	// Deal with publishers
	publishers := make(map[int64]*publisher, len(ns))
	for i := range ns {
		currNode := ns[i]
		if currNode.outputC == nil {
			continue
		}
		publisher := &publisher{
			id:          currNode.id,
			publisher:   currNode.outputC,
			subscribers: make([]chan<- gorgonia.Value, 0),
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
		subscriber := &subscriber{
			id:         currNode.id,
			subscriber: currNode.inputC,
			publishers: make([]<-chan gorgonia.Value, from.Len()),
		}
		for i := 0; from.Next(); i++ {
			pub := publishers[from.Node().ID()]
			c := make(chan gorgonia.Value, 0)
			pub.subscribers = append(pub.subscribers, c)

			subscriber.publishers[i] = c
		}
		ps.subscribers = append(ps.subscribers, subscriber)
	}
	return ps
}

// Run the computation
func (m *Machine) Run(ctx context.Context) error {
	cancel, wg := m.pubsub.run(ctx)
	err := m.runAllNodes(ctx)
	cancel()
	// wait for the infrastructure to settle
	wg.Wait()
	return err
}

// Close all the plumbing to avoid leaking
func (m *Machine) Close() {
	allChans := make(map[chan<- gorgonia.Value]struct{}, 0)
	for _, pub := range m.pubsub.publishers {
		for i := range pub.subscribers {
			allChans[pub.subscribers[i]] = struct{}{}
		}
	}
	for ch := range allChans {
		close(ch)
	}
	for _, n := range m.nodes {
		if n.inputC != nil {
			close(n.inputC)
		}
		if n.outputC != nil {
			close(n.outputC)
		}
	}
}

type nodeError struct {
	id  int64
	t   time.Time
	err error
}

type nodeErrors []nodeError

func (e nodeErrors) Error() string {
	var sb strings.Builder
	for _, e := range e {
		sb.WriteString(strconv.Itoa(int(e.id)))
		sb.WriteString(":")
		sb.WriteString(e.err.Error())
		sb.WriteString("\n")
	}
	return sb.String()
}

// Run performs the computation
func (m *Machine) runAllNodes(ctx context.Context) error {
	ctx, cancel := context.WithCancel(ctx)
	errC := make(chan nodeError, 0)
	total := len(m.nodes)
	for i := range m.nodes {
		go func(n *node) {
			err := n.Compute(ctx)
			errC <- nodeError{
				id:  n.id,
				t:   time.Now(),
				err: err,
			}
		}(m.nodes[i])
	}
	errs := make([]nodeError, 0)
	for e := range errC {
		total--
		if e.err != nil {
			errs = append(errs, e)
			// failfast, on error, cancel
			cancel()
		}
		if total == 0 {
			break
		}
	}
	cancel()
	close(errC)
	if len(errs) != 0 {
		return nodeErrors(errs)
	}
	return nil
}

// GetResult stored in a node
func (m *Machine) GetResult(id int64) gorgonia.Value {
	for i := range m.nodes {
		if m.nodes[i].id == id {
			return m.nodes[i].output
		}
	}
	return nil
}
