package xvm

import (
	"log"

	"gorgonia.org/gorgonia"
)

// GoMachine is a computation VM for Gorgonia.
// Every edge of the graph is associated with a channel of Value.
// The channels are identified by two IDs, tail and head, which are the IDs of the starting node and the ending node.
//
// Every node with a non-nil Op launches a goroutine.
//
// Each goroutine is expecting Values from all of its input channels (those with a tail matching the current node's ID).
// Then it calls the Do method of the operator, sets the own node's Value (thanks to the `Let` function),
// and sends the Value to the output channel (the channels with a head matching the current node'ID).
//
// Every input *Node, sends its Value to the channel with a tail matching its node ID and head matching a constant negative value.
type GoMachine struct {
	g  *gorgonia.ExprGraph
	db *chanDB
}

// RunAll triggers all the goroutines and wait for the all the output channel to be filled with a value.
//
// Caution: there is no safety mechanism, and this method would never return (deadlock) in some circumstances.
func (g *GoMachine) RunAll() error {
	g.populateChanDB()
	nodesIt := g.g.Nodes()
	for nodesIt.Next() {
		currentNode := nodesIt.Node().(*gorgonia.Node)
		// run all the nodes carrying an Op inside a go-routine
		outputC := g.db.getAllFromHead(currentNode.ID())
		switch {
		case currentNode.Op() != nil:
			children := g.g.From(currentNode.ID())
			inputC := make([]<-chan gorgonia.Value, children.Len())
			for i := 0; children.Next(); i++ {
				child := children.Node()
				var ok bool
				inputC[i], ok = g.db.getChan(currentNode.ID(), child.ID())
				if !ok {
					log.Fatal("chan edge not found")
				}
			}
			go opWorker(currentNode, inputC, outputC)
			// Send the input to the self nodes...
		case currentNode.Op() == nil && currentNode.Value() != nil:
			go valueFeeder(currentNode, outputC)
		default:
			log.Fatal("Yerk?")
		}
	}
	// wait for all values to be computed
	for _, outputC := range g.db.getAllFromTail(g.db.outputNodeID) {
		<-outputC
	}
	return nil
}

// Reset close all communication channels and created a new channel dictionary
func (g *GoMachine) Reset() {
	g.db.closeAll()
	g.db = newChanDB()
}

// Close all channels
func (g *GoMachine) Close() error {
	g.db.closeAll()
	return nil
}

// NewGoMachine creates a new VM able to run a program in a concurrent way.
// by now, only forward pass is supported
func NewGoMachine(g *gorgonia.ExprGraph) *GoMachine {
	return &GoMachine{
		g:  g,
		db: newChanDB(),
	}
}

func opWorker(n *gorgonia.Node, inputC []<-chan gorgonia.Value, outputC []chan<- gorgonia.Value) {
	vals := make([]gorgonia.Value, len(inputC))
	for i := range inputC {
		vals[i] = <-inputC[i]
	}
	output, err := n.Op().Do(vals...)
	if err != nil {
		log.Fatal(err)
	}
	gorgonia.UnsafeLet(n, output)
	for i := range outputC {
		outputC[i] <- output
	}
}

func valueFeeder(n *gorgonia.Node, feedC []chan<- gorgonia.Value) {
	for i := range feedC {
		feedC[i] <- n.Value()
	}
}

func (g *GoMachine) populateChanDB() error {
	edgesIt := g.g.Edges()
	for edgesIt.Next() {
		currentEdge := edgesIt.Edge()
		head := currentEdge.From().ID()
		tail := currentEdge.To().ID()
		g.db.upsert(make(chan gorgonia.Value, 0), tail, head)
	}
	nodesIt := g.g.Nodes()
	for nodesIt.Next() {
		currentNode := nodesIt.Node().(*gorgonia.Node)
		if g.g.From(currentNode.ID()).Len() == 0 {
			// Node is an input
			g.db.upsert(make(chan gorgonia.Value, 0), currentNode.ID(), g.db.inputNodeID)
		}
		if g.g.To(currentNode.ID()).Len() == 0 {
			// Node is an output
			g.db.upsert(make(chan gorgonia.Value, 0), g.db.outputNodeID, currentNode.ID())
		}
	}
	return nil
}
