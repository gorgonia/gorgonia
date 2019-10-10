package xvm

import (
	"log"

	"gorgonia.org/gorgonia"
)

const (
	inputNode  int64 = -1
	outputNode int64 = -2
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

type chanDB struct {
	// map[tail][head]
	dico map[int64]map[int64]chan gorgonia.Value
	// map[head][tail]
	reverseDico map[int64]map[int64]chan gorgonia.Value
}

func (c *chanDB) closeAll() {
	for i := range c.dico {
		for j := range c.dico[i] {
			close(c.dico[i][j])
		}
	}
}

// upsert the channel to the DB, if id already exists it is overwritten
func (c *chanDB) upsert(channel chan gorgonia.Value, tail, head int64) {
	if _, ok := c.dico[tail]; !ok {
		c.dico[tail] = make(map[int64]chan gorgonia.Value, 0)
	}
	if _, ok := c.reverseDico[head]; !ok {
		c.reverseDico[head] = make(map[int64]chan gorgonia.Value, 0)
	}
	c.dico[tail][head] = channel
	c.reverseDico[head][tail] = channel
}

func newChanDB() *chanDB {
	return &chanDB{
		dico:        make(map[int64]map[int64]chan gorgonia.Value, 0),
		reverseDico: make(map[int64]map[int64]chan gorgonia.Value, 0),
	}
}

func (c *chanDB) getAllFromTail(tail int64) []chan gorgonia.Value {
	edges, ok := c.dico[tail]
	if !ok {
		return nil
	}
	output := make([]chan gorgonia.Value, 0, len(edges))
	for _, edge := range edges {
		output = append(output, edge)
	}
	return output
}

func (c *chanDB) getAllFromHead(head int64) []chan<- gorgonia.Value {
	edges, ok := c.reverseDico[head]
	if !ok {
		return nil
	}
	output := make([]chan<- gorgonia.Value, 0, len(edges))
	for _, edge := range edges {
		output = append(output, edge)
	}
	return output
}

func (c *chanDB) getChan(tail, head int64) (chan gorgonia.Value, bool) {
	v, ok := c.dico[tail][head]
	return v, ok
}

func (c *chanDB) len() int {
	return len(c.dico)
}

// RunAll triggers all the goroutines and wait for the all the output channel to be filled with a value.
//
// Caution: there is no safety mechanism, and this method would never return (deadlock) in some circumstances.
func (g *GoMachine) RunAll() error {
	nodesIt := g.g.Nodes()
	if g.db.len() == 0 {
		edgesIt := g.g.Edges()
		for edgesIt.Next() {
			currentEdge := edgesIt.Edge()
			head := currentEdge.From().ID()
			tail := currentEdge.To().ID()
			g.db.upsert(make(chan gorgonia.Value, 0), tail, head)
		}
		for nodesIt.Next() {
			currentNode := nodesIt.Node().(*gorgonia.Node)
			if g.g.From(currentNode.ID()).Len() == 0 {
				// Node is an input
				g.db.upsert(make(chan gorgonia.Value, 0), currentNode.ID(), inputNode)
			}
			if g.g.To(currentNode.ID()).Len() == 0 {
				// Node is an output
				g.db.upsert(make(chan gorgonia.Value, 0), outputNode, currentNode.ID())
			}
		}
		nodesIt.Reset()
	}
	for nodesIt.Next() {
		currentNode := nodesIt.Node().(*gorgonia.Node)
		// run all the nodes carrying an Op inside a go-routine
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
			go g.opWorker(currentNode, inputC, g.db.getAllFromHead(currentNode.ID()))
			// Send the input to the self nodes...
		case currentNode.Op() == nil && currentNode.Value() != nil:
			go g.valueFeeder(currentNode, g.db.getAllFromHead(currentNode.ID()))
		default:
			log.Fatal("Yerk?")
		}
	}
	// wait for all values to be computed
	for _, outputC := range g.db.getAllFromTail(outputNode) {
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

func (g *GoMachine) opWorker(n *gorgonia.Node, inputC []<-chan gorgonia.Value, outputC []chan<- gorgonia.Value) {
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

func (g *GoMachine) valueFeeder(n *gorgonia.Node, feedC []chan<- gorgonia.Value) {
	for i := range feedC {
		feedC[i] <- n.Value()
	}
}
