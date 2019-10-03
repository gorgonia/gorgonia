package gorgonia

import (
	"log"

	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/iterator"
)

const (
	inputNode  int64 = -1
	outputNode int64 = -2
)

type GoMachine struct {
	g  *ExprGraph
	db *chanDB
}

type chanDB struct {
	// map[tail][head]
	dico map[int64]map[int64]chan Value
	// map[head][tail]
	reverseDico map[int64]map[int64]chan Value
}

func (c *chanDB) closeAll() {
	for i := range c.dico {
		for j := range c.dico[i] {
			close(c.dico[i][j])
		}
	}
}

// upsert the channel to the DB, if id already exists it is overwritten
func (c *chanDB) upsert(channel chan Value, tail, head int64) {
	if _, ok := c.dico[tail]; !ok {
		c.dico[tail] = make(map[int64]chan Value, 0)
	}
	if _, ok := c.reverseDico[head]; !ok {
		c.reverseDico[head] = make(map[int64]chan Value, 0)
	}
	c.dico[tail][head] = channel
	c.reverseDico[head][tail] = channel
}

func newChanDB() *chanDB {
	return &chanDB{
		dico:        make(map[int64]map[int64]chan Value, 0),
		reverseDico: make(map[int64]map[int64]chan Value, 0),
	}
}

func (c *chanDB) getAllFromTail(tail int64) []chan Value {
	edges, ok := c.dico[tail]
	if !ok {
		return nil
	}
	output := make([]chan Value, 0, len(edges))
	for _, edge := range edges {
		output = append(output, edge)
	}
	return output
}

func (c *chanDB) getAllFromHead(head int64) []chan Value {
	edges, ok := c.reverseDico[head]
	if !ok {
		return nil
	}
	output := make([]chan Value, 0, len(edges))
	for _, edge := range edges {
		output = append(output, edge)
	}
	return output
}

func (c *chanDB) getChan(tail, head int64) (chan Value, bool) {
	v, ok := c.dico[tail][head]
	return v, ok
}

func (g *GoMachine) RunAll() error {
	edgesIt := getEdges(g.g)
	for edgesIt.Next() {
		currentEdge := edgesIt.Edge()
		head := currentEdge.From().ID()
		tail := currentEdge.To().ID()
		g.db.upsert(make(chan Value, 0), tail, head)
	}
	nodesIt := g.g.Nodes()
	for nodesIt.Next() {
		currentNode := nodesIt.Node().(*Node)
		if g.g.From(currentNode.ID()).Len() == 0 {
			// Node is an input
			g.db.upsert(make(chan Value, 0), currentNode.ID(), inputNode)
		}
		if g.g.To(currentNode.ID()).Len() == 0 {
			// Node is an output
			g.db.upsert(make(chan Value, 0), outputNode, currentNode.ID())
		}
	}
	nodesIt.Reset()
	for nodesIt.Next() {
		currentNode := nodesIt.Node().(*Node)
		// run all the nodes carrying an Op inside a go-routine
		switch {
		case currentNode.Op() != nil:
			go func(n *Node) {
				children := currentNode.children
				vals := make([]Value, len(children))
				inputC := make([]chan Value, len(children))
				for i, child := range children {
					var ok bool
					inputC[i], ok = g.db.getChan(currentNode.ID(), child.ID())
					if !ok {
						log.Fatal("chan edge not found")
					}
				}
				//inputC := g.db.getAllFromTail(currentNode.ID())
				//vals := make([]Value, len(inputC))
				for i := range inputC {
					//fmt.Printf("[%v] %v ==> get value i=%v chan=%v\n", n.ID(), n.Op(), i, inputC[i])
					vals[i] = <-inputC[i]
					//fmt.Printf("[%v] %v ==> get value i=%v val.Shape()=%v chan=%v\n", n.ID(), n.Op(), i, vals[i].Shape(), inputC[i])
				}
				//fmt.Printf("[%v] %v ==> start\n", n.ID(), n.Op())
				//fmt.Printf("[%v] %v ==> overwrites %v\n", n.ID(), n.Op(), n.Op().OverwritesInput())

				output, err := n.Op().Do(vals...)
				if err != nil {
					log.Fatal(err)
				}
				//fmt.Printf("[%v] %v ==> stop\n", n.ID(), n.Op())
				for _, c := range g.db.getAllFromHead(currentNode.ID()) {
					n.boundTo = output
					//fmt.Printf("[%v] %v ==> sending output value %v into chan %v\n", n.ID(), n.Op(), output.Shape(), c)
					c <- output
				}
			}(currentNode)
			// Send the input to the self nodes...
		case currentNode.Value() != nil:
			go func(n *Node) {
				for _, inputC := range g.db.getAllFromHead(currentNode.ID()) {
					inputC <- currentNode.Value()
				}
			}(currentNode)
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

func (g *GoMachine) Reset() {
	g.db.closeAll()
	g.db = newChanDB()
}

func (g *GoMachine) Close() error {
	g.db.closeAll()
	return nil
}

// NewGoMachine creates a new VM able to run a program in a concurrent way.
// by now, only forward pass is supported
func NewGoMachine(g *ExprGraph) *GoMachine {
	return &GoMachine{
		g:  g,
		db: newChanDB(),
	}
}

func getEdges(g *ExprGraph) graph.Edges {
	var edges []graph.Edge
	for _, n := range g.all {
		for _, toN := range g.to[n] {
			edges = append(edges, edge{
				from: n,
				to:   toN,
			})
		}
	}
	if len(edges) == 0 {
		return graph.Empty
	}
	return iterator.NewOrderedEdges(edges)
}
