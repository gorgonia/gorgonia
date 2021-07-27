package exprgraph_test

import (
	"context"
	"fmt"
	"sync"

	"gorgonia.org/gorgonia"
	"gorgonia.org/gorgonia/exprgraph"
	"gorgonia.org/gorgonia/ops"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/tensor"
)

var _ tensor.Adder = &RxEngine{}

type obs struct {
	n      *exprgraph.Node
	ctx    context.Context
	cancel context.CancelFunc
}

// RxEngine is a reactive engine for Gorgonia
type RxEngine struct {
	tensor.StdEng
	g *exprgraph.Graph
	q chan obs // queue of nodes to be updated

	l             sync.Mutex
	current       context.Context
	cancelCurrent context.CancelFunc // cancelCurrent is the cancel function for the top level "job" (which is a `Let``)
	wg            sync.WaitGroup     // cannot be embedded because it has a .Add() method, which confuses Go

}

func NewRx() *RxEngine {
	eng := &RxEngine{}
	g := exprgraph.NewGraph(eng)
	eng.g = g
	eng.q = make(chan obs, 1024)
	go eng.loop()
	return eng
}
func (e *RxEngine) Graph() *exprgraph.Graph { return e.g }

func (e *RxEngine) SetGraph(g *exprgraph.Graph) { e.g = g }

// NotifyUpdated tells the engine that `a` has been updated.
func (e *RxEngine) NotifyUpdated(a gorgonia.Tensor) {
	e.l.Lock()
	if e.cancelCurrent != nil {
		e.cancelCurrent()
		e.cancelCurrent = nil
	}
	ctx, cancel := context.WithCancel(context.Background())
	e.current = ctx
	e.cancelCurrent = cancel
	e.l.Unlock()
	g := e.Graph()
	n := g.NodeOf(a)
	e.wg.Add(1)
	e.q <- obs{n, ctx, cancel}
}

// Wait waits for the engine to finish updating.
func (e *RxEngine) Wait() { e.wg.Wait() }

// loop is the main loop for doing things. It pick nodes up from the `e.q` channel, and then
// flow the data up and down the graph.
func (e *RxEngine) loop() {
	for o := range e.q {

		n := o.n
		ctx := o.ctx
		cancel := o.cancel

		// compute all dependencies first
		if err := e.flowDown(ctx, n); err != nil {
			// ???? TODO
		}

		nonRootParents := e.flowUp(ctx, n)

		// if there are no more parents, then we are done
		if nonRootParents == 0 {
			cancel()
			e.wg.Done()
		}
	}
}

// flowUp makes the tensor flows upwards towards the root.
// Given a node, it computes the results of the parent node(s).
// If the parent node(s) themselves have parent node(s), those parent nodes will
// be placed into the queue.
func (e *RxEngine) flowUp(ctx context.Context, n *exprgraph.Node) int {
	var parents []*exprgraph.Node
	ns, ok := e.g.To(n.ID()).(*exprgraph.Nodes)
	if ok {
		parents = ns.NodeSlice()
	}
	var nonRootParents int
	for _, parent := range parents {
		if err := e.flowDown(ctx, parent); err != nil {
			continue
		}

		if e.g.To(parent.ID()).Len() != 0 {
			ctx2, cancel2 := context.WithCancel(ctx)
			e.q <- obs{parent, ctx2, cancel2}
			nonRootParents++
		}
	}
	return nonRootParents
}

// flowDown recomputes the value of `n`, and recomputes any of the children if need be.
// The criteria for recomputation is in the .Waiting() method of a `*Node`.
func (e *RxEngine) flowDown(ctx context.Context, n *exprgraph.Node) error {
	children := e.g.From(n.ID())
	if children.Len() == 0 {
		// done
		return nil
	}
	children.Reset()
	childNodes := children.(*exprgraph.Nodes).NodeSlice()

	// Depth first search.
	// TODO: maybe traverse and process the graph concurrently?
	for _, c := range childNodes {
		if c.Waiting() > 0 {
			// then it needs to be reprocessed
			ctx2, cancel2 := context.WithCancel(ctx)
			e.flowDown(ctx2, c)
			c.ZeroWaiting()
			cancel2()
		}
	}

	childValues := make([]values.Value, 0, len(childNodes))
	for _, n := range childNodes {
		childValues = append(childValues, n.Value())
	}

	switch o := n.Op.(type) {
	case ops.PreallocOp:
		if _, err := o.PreallocDo(ctx, n.Value(), childValues...); err != nil {
			n.AddWaiting()
			return err
		}
	default:
		// TODO: non PreallocOp
	}
	return nil

}

// LetRx is Let() but for reactive engine.
func LetRx(a gorgonia.Tensor, v values.Value) {
	// do Let
	switch at := a.(type) {
	case *exprgraph.Node:
		av := at.Value()
		values.Copy(av, v)
	case values.Value:
		values.Copy(at, v)
	}

	// do reactive things
	eng := a.Engine()
	switch e := eng.(type) {
	case *RxEngine:
		e.NotifyUpdated(a)
	}
}

func Example_rx_engine() {
	engine := NewRx()
	g := engine.Graph()
	x := exprgraph.NewNode(g, "x", tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))
	y := exprgraph.NewNode(g, "y", tensor.WithShape(3, 2), tensor.WithBacking([]float64{6, 5, 4, 3, 2, 1}))
	z := exprgraph.NewNode(g, "z", tensor.WithShape(), tensor.WithBacking([]float64{1}))

	xy, err := MatMul(x, y)
	if err != nil {
		fmt.Println(err)
	}
	xypz, err := Add(xy, z)
	if err != nil {
		fmt.Println(err)
	}

	fmt.Printf("x:\n%v\ny:\n%v\nxy:\n%v\nxy+z:\n%v\n", x, y, xy, xypz)

	// Update
	xv := tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{100, 200, 300, 400, 500, 600}))
	yv := tensor.New(tensor.WithShape(3, 2), tensor.WithBacking([]float64{60, 50, 40, 30, 20, 10}))
	zv := tensor.New(tensor.WithShape(), tensor.WithBacking([]float64{1010}))
	LetRx(x, xv)
	LetRx(y, yv)
	LetRx(z, zv)
	engine.Wait()
	fmt.Printf("After Updating\n-----\nx:\n%v\ny:\n%v\nxy:\n%v\nxy+z:\n%v\n", x, y, xy, xypz)

	// Output:
	// x:
	// ⎡1  2  3⎤
	// ⎣4  5  6⎦
	//
	// y:
	// ⎡6  5⎤
	// ⎢4  3⎥
	// ⎣2  1⎦
	//
	// xy:
	// ⎡20  14⎤
	// ⎣56  41⎦
	//
	// xy+z:
	// ⎡21  15⎤
	// ⎣57  42⎦
	//
	// After Updating
	// -----
	// x:
	// ⎡100  200  300⎤
	// ⎣400  500  600⎦
	//
	// y:
	// ⎡60  50⎤
	// ⎢40  30⎥
	// ⎣20  10⎦
	//
	// xy:
	// ⎡20000  14000⎤
	// ⎣56000  41000⎦
	//
	// xy+z:
	// ⎡21010  15010⎤
	// ⎣57010  42010⎦

}
