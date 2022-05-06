package engines

import (
	"context"
	"sync"

	"gorgonia.org/gorgonia/exprgraph"
	"gorgonia.org/tensor"
)

// Rx is a reactive engine.
type Rx struct {
	tensor.StandardEngine
	g *exprgraph.Graph
	q chan obs // queue of nodes to be updated

	l             sync.Mutex
	current       context.Context
	cancelCurrent context.CancelFunc // cancelCurrent is the cancel function for the top level "job" (which is a `Let``)
	wg            sync.WaitGroup     // cannot be embedded because it has a .Add() method, which confuses Go
}

func (e *RxEngine) Graph() *exprgraph.Graph { return e.g }

func (e *RxEngine) SetGraph(g *exprgraph.Graph) { e.g = g }

// Lift implements lift only iff the underlying StandardEngine is a Lifter.
func (e *RxEngine) Lift(a gorgonia.Tensor) gorgonia.Tensor {
	if lifter, ok := e.StandardEngine.(exprgraph.Lifter); ok {
		return lifter.Lift(a)
	}
	return a
}

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
	parents := e.g.ParentsOfAsNodes(n)

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
	children := e.g.ChildrenOfAsNodes(n)

	// Depth first search.
	// TODO: maybe traverse and process the graph concurrently?
	for _, c := range children {
		if c.Waiting() > 0 {
			// then it needs to be reprocessed
			ctx2, cancel2 := context.WithCancel(ctx)
			e.flowDown(ctx2, c)
			c.ZeroWaiting()
			cancel2()
		}
	}

	childValues := make([]values.Value, 0, len(children))
	for _, n := range children {
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
