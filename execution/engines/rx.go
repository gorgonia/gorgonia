package engines

import (
	"context"
	"fmt"
	"sync"

	"gorgonia.org/gorgonia/exprgraph"
	"gorgonia.org/gorgonia/internal/datatypes"
	"gorgonia.org/gorgonia/ops"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/tensor"
)

type identityLift struct{}

func (_ identityLift) Lift(a datatypes.Tensor) datatypes.Tensor { return a }

type Notifier interface {
	NotifyUpdated(datatypes.Tensor)
}

type obs struct {
	n      exprgraph.Node
	ctx    context.Context
	cancel context.CancelFunc
}

// RxEngine is a reactive engine for Gorgonia
type RxEngine[DT any, T tensor.Basic[DT]] struct {
	StandardEngine[DT, T]
	g        *exprgraph.Graph
	q        chan obs // queue of nodes to be updated
	lifter   exprgraph.Lifter
	isLifter bool

	l             sync.Mutex
	current       context.Context
	cancelCurrent context.CancelFunc // cancelCurrent is the cancel function for the top level "job" (which is a `Let``)
	wg            sync.WaitGroup     // cannot be embedded because it has a .Add() method, which confuses Go

}

func NewRx[DT tensor.Num, T tensor.Basic[DT]](e StandardEngine[DT, T], g *exprgraph.Graph) *RxEngine[DT, T] {
	if e == nil {
		panic("Pass in a StandardEngine")
	}
	eng := &RxEngine[DT, T]{
		StandardEngine: e,
	}

	if g == nil {
		g = exprgraph.NewGraph(eng)
	} else {
		g.Engine = eng
	}
	eng.lifter = identityLift{}
	if lifter, ok := e.(exprgraph.Lifter); ok {
		eng.lifter = lifter
		eng.isLifter = true
	}
	eng.g = g
	eng.q = make(chan obs, 1024)
	go eng.loop()
	return eng
}

func (e *RxEngine[DT, T]) BasicEng() tensor.Engine {
	return &RxEngine[DT, tensor.Basic[DT]]{
		StandardEngine: e.StandardEngine.BasicEng().(StandardEngine[DT, tensor.Basic[DT]]),
		g:              e.g,
		q:              e.q,
	}
}

func (e *RxEngine[DT, T]) Graph() *exprgraph.Graph { return e.g }

func (e *RxEngine[DT, T]) SetGraph(g *exprgraph.Graph) { e.g = g }

// Lift implements lift only iff the underlying StandardEngine is a Lifter.
func (e *RxEngine[DT, T]) Lift(a datatypes.Tensor) datatypes.Tensor {
	return e.lifter.Lift(a)
	// if lifter, ok := e.StandardEngine.(exprgraph.Lifter); ok {
	// 	return lifter.Lift(a)
	// }
	// return a
}

// NotifyUpdated tells the engine that `a` has been updated.
func (e *RxEngine[DT, T]) NotifyUpdated(a datatypes.Tensor) {
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
func (e *RxEngine[DT, T]) Wait() { e.wg.Wait() }

// loop is the main loop for doing things. It pick nodes up from the `e.q` channel, and then
// flow the data up and down the graph.
func (e *RxEngine[DT, T]) loop() {
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
func (e *RxEngine[DT, T]) flowUp(ctx context.Context, n exprgraph.Node) int {
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
// The criteria for recomputation is in the .Waiting() method of a `Node`.
func (e *RxEngine[DT, T]) flowDown(ctx context.Context, node exprgraph.Node) error {
	n := node.(exprgraph.RxNode)
	childrenOfN := e.g.ChildrenOfAsNodes(node)
	children := exprgraph.TsFromNodes[exprgraph.RxNode](childrenOfN)

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

	childValues := make([]T, 0, len(children))
	for _, c := range children {
		v := exprgraph.T2B[DT](c)
		childValues = append(childValues, v.(T))
	}

	switch o := n.O().(type) {
	case ops.PreallocOp[DT, T]:
		v := exprgraph.T2B[DT](n)
		if _, err := o.PreallocDo(ctx, v.(T), childValues...); err != nil {
			n.AddWaiting()
			return err
		}
	default:
		// TODO: non PreallocOp
	}
	return nil

}

// LetRx is Let() but for reactive engine.
func LetRx[DT any](a datatypes.Tensor, v values.Value[DT]) {
	// do Let
	switch at := a.(type) {
	case exprgraph.ValueNode:
		av := at.V().(values.Value[DT])
		values.Copy[DT](av, v) // in real life you gotta return error
	case values.Value[DT]:
		values.Copy[DT](at, v)
	default:
		fmt.Printf("Cannot do Let %s %T\n%v", a, a, a)
	}

	// do reactive things
	eng := a.Engine().Workhorse()
	switch e := eng.(type) {
	case Notifier:
		e.NotifyUpdated(a)
	default:
		fmt.Printf("ENGINE %T NOT HANDLED\n", eng)
	}
}
