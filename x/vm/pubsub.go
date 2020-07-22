package xvm

import (
	"context"
	"sync"

	"gorgonia.org/gorgonia"
)

type publisher struct {
	id          int64
	publisher   <-chan gorgonia.Value
	subscribers []chan<- gorgonia.Value
}

type subscriber struct {
	id         int64
	publishers []<-chan gorgonia.Value
	subscriber chan<- ioValue
}

type pubsub struct {
	publishers  []*publisher
	subscribers []*subscriber
}

func (p *pubsub) run(ctx context.Context) (context.CancelFunc, *sync.WaitGroup) {
	var wg sync.WaitGroup
	ctx, cancel := context.WithCancel(ctx)
	for i := range p.publishers {
		wg.Add(1)
		go broadcast(ctx, &wg, p.publishers[i].publisher, p.publishers[i].subscribers...)
	}
	for i := range p.subscribers {
		wg.Add(1)
		go merge(ctx, &wg, p.subscribers[i].subscriber, p.subscribers[i].publishers...)
	}
	return cancel, &wg
}

func merge(ctx context.Context, globalWG *sync.WaitGroup, out chan<- ioValue, cs ...<-chan gorgonia.Value) {
	defer globalWG.Done()
	var wg sync.WaitGroup

	// Start an output goroutine for each input channel in cs.  output
	// copies values from c to out until c or done is closed, then calls
	// wg.Done.
	output := func(ctx context.Context, c <-chan gorgonia.Value, pos int) {
		defer wg.Done()
		for {
			select {
			case n := <-c:
				select {
				case out <- ioValue{
					pos: pos,
					v:   n,
				}:
				case <-ctx.Done():
					return
				}
			case <-ctx.Done():
				return
			}
		}
	}
	wg.Add(len(cs))
	for i, c := range cs {
		go output(ctx, c, i)
	}

	// Start a goroutine to close out once all the output goroutines are
	// done.  This must start after the wg.Add call.
	go func() {
		wg.Wait()
		//close(out)
	}()
}

func broadcast(ctx context.Context, globalWG *sync.WaitGroup, ch <-chan gorgonia.Value, cs ...chan<- gorgonia.Value) {
	defer globalWG.Done()
	for {
		select {
		case msg := <-ch:
			for _, c := range cs {
				select {
				case c <- msg:
				case <-ctx.Done():
					return
				}
			}
		case <-ctx.Done():
			return
		}
	}
}
