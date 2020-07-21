package xvm

import (
	"context"
	"sync"

	"gorgonia.org/gorgonia"
)

type publisher struct {
	id          int64
	subscribed  int
	publisher   chan gorgonia.Value
	subscribers []chan gorgonia.Value
}

type subscriber struct {
	id         int64
	publishers []chan gorgonia.Value
	subscriber chan ioValue
}

type pubsub struct {
	publishers  []publisher
	subscribers []subscriber
}

func (p *pubsub) run(ctx context.Context) context.CancelFunc {
	ctx, cancel := context.WithCancel(ctx)
	for i := range p.publishers {
		go broadcast(ctx, p.publishers[i].publisher, p.publishers[i].subscribers)
	}
	for i := range p.subscribers {
		go merge(ctx, p.subscribers[i].publishers, p.subscribers[i].subscriber)
	}
	//broadcast(ctx, merge(ctx, p.publishers...), p.subscribers)
	return cancel
}

func merge(ctx context.Context, cs []chan gorgonia.Value, out chan ioValue) {
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
		close(out)
	}()
}

func fanOut(ctx context.Context, ch <-chan gorgonia.Value, size, lag int) []chan gorgonia.Value {
	cs := make([]chan gorgonia.Value, size)
	for i := range cs {
		// The size of the channels buffer controls how far behind the receivers
		// of the fanOut channels can lag the other channels.
		cs[i] = make(chan gorgonia.Value, lag)
	}
	go func() {
		for {
			select {
			case msg := <-ch:
				for _, c := range cs {
					select {
					case c <- msg:
					case <-ctx.Done():
						for _, c := range cs {
							// close all our fanOut channels when the input channel is exhausted.
							close(c)
						}
						return
					}
				}
			case <-ctx.Done():
				for _, c := range cs {
					// close all our fanOut channels when the input channel is exhausted.
					close(c)
				}
				return
			}
		}
	}()
	return cs
}
func broadcast(ctx context.Context, ch <-chan gorgonia.Value, cs []chan gorgonia.Value) {
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
