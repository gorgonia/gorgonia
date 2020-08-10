package xvm

import (
	"context"
	"reflect"
	"runtime"
	"time"
)

// Trace the nodes states
type Trace struct {
	//fmt.Println(runtime.FuncForPC(reflect.ValueOf(state).Pointer()).Name())
	StateFunction string
	ID            int64
	Start         time.Time
	End           time.Time `json:",omitempty"`
}

type chanTracerContextKey int

const (
	globalTracerContextKey chanTracerContextKey = 0
)

// WithTracing initializes a tracing channel and adds it to the context
func WithTracing(parent context.Context) (context.Context, <-chan Trace) {
	c := make(chan Trace, 0)
	return context.WithValue(parent, globalTracerContextKey, c), c
}

// CloseTracing the tracing channel to avoid context leak.
// it is a nil op if context does not carry tracing information
func CloseTracing(ctx context.Context) {
	c := extractTracingChannel(ctx)
	if c != nil {
		close(c)
	}
}

func extractTracingChannel(ctx context.Context) chan<- Trace {
	if ctx == nil {
		return nil
	}
	if c := ctx.Value(globalTracerContextKey); c != nil {
		return c.(chan Trace)
	}
	return nil
}

var now = time.Now

func trace(ctx context.Context, t *Trace, n *node, state stateFn) *Trace {
	traceC := extractTracingChannel(ctx)
	if traceC == nil {
		return t
	}
	if t == nil {
		t = &Trace{
			ID:            n.id,
			StateFunction: runtime.FuncForPC(reflect.ValueOf(state).Pointer()).Name(),
			Start:         now(),
		}
	} else {
		t.End = now()
	}
	select {
	case traceC <- *t:
	case <-ctx.Done():
	}
	return t
}
