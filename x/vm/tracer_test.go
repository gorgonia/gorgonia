package xvm

import (
	"context"
	"fmt"
	"reflect"
	"testing"
	"time"

	"gorgonia.org/gorgonia"
)

func ExampleWithTracing() {
	ctx, tracingC := WithTracing(context.Background())
	defer CloseTracing(ctx)
	go func() {
		for t := range tracingC {
			fmt.Println(t)
		}
	}()
	g := gorgonia.NewGraph()
	// add operations etc...
	machine := NewMachine(g)
	defer machine.Close()
	machine.Run(ctx)
}

func TestWithTracing(t *testing.T) {
	ctx, c := WithTracing(context.Background())
	cn := ctx.Value(globalTracerContextKey)
	if cn == nil {
		t.Fail()
	}
	if cn.(chan Trace) != c {
		t.Fail()
	}
}

func Test_extractTracingChannel(t *testing.T) {
	ctx, _ := WithTracing(context.Background())
	c := ctx.Value(globalTracerContextKey).(chan Trace)
	type args struct {
		ctx context.Context
	}
	tests := []struct {
		name string
		args args
		want chan<- Trace
	}{
		{
			"nil",
			args{
				context.Background(),
			},
			nil,
		},
		{
			"ok",
			args{
				ctx,
			},
			c,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := extractTracingChannel(tt.args.ctx); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("extractTracingChannel() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestCloseTracing(t *testing.T) {
	ctx, _ := WithTracing(context.Background())
	type args struct {
		ctx context.Context
	}
	tests := []struct {
		name string
		args args
	}{
		{
			"no trace",
			args{
				context.Background(),
			},
		},
		{
			"trace",
			args{
				ctx,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			CloseTracing(tt.args.ctx)
		})
	}
}

func Test_trace(t *testing.T) {
	now = func() time.Time { return time.Date(1977, time.September, 10, 10, 25, 00, 00, time.UTC) }
	ctx, c := WithTracing(context.Background())
	defer CloseTracing(ctx)
	go func() {
		for range c {
		}
	}()
	type args struct {
		ctx   context.Context
		t     *Trace
		n     *node
		state stateFn
	}
	tests := []struct {
		name string
		args args
		want *Trace
	}{
		{
			"no tracing context",
			args{
				context.Background(),
				nil,
				nil,
				nil,
			},
			nil,
		},
		{
			"Context with nil trace",
			args{
				ctx,
				nil,
				&node{
					id: 0,
				},
				nil,
			},
			&Trace{
				ID:    0,
				Start: now(),
			},
		},
		{
			"Context existing trace",
			args{
				ctx,
				&Trace{
					ID:    1,
					Start: now(),
				},
				nil,
				nil,
			},
			&Trace{
				ID:    1,
				Start: now(),
				End:   now(),
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := trace(tt.args.ctx, tt.args.t, tt.args.n, tt.args.state); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("trace() = %v, want %v", got, tt.want)
			}
		})
	}
}
