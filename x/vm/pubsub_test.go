package xvm

import (
	"context"
	"reflect"
	"testing"

	"gorgonia.org/gorgonia"
)

func Test_merge(t *testing.T) {
	fortyTwo := gorgonia.F32(42.0)
	fortyThree := gorgonia.F32(43.0)
	t.Run("context cancel without value", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		c := make(chan gorgonia.Value, 1)
		output := merge(ctx, c)
		cancel()
		<-output
	})
	t.Run("context cancel with one value", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		c := make(chan gorgonia.Value, 1)
		output := merge(ctx, c)
		c <- &fortyTwo
		out := <-output
		if !reflect.DeepEqual(fortyTwo.Data(), out.v.Data()) {
			t.Errorf("Expected %v, got %v", fortyTwo, out)
		}
		c <- &fortyTwo
		cancel()
	})
	t.Run("with one value", func(t *testing.T) {
		ctx := context.Background()
		c := make(chan gorgonia.Value, 1)
		output := merge(ctx, c)
		c <- &fortyTwo
		out := <-output
		if !reflect.DeepEqual(fortyTwo.Data(), out.v.Data()) {
			t.Errorf("Expected %v, got %v", fortyTwo, out)
		}
	})
	t.Run("2 channels with two values", func(t *testing.T) {
		ctx := context.Background()
		cs := make([]chan gorgonia.Value, 2)
		for i := range cs {
			// The size of the channels buffer controls how far behind the receivers
			// of the fanOut channels can lag the other channels.
			cs[i] = make(chan gorgonia.Value, 0)
		}
		output := merge(ctx, cs[0], cs[1])
		cs[1] <- &fortyThree
		cs[0] <- &fortyTwo
		out := <-output
		if !reflect.DeepEqual(fortyThree.Data(), out.v.Data()) {
			t.Errorf("Expected %v, got %v", fortyTwo, out)
		}
		if out.pos != 1 {
			t.Errorf("bad position, expected 0, got %v", out.pos)
		}
		out = <-output
		if !reflect.DeepEqual(fortyTwo.Data(), out.v.Data()) {
			t.Errorf("Expected %v, got %v", fortyThree, out)
		}
		if out.pos != 0 {
			t.Errorf("bad position, expected 0, got %v", out.pos)
		}
	})
}

func Test_fanOut(t *testing.T) {
	t.Run("context cancel without value", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		c := make(chan gorgonia.Value, 0)
		cs := fanOut(ctx, c, 1, 0)
		cancel()
		<-cs[0]
	})
	t.Run("context cancel with one value", func(t *testing.T) {
		fortyTwo := gorgonia.F32(42.0)
		ctx, cancel := context.WithCancel(context.Background())
		c := make(chan gorgonia.Value, 0)
		cs := fanOut(ctx, c, 1, 0)
		c <- &fortyTwo
		out := <-cs[0]
		if !reflect.DeepEqual(fortyTwo.Data(), out.Data()) {
			t.Errorf("Expected %v, got %v", fortyTwo, out)
		}
		c <- &fortyTwo
		cancel()
	})
	t.Run("two chans", func(t *testing.T) {
		fortyTwo := gorgonia.F32(42.0)
		ctx := context.Background()
		c := make(chan gorgonia.Value, 0)
		cs := fanOut(ctx, c, 2, 0)
		c <- &fortyTwo
		out := <-cs[0]
		if !reflect.DeepEqual(fortyTwo.Data(), out.Data()) {
			t.Errorf("Expected %v, got %v", fortyTwo, out)
		}
		out = <-cs[1]
		if !reflect.DeepEqual(fortyTwo.Data(), out.Data()) {
			t.Errorf("Expected %v, got %v", fortyTwo, out)
		}
	})

}

func Test_newPubsub(t *testing.T) {
	type args struct {
		subscribers []chan ioValue
		publishers  []<-chan gorgonia.Value
	}
	tests := []struct {
		name string
		args args
		want *pubsub
	}{
		{
			"simple",
			args{
				nil,
				nil,
			},
			&pubsub{},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := newPubsub(tt.args.subscribers, tt.args.publishers); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("newPubsub() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_pubsub_run(t *testing.T) {
	type fields struct {
		publishers  []<-chan gorgonia.Value
		subscribers []chan ioValue
	}
	type args struct {
		ctx context.Context
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		wantErr bool
	}{
		{
			"nil channel",
			fields{},
			args{
				context.Background(),
			},
			false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &pubsub{
				publishers:  tt.fields.publishers,
				subscribers: tt.fields.subscribers,
			}
			if err := p.run(tt.args.ctx); (err != nil) != tt.wantErr {
				t.Errorf("pubsub.run() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func Test_broadcast(t *testing.T) {
	t.Run("context cancel without value", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cs := make([]chan ioValue, 1)
		for i := range cs {
			// The size of the channels buffer controls how far behind the receivers
			// of the fanOut channels can lag the other channels.
			cs[i] = make(chan ioValue, 0)
		}
		c := make(<-chan ioValue, 0)
		broadcast(ctx, c, cs)
		cancel()
	})
	t.Run("context cancel without value", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cs := make([]chan ioValue, 1)
		for i := range cs {
			// The size of the channels buffer controls how far behind the receivers
			// of the fanOut channels can lag the other channels.
			cs[i] = make(chan ioValue, 0)
		}
		c := make(chan ioValue, 0)
		broadcast(ctx, c, cs)
		c <- ioValue{}
		cancel()
	})
	t.Run("broadcast ", func(t *testing.T) {
		fortyTwo := gorgonia.F32(42.0)
		ctx := context.Background()
		cs := make([]chan ioValue, 2)
		for i := range cs {
			// The size of the channels buffer controls how far behind the receivers
			// of the fanOut channels can lag the other channels.
			cs[i] = make(chan ioValue, 0)
		}
		c := make(chan ioValue, 0)
		broadcast(ctx, c, cs)
		val := ioValue{
			pos: 0,
			v:   &fortyTwo,
		}
		c <- val
		v0 := <-cs[0]
		v1 := <-cs[1]
		if !reflect.DeepEqual(v0, val) {
			t.Errorf("broadcast want %v, got %v", val, v0)
		}
		if !reflect.DeepEqual(v1, val) {
			t.Errorf("broadcast want %v, got %v", val, v1)
		}
	})
}
