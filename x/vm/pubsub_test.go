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
		c := make([]chan gorgonia.Value, 1)
		c[0] = make(chan gorgonia.Value, 0)
		output := make(chan ioValue, 0)
		merge(ctx, c, output)
		cancel()
		<-output
	})
	t.Run("context cancel with one value", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		c := make([]chan gorgonia.Value, 1)
		c[0] = make(chan gorgonia.Value, 0)
		output := make(chan ioValue, 0)
		merge(ctx, c, output)
		c[0] <- &fortyTwo
		out := <-output
		if !reflect.DeepEqual(fortyTwo.Data(), out.v.Data()) {
			t.Errorf("Expected %v, got %v", fortyTwo, out)
		}
		c[0] <- &fortyTwo
		cancel()
	})
	t.Run("with one value", func(t *testing.T) {
		ctx := context.Background()
		c := make([]chan gorgonia.Value, 1)
		c[0] = make(chan gorgonia.Value, 0)
		output := make(chan ioValue, 0)
		merge(ctx, c, output)
		c[0] <- &fortyTwo
		out := <-output
		if !reflect.DeepEqual(fortyTwo.Data(), out.v.Data()) {
			t.Errorf("Expected %v, got %v", fortyTwo, out)
		}
	})
	t.Run("2 channels with two values", func(t *testing.T) {
		ctx := context.Background()
		lenChan := 2
		cs := make([]chan gorgonia.Value, lenChan)
		for i := range cs {
			// The size of the channels buffer controls how far behind the receivers
			// of the fanOut channels can lag the other channels.
			cs[i] = make(chan gorgonia.Value, 0)
		}
		output := make(chan ioValue, 0)
		merge(ctx, cs, output)
		cs[1] <- &fortyThree
		cs[0] <- &fortyTwo
		missFortyTwo := true
		missFortyThree := true
		for i := 0; i < lenChan; i++ {
			out := <-output
			switch {
			case out.pos == 0 && out.v.Data().(float32) == 42.0:
				missFortyTwo = false
			case out.pos == 1 && out.v.Data().(float32) == 43.0:
				missFortyThree = false
			default:
				t.Errorf("bad conbination %v/%v", out.pos, out.v.Data())
			}
		}
		if missFortyThree || missFortyTwo {
			t.Error("Missing value")
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

func Test_broadcast(t *testing.T) {
	t.Run("context cancel without value", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cs := make([]chan gorgonia.Value, 1)
		for i := range cs {
			// The size of the channels buffer controls how far behind the receivers
			// of the fanOut channels can lag the other channels.
			cs[i] = make(chan gorgonia.Value, 0)
		}
		c := make(<-chan gorgonia.Value, 0)
		go broadcast(ctx, c, cs)
		cancel()
	})
	t.Run("context cancel without value", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cs := make([]chan gorgonia.Value, 1)
		for i := range cs {
			// The size of the channels buffer controls how far behind the receivers
			// of the fanOut channels can lag the other channels.
			cs[i] = make(chan gorgonia.Value, 0)
		}
		c := make(chan gorgonia.Value, 0)
		go broadcast(ctx, c, cs)
		c <- nil
		cancel()
	})
	t.Run("broadcast ", func(t *testing.T) {
		fortyTwo := gorgonia.F32(42.0)
		ctx := context.Background()
		cs := make([]chan gorgonia.Value, 2)
		for i := range cs {
			// The size of the channels buffer controls how far behind the receivers
			// of the fanOut channels can lag the other channels.
			cs[i] = make(chan gorgonia.Value, 0)
		}
		c := make(chan gorgonia.Value, 0)
		go broadcast(ctx, c, cs)
		c <- &fortyTwo
		v0 := <-cs[0]
		v1 := <-cs[1]
		if !reflect.DeepEqual(v0, &fortyTwo) {
			t.Errorf("broadcast want %v, got %v", &fortyTwo, v0)
		}
		if !reflect.DeepEqual(v1, &fortyTwo) {
			t.Errorf("broadcast want %v, got %v", &fortyTwo, v1)
		}
	})
}

func Test_pubsub_run(t *testing.T) {
	i0 := make(chan gorgonia.Value, 0)
	m0 := make(chan gorgonia.Value, 0)
	m1 := make(chan gorgonia.Value, 0)
	o0 := make(chan ioValue, 0)
	type fields struct {
		publishers  []publisher
		subscribers []subscriber
	}
	type args struct {
		ctx context.Context
	}
	tests := []struct {
		name   string
		fields fields
		args   args
	}{
		{
			"i0 -> m0 ; i0 -> m1; m0 -> o0; m1 -> o0",
			fields{
				publishers: []publisher{
					{
						publisher: i0,
						subscribers: []chan gorgonia.Value{
							m0, m1,
						},
					},
				},
				subscribers: []subscriber{
					{
						publishers: []chan gorgonia.Value{
							m0, m1,
						},
						subscriber: o0,
					},
				},
			},
			args{
				context.TODO(),
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &pubsub{
				publishers:  tt.fields.publishers,
				subscribers: tt.fields.subscribers,
			}
			fortyTwo := gorgonia.F32(42.0)
			cancel := p.run(tt.args.ctx)
			i0 <- &fortyTwo
			v1 := <-o0
			v2 := <-o0
			if v1.v.Data().(float32) != 42 {
				t.Fail()
			}
			if v2.v.Data().(float32) != 42 {
				t.Fail()
			}
			if v2.pos+v1.pos != 1 {
				t.Fail()
			}

			cancel()
		})
	}
}
