package exprgraph_test

import (
	"context"
	"fmt"
	"math/rand"
	"strings"
	"testing"
	"time"

	"gorgonia.org/gorgonia"
	"gorgonia.org/gorgonia/exprgraph"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/dense"
)

const chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

func rndName() string {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	b := new(strings.Builder)
	for i := 0; i < 5; i++ {
		idx := r.Intn(len(chars))
		b.WriteByte(chars[idx])
	}
	return b.String()
}
func randNum(x []float64) []float64 {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	for i := range x {
		x[i] = r.Float64()
	}
	return x
}

var sixtyfour = randNum(make([]float64, 64*64))

func shortExpr[DT tensor.Num, T values.Value[DT]](g *exprgraph.Graph, x gorgonia.Tensor) (gorgonia.Tensor, error) {
	if x == nil {
		x = exprgraph.New[DT](g, rndName(), tensor.WithShape(2, 2), tensor.WithInit(randNum))
	}
	y := exprgraph.New[DT](g, rndName(), tensor.WithShape(2, 2), tensor.WithInit(randNum))
	z := exprgraph.New[DT](g, rndName(), tensor.WithShape(), tensor.WithBacking([]float64{0}))
	xy, err := MatMul[DT, T](x, y)
	if err != nil {
		return nil, err
	}
	return Add[DT, T](xy, z)
}

func longExpr[DT tensor.Num, T values.Value[DT]](g *exprgraph.Graph, n int) (gorgonia.Tensor, error) {
	expr, err := shortExpr[DT, T](g, nil)
	if err != nil {
		return nil, err
	}
	for i := 1; i < n; i++ {
		if expr, err = shortExpr[DT, T](g, expr); err != nil {
			return nil, err
		}
	}
	return expr, nil
}

func TestForwardDiff(t *testing.T) {
	engine := &FwdEngine[float64, *dense.Dense[float64]]{StandardEngine: dense.StdFloat64Engine[*dense.Dense[float64]]{}}
	g := exprgraph.NewGraph(engine)
	engine.g = g
	z, err := longExpr[float64, tensor.Basic[float64]](g, 1)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%T | %v ", z, z)
}

func TestBackwardDiffBasic(t *testing.T) {
	engine := &BwdEngine[float64, *dense.Dense[float64]]{StandardEngine: dense.StdFloat64Engine[*dense.Dense[float64]]{}}
	g := exprgraph.NewGraph(engine)
	engine.g = g
	z, err := longExpr[float64, tensor.Basic[float64]](g, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := engine.Backwards(context.Background()); err != nil {
		t.Fatal(err)
	}
	t.Logf("%T | %v", z, z)
}
func TestBackwardDiffDense(t *testing.T) {
	engine := &BwdEngine[float64, *dense.Dense[float64]]{StandardEngine: dense.StdFloat64Engine[*dense.Dense[float64]]{}}
	g := exprgraph.NewGraph(engine)
	engine.g = g
	z, err := longExpr[float64, *dense.Dense[float64]](g, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := engine.Backwards(context.Background()); err != nil {
		t.Fatal(err)
	}
	t.Logf("%T | %v", z, z)
}

func BenchmarkForwardDiff(b *testing.B) {
	exprLengths := []int{1, 5, 10, 100, 1000}
	for _, bm := range exprLengths {
		b.Run(fmt.Sprintf("Expr Length %d", bm), func(b *testing.B) {
			b.StopTimer()
			engine := &FwdEngine[float64, *dense.Dense[float64]]{StandardEngine: dense.StdFloat64Engine[*dense.Dense[float64]]{}}
			g := exprgraph.NewGraph(engine)
			engine.g = g
			b.ResetTimer()
			b.StartTimer()

			var z gorgonia.Tensor
			var err error
			for i := 0; i < b.N; i++ {
				if z, err = longExpr[float64, tensor.Basic[float64]](g, bm); err != nil {
					b.Error(err)
				}
			}
			_ = z
		})
	}
}

func BenchmarkBackwardDiff(b *testing.B) {
	exprLengths := []int{1, 5, 10, 100, 1000}
	for _, bm := range exprLengths {
		b.Run(fmt.Sprintf("dense/length=%d", bm), func(b *testing.B) {
			b.StopTimer()
			engine := &BwdEngine[float64, *dense.Dense[float64]]{StandardEngine: dense.StdFloat64Engine[*dense.Dense[float64]]{}}
			g := exprgraph.NewGraph(engine)
			engine.g = g

			b.ResetTimer()
			b.StartTimer()

			var z gorgonia.Tensor
			var err error
			for i := 0; i < b.N; i++ {
				if z, err = longExpr[float64, *dense.Dense[float64]](g, bm); err != nil {
					b.Error(err)
				}
				if err := engine.Backwards(context.Background()); err != nil {
					fmt.Printf("Backwards failed. Err: %v\n", err)
					return
				}
			}
			_ = z
		})
	}

	for _, bm := range exprLengths {
		b.Run(fmt.Sprintf("Basic/length=%d", bm), func(b *testing.B) {
			b.StopTimer()
			engine := &BwdEngine[float64, *dense.Dense[float64]]{StandardEngine: dense.StdFloat64Engine[*dense.Dense[float64]]{}}
			g := exprgraph.NewGraph(engine)
			engine.g = g

			b.ResetTimer()
			b.StartTimer()

			var z gorgonia.Tensor
			var err error
			for i := 0; i < b.N; i++ {
				if z, err = longExpr[float64, tensor.Basic[float64]](g, bm); err != nil {
					b.Error(err)
				}
				if err := engine.Backwards(context.Background()); err != nil {
					fmt.Printf("Backwards failed. Err: %v\n", err)
					return
				}
			}
			_ = z
		})
	}
}
