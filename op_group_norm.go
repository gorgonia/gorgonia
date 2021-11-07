package gorgonia

import (
	"fmt"
	"hash"
	"log"
	"math"

	"github.com/chewxy/hm"
	"gorgonia.org/tensor"
	"gorgonia.org/vecf64"
)

const (
	groupNormChunkSize = 16
	groupNormVecSize   = 8
)

func GroupNorm(x, scale, bias *Node, numGroups, numChannels int, epsilon float64) (*Node, error) {
	xShape := x.Shape()

	mean := tensor.New(
		tensor.Of(x.Dtype()),
		tensor.WithShape(xShape[0]*numGroups),
	)

	rstd := tensor.New(
		tensor.Of(x.Dtype()),
		tensor.WithShape(xShape[0]*numGroups),
	)

	op := &GroupNormOp{
		numGroups:   numGroups,
		numChannels: numChannels,
		epsilon:     epsilon,
		mean:        mean,
		rstd:        rstd,
	}

	result, err := ApplyOp(op, x)
	if err != nil {
		return nil, err
	}

	if result, err = Auto(BroadcastHadamardProd, scale, result); err != nil {
		return nil, err
	}

	result, err = Auto(BroadcastAdd, result, bias)

	return result, err
}

type GroupNormOp struct {
	numGroups, numChannels int
	epsilon                float64

	mean, rstd *tensor.Dense
}

func (op *GroupNormOp) Arity() int { return 1 }

func (op *GroupNormOp) ReturnsPtr() bool { return false }

func (op *GroupNormOp) CallsExtern() bool { return false }

func (op *GroupNormOp) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, op.String())
}

func (op *GroupNormOp) Hashcode() uint32 { return simpleHash(op) }

func (op *GroupNormOp) String() string {
	return fmt.Sprintf("GroupNorm{%d, %v}()", op.numGroups, op.numChannels)
}

func (op *GroupNormOp) InferShape(inputs ...DimSizer) (tensor.Shape, error) {
	s := inputs[0].(tensor.Shape).Clone()

	return s, nil
}

func (op *GroupNormOp) Type() hm.Type {
	a := hm.TypeVariable('a')

	return hm.NewFnType(a, a) // f(float64) float64
}

func (op *GroupNormOp) OverwritesInput() int { return -1 }

func (op *GroupNormOp) Do(inputs ...Value) (Value, error) {
	input := inputs[0]
	prealloc := tensor.New(tensor.WithShape(input.Shape().Clone()...), tensor.Of(input.Dtype()))

	return op.UsePreallocDo(prealloc, inputs...)
}

func (op *GroupNormOp) UsePreallocDo(prealloc Value, inputs ...Value) (Value, error) {
	if err := checkArity(op, len(inputs)); err != nil {
		return nil, err
	}

	xT := inputs[0].(*tensor.Dense)
	xShape := xT.Shape()

	log.Printf("x: %v", xT)

	n := xShape[0]
	channels := xShape[1]
	imageSize := 1
	if len(xShape) > 2 {
		imageSize = tensor.Shape(xShape[2:]).TotalSize()
	}

	d := channels / op.numGroups
	innerSize := d * imageSize

	log.Printf("inner size: %v", innerSize)

	x := xT.Float64s()
	y := prealloc.(*tensor.Dense).Float64s()

	meanA := op.mean.Float64s()
	rstdA := op.rstd.Float64s()

	for i := 0; i < op.numGroups*n; i++ {
		baseIndex := i * innerSize
		xSection := x[baseIndex : baseIndex+innerSize]

		mean, rstd := op.rowwiseMomentsF64(xSection, innerSize, 0)
		rstd = 1 / math.Sqrt(math.Max(rstd, 0)+op.epsilon)

		log.Printf("mean: %v rstd: %v", mean, rstd)

		// g := i % op.numGroups
		for j := 0; j < d; j++ {
			// c := g*d + j
			scale := rstd
			bias := -scale * mean

			baseIndex := (i*d + j) * imageSize
			xSection := x[baseIndex : baseIndex+imageSize]
			ySection := y[baseIndex : baseIndex+imageSize]

			for k := 0; k < imageSize; k++ {
				ySection[k] =
					scale*xSection[k] + bias
			}
		}

		meanA[i] = mean
		rstdA[i] = rstd
	}

	_ = x

	return prealloc, nil
}

func (op *GroupNormOp) rowwiseMomentsF64(x []float64, n int, ddof int) (mean float64, variance float64) {
	nn := n / groupNormVecSize
	m := (nn + groupNormChunkSize - 1) / groupNormChunkSize
	depth := op.ceilLog2(m)

	log.Printf("m: %v depth: %v", m, depth)

	m0stk := make([]int, depth)
	m1stk := make([][]float64, depth)
	m2stk := make([][]float64, depth)

	for i := 0; i < depth; i++ {
		m1stk[i] = make([]float64, groupNormVecSize)
		m2stk[i] = make([]float64, groupNormVecSize)
	}

	for i := 0; i < m; i++ {
		baseIndex := i * groupNormChunkSize * groupNormVecSize
		xSection1 := x[baseIndex : baseIndex+groupNormVecSize]

		m0 := int(math.Min(groupNormChunkSize, float64(nn-i*groupNormChunkSize)))
		log.Printf("xint: %v", xSection1)

		// TODO: optimize allocs
		m1vec := make([]float64, groupNormVecSize)
		m2vec := make([]float64, groupNormVecSize)
		delta := make([]float64, groupNormVecSize)
		tmp := make([]float64, groupNormVecSize)

		for j := 0; j < m0; j++ {
			baseIndex := j * groupNormVecSize
			xSection2 := x[baseIndex : baseIndex+groupNormVecSize]

			copy(tmp, xSection2)
			vecf64.Sub(tmp, m1vec)

			c := 1.0 / float64(j+1)

			// update m1vec
			copy(delta, tmp)
			vecf64.Scale(tmp, c)
			vecf64.Add(m1vec, tmp)

			// update m2vec
			copy(tmp, xSection2)
			vecf64.Sub(tmp, m1vec)
			vecf64.Mul(tmp, delta)
			vecf64.Add(m2vec, tmp)
		}

		log.Printf("m1_vec: %v", m1vec)
		log.Printf("m2_vec: %v", m2vec)

		op.addMomentsVecF64(m0, m1vec, m2vec, &m0stk[0], m1stk[0], m2stk[0])

		log.Printf("AFTER m0stk: %v", m0stk)
		log.Printf("AFTER m1stk: %v", m1stk)
		log.Printf("AFTER m2stk: %v", m2stk)

		mask := i + 1
		for j := 1; j < depth && mask&1 == 0; j++ {
			op.addMomentsVecF64(m0stk[j-1], m1stk[j-1], m2stk[j-1], &m0stk[j], m1stk[j], m2stk[j])
			m0stk[j-1] = 0
			m1stk[j-1] = make([]float64, groupNormVecSize) // is this optimized by the compiler?
			m2stk[j-1] = make([]float64, groupNormVecSize)
			mask >>= 1
		}
	}

	for i := 1; i < depth; i++ {
		op.addMomentsVecF64(m0stk[i], m1stk[i], m2stk[i], &m0stk[0], m1stk[0], m1stk[0])
	}

	var (
		m0     int
		m1, m2 float64
	)

	for i := nn * groupNormVecSize; i < n; i++ {
		delta := x[i] - m1
		m0++
		m1 += delta / float64(m0)
		m2 += delta * (x[i] - m1)
	}

	for i := 0; i < groupNormVecSize; i++ {
		op.addMomentsF64(nn, m1stk[0][i], m2stk[0][i], &m0, &m1, &m2)
	}

	return m1, m2 / float64(n-ddof)
}

func (op *GroupNormOp) addMomentsF64(m0add int, m1add, m2add float64, m0 *int, m1, m2 *float64) {
	n := *m0 + m0add
	c := 0.0
	if n != 0 {
		c = float64(m0add) / float64(n)
	}

	delta := m1add - *m1

	*m1 += c * delta
	*m2 += m2add + delta*delta*c*float64(*m0)
	*m0 = n
}

func (op *GroupNormOp) addMomentsVecF64(m0add int, m1add, m2add []float64, m0 *int, m1, m2 []float64) {
	n := *m0 + m0add
	c := 0.0
	if n != 0 {
		c = float64(m0add) / float64(n)
	}

	delta := make([]float64, len(m1add))
	copy(delta, m1add)
	vecf64.Sub(delta, m1)

	// update m1
	tmp := make([]float64, len(delta))
	copy(tmp, delta)
	vecf64.Scale(tmp, c) // delta * c
	vecf64.Add(m1, tmp)

	// update m2
	copy(tmp, delta)
	vecf64.Mul(tmp, delta) // delta * delta
	vecf64.Scale(tmp, c)
	vecf64.Scale(tmp, float64(*m0))
	vecf64.Add(tmp, m2add)
	vecf64.Add(m2, tmp)

	// update m0
	*m0 = n
}

func (op *GroupNormOp) ceilLog2(x int) int {
	if x <= 2 {
		return 1
	}

	return int(math.Ceil(math.Log2(float64(x))))
}

// DoDiff calculates the diff and sets its value to the output node. Implementation for ADOp interface.
func (op *GroupNormOp) DoDiff(ctx ExecutionContext, inputs Nodes, output *Node) error {
	if len(inputs) != 1 {
		return fmt.Errorf("GroupNorm.DoDiff needs 1 arguments")
	}

	// odv := output.boundTo.(*dualValue)
	// idv := inputs[0].boundTo.(*dualValue)
	// idvd := idv.d.(*tensor.Dense)
	// diffOp := &diffOp{op}

	// result, err := diffOp.Do(idv.Value, odv.Value, odv.d)
	// if err != nil {
	// 	return err
	// }

	// sum, err := idvd.Add(result.(*tensor.Dense), tensor.UseUnsafe())
	// if err != nil {
	// 	return err
	// }

	// odv.d = sum

	return nil
}

// SymDiff applies the diff op. Implementation for SDOp interface.
func (op *GroupNormOp) SymDiff(inputs Nodes, output, grad *Node) (Nodes, error) {
	err := checkArity(op, len(inputs))
	if err != nil {
		return nil, err
	}

	// diffOp := &GroupNormOp{op}
	nodes := Nodes{
		inputs[0],
	}

	// nodes[0], err = ApplyOp(diffOp, inputs[0], output, grad)

	return nodes, err
}

// DiffWRT is an implementation for the SDOp interface
func (op *GroupNormOp) DiffWRT(inputs int) []bool {
	return []bool{true}
}

// ensure it complies with the Op interface
var (
	_ Op   = &GroupNormOp{}
	_ ADOp = &GroupNormOp{}
	_ SDOp = &GroupNormOp{}
)
