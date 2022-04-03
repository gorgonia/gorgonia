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

	// cache
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

	batchSize := xShape[0]
	channels := xShape[1]
	imageSize := 1
	if len(xShape) > 2 {
		imageSize = tensor.Shape(xShape[2:]).TotalSize()
	}

	d := channels / op.numGroups
	innerSize := d * imageSize

	x := xT.Float64s()
	y := prealloc.(*tensor.Dense).Float64s()

	meanA := op.mean.Float64s()
	rstdA := op.rstd.Float64s()

	for i := 0; i < op.numGroups*batchSize; i++ { // TODO: parallelize
		baseIndex := i * innerSize
		xSection := x[baseIndex : baseIndex+innerSize]

		mean, rstd := op.rowwiseMomentsF64(xSection, innerSize, 0)
		rstd = 1 / math.Sqrt(math.Max(rstd, 0)+op.epsilon)

		for j := 0; j < d; j++ {
			scale := rstd
			bias := -scale * mean

			baseIndex := (i*d + j) * imageSize
			xSection := x[baseIndex : baseIndex+imageSize]
			ySection := y[baseIndex : baseIndex+imageSize]

			for k := 0; k < imageSize; k++ {
				ySection[k] = scale*xSection[k] + bias
			}
		}

		meanA[i] = mean
		rstdA[i] = rstd
	}

	log.Printf("mean: %v", op.mean)
	log.Printf("rstd: %v", op.rstd)

	log.Printf("output: %v", prealloc)

	return prealloc, nil
}

func (op *GroupNormOp) rowwiseMomentsF64(x []float64, n int, ddof int) (mean float64, variance float64) {
	nn := n / groupNormVecSize
	m := (nn + groupNormChunkSize - 1) / groupNormChunkSize
	depth := op.ceilLog2F64(m)

	m0stk := make([]int, depth)
	m1stk := make([][]float64, depth)
	m2stk := make([][]float64, depth)

	for i := 0; i < depth; i++ {
		m1stk[i] = make([]float64, groupNormVecSize)
		m2stk[i] = make([]float64, groupNormVecSize)
	}

	for i := 0; i < m; i++ {
		m0 := int(math.Min(groupNormChunkSize, float64(nn-i*groupNormChunkSize)))

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

		op.addMomentsVecF64(m0, m1vec, m2vec, &m0stk[0], m1stk[0], m2stk[0])

		mask := i + 1
		for j := 1; j < depth && (mask&1 == 0); j++ {
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

func (op *GroupNormOp) ceilLog2F64(x int) int {
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

	odv := output.boundTo.(*dualValue)
	idv := inputs[0].boundTo.(*dualValue)
	idvd := idv.d.(*tensor.Dense)
	diffOp := &groupNormDiffOp{op}

	result, err := diffOp.Do(idv.Value, odv.Value, odv.d)
	if err != nil {
		return err
	}

	sum, err := idvd.Add(result.(*tensor.Dense), tensor.UseUnsafe())
	if err != nil {
		return err
	}

	odv.d = sum

	return nil
}

// SymDiff applies the diff op. Implementation for SDOp interface.
func (op *GroupNormOp) SymDiff(inputs Nodes, output, grad *Node) (Nodes, error) {
	err := checkArity(op, len(inputs))
	if err != nil {
		return nil, err
	}

	diffOp := &groupNormDiffOp{op}

	dy, err := ApplyOp(diffOp, inputs[0], grad)
	if err != nil {
		return nil, err
	}

	return Nodes{dy, nil, nil}, err
}

// DiffWRT is an implementation for the SDOp interface
func (op *GroupNormOp) DiffWRT(inputs int) []bool {
	return []bool{true}
}

type groupNormDiffOp struct {
	*GroupNormOp
}

func (op *groupNormDiffOp) Arity() int { return 2 }

func (op *groupNormDiffOp) ReturnsPtr() bool { return false }

func (op *groupNormDiffOp) CallsExtern() bool { return false }

func (op *groupNormDiffOp) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, op.String())
}

func (op *groupNormDiffOp) Hashcode() uint32 { return simpleHash(op) }

func (op *groupNormDiffOp) String() string {
	return fmt.Sprintf("groupNormDiff{%d, %v}()", op.numGroups, op.numChannels)
}

func (op *groupNormDiffOp) InferShape(inputs ...DimSizer) (tensor.Shape, error) {
	s := inputs[0].(tensor.Shape).Clone()

	return s, nil
}

func (op *groupNormDiffOp) Type() hm.Type {
	a := hm.TypeVariable('a')

	return hm.NewFnType(a, a, a) // f(float64) float64
}

func (op *groupNormDiffOp) OverwritesInput() int { return -1 }

func (op *groupNormDiffOp) Do(inputs ...Value) (Value, error) {
	input := inputs[0]
	grad := inputs[1]

	dy, err := CloneValue(input)
	if err != nil {
		return nil, err
	}

	return op.UsePreallocDo(dy, input, grad)
}

func (op *groupNormDiffOp) UsePreallocDo(prealloc Value, inputs ...Value) (Value, error) {
	var err error

	if err = checkArity(op, len(inputs)); err != nil {
		return nil, err
	}

	input := inputs[0].(*tensor.Dense)
	buffer := prealloc.(*tensor.Dense)
	outGrad := inputs[1].(*tensor.Dense)

	switch input.Dtype() {
	case Float64:
		err = op.f64s(input, buffer, outGrad)
	case Float32:
		panic("not supported yet")
		// err = op.f32s(input, buffer, outGrad)
	default:
		return nil, nyi("batchnormDiffOp", "Do")
	}

	return prealloc, err
}

func (op *groupNormDiffOp) f64s(input, prealloc, outGrad *tensor.Dense) (err error) {
	in := input.Float64s()
	dx := prealloc.Float64s()

	mean := op.mean.Float64s()
	rstd := op.rstd.Float64s()

	dy := outGrad.Float64s()

	log.Printf("dy: %v", outGrad)

	xShape := input.Shape()

	batchSize := xShape[0]
	channels := xShape[1]
	imageSize := 1
	if len(xShape) > 2 {
		imageSize = tensor.Shape(xShape[2:]).TotalSize()
	}

	ds, db := op.computeInternalGradientsF64(batchSize, channels, imageSize, input, outGrad)
	d := channels / op.numGroups
	s := 1.0 / float64(d*imageSize)

	for i := 0; i < batchSize*op.numGroups; i++ {
		baseIndex := i * d

		dsSection := ds[baseIndex : baseIndex+d]
		dbSection := db[baseIndex : baseIndex+d]

		ds := 0.0
		db := 0.0

		for j := 0; j < d; j++ {
			ds += dsSection[j]
			db += dbSection[j]
		}

		c1 := rstd[i]
		c2 := (db*mean[i] - ds) * c1 * c1 * c1 * s
		c3 := -c2*mean[i] - db*c1*s

		for j := 0; j < d; j++ {
			baseIndex := (i*d + j) * imageSize
			xSection := in[baseIndex : baseIndex+imageSize]
			dySection := dy[baseIndex : baseIndex+imageSize]
			dxSection := dx[baseIndex : baseIndex+imageSize]

			for k := 0; k < imageSize; k++ {
				dxSection[k] = c1*dySection[k] + c2*xSection[k] + c3
			}
		}
	}

	return nil
}

func (op *groupNormDiffOp) computeInternalGradientsF64(batchSize, channels, imageSize int, input, dyT *tensor.Dense) ([]float64, []float64) {
	in := input.Float64s()
	dy := dyT.Float64s()

	// FIXME: test large image

	dsA := make([]float64, batchSize*channels)
	dbA := make([]float64, batchSize*channels)

	for i := 0; i < batchSize*channels; i++ { // TODO: paralellize
		baseIndex := i * imageSize

		dySection := dy[baseIndex : baseIndex+imageSize]
		inSection := in[baseIndex : baseIndex+imageSize]

		for j := 0; j < imageSize; j++ {
			dsA[i] += dySection[j] * inSection[j]
			dbA[i] += dySection[j]
		}
	}

	return dsA, dbA
}

// DoDiff calculates the diff and sets its value to the output node. Implementation for ADOp interface.
func (op *groupNormDiffOp) DoDiff(ctx ExecutionContext, inputs Nodes, output *Node) error {
	return nyi("DoDiff", "groupNormDiffOp")
}

// SymDiff applies the diff op. Implementation for SDOp interface.
func (op *groupNormDiffOp) SymDiff(inputs Nodes, output, grad *Node) (Nodes, error) {
	return nil, nyi("SymDiff", "groupNormDiffOp")
}

// DiffWRT is an implementation for the SDOp interface
func (op *groupNormDiffOp) DiffWRT(inputs int) []bool {
	return []bool{false, false}
}

// ensure it complies with the Op interface
var (
	_ Op   = &GroupNormOp{}
	_ ADOp = &GroupNormOp{}
	_ SDOp = &GroupNormOp{}

	_ Op = &groupNormDiffOp{}
)
