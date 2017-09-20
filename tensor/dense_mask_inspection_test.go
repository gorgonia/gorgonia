package tensor

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestMaskedInspection(t *testing.T) {
	assert := assert.New(t)

	var retT *Dense

	//vector case
	T := New(Of(Bool), WithShape(1, 12))
	T.ResetMask(false)
	assert.False(T.MaskedAny().(bool))
	for i := 0; i < 12; i += 2 {
		T.mask[i] = true
	}
	assert.True(T.MaskedAny().(bool))
	assert.True(T.MaskedAny(0).(bool))
	assert.False(T.MaskedAll().(bool))
	assert.False(T.MaskedAll(0).(bool))
	assert.Equal(6, T.MaskedCount())
	assert.Equal(6, T.MaskedCount(0))
	assert.Equal(6, T.NonMaskedCount())
	assert.Equal(6, T.NonMaskedCount(0))

	//contiguous mask case
	/*equivalent python code
	  ---------
	  import numpy.ma as ma
	  a = ma.arange(12).reshape((2, 3, 2))
	  a[0,0,0]=ma.masked
	  a[0,2,0]=ma.masked
	  print(ma.getmask(a).all())
	  print(ma.getmask(a).any())
	  print(ma.count_masked(a))
	  print(ma.count(a))
	  print(ma.getmask(a).all(0))
	  print(ma.getmask(a).any(0))
	  print(ma.count_masked(a,0))
	  print(ma.count(a,0))
	  print(ma.getmask(a).all(1))
	  print(ma.getmask(a).any(1))
	  print(ma.count_masked(a,1))
	  print(ma.count(a,1))
	  print(ma.getmask(a).all(2))
	  print(ma.getmask(a).any(2))
	  print(ma.count_masked(a,2))
	  print(ma.count(a,2))
	  -----------
	*/
	T = New(Of(Bool), WithShape(2, 3, 2))
	T.ResetMask(false)

	for i := 0; i < 2; i += 2 {
		for j := 0; j < 3; j += 2 {
			for k := 0; k < 2; k += 2 {
				a, b, c := T.strides[0], T.strides[1], T.strides[2]
				T.mask[i*a+b*j+c*k] = true
			}
		}
	}

	assert.Equal([]bool{true, false, false, false, true, false,
		false, false, false, false, false, false}, T.mask)

	assert.Equal(false, T.MaskedAll())
	assert.Equal(true, T.MaskedAny())
	assert.Equal(2, T.MaskedCount())
	assert.Equal(10, T.NonMaskedCount())

	retT = T.MaskedAll(0).(*Dense)
	assert.Equal([]int{3, 2}, []int(retT.shape))
	assert.Equal([]bool{false, false, false, false, false, false}, retT.Bools())
	retT = T.MaskedAny(0).(*Dense)
	assert.Equal([]int{3, 2}, []int(retT.shape))
	assert.Equal([]bool{true, false, false, false, true, false}, retT.Bools())
	retT = T.MaskedCount(0).(*Dense)
	assert.Equal([]int{3, 2}, []int(retT.shape))
	assert.Equal([]int{1, 0, 0, 0, 1, 0}, retT.Ints())
	retT = T.NonMaskedCount(0).(*Dense)
	assert.Equal([]int{1, 2, 2, 2, 1, 2}, retT.Ints())

	retT = T.MaskedAll(1).(*Dense)
	assert.Equal([]int{2, 2}, []int(retT.shape))
	assert.Equal([]bool{false, false, false, false}, retT.Bools())
	retT = T.MaskedAny(1).(*Dense)
	assert.Equal([]int{2, 2}, []int(retT.shape))
	assert.Equal([]bool{true, false, false, false}, retT.Bools())
	retT = T.MaskedCount(1).(*Dense)
	assert.Equal([]int{2, 2}, []int(retT.shape))
	assert.Equal([]int{2, 0, 0, 0}, retT.Ints())
	retT = T.NonMaskedCount(1).(*Dense)
	assert.Equal([]int{1, 3, 3, 3}, retT.Ints())

	retT = T.MaskedAll(2).(*Dense)
	assert.Equal([]int{2, 3}, []int(retT.shape))
	assert.Equal([]bool{false, false, false, false, false, false}, retT.Bools())
	retT = T.MaskedAny(2).(*Dense)
	assert.Equal([]int{2, 3}, []int(retT.shape))
	assert.Equal([]bool{true, false, true, false, false, false}, retT.Bools())
	retT = T.MaskedCount(2).(*Dense)
	assert.Equal([]int{2, 3}, []int(retT.shape))
	assert.Equal([]int{1, 0, 1, 0, 0, 0}, retT.Ints())
	retT = T.NonMaskedCount(2).(*Dense)
	assert.Equal([]int{1, 2, 1, 2, 2, 2}, retT.Ints())

}

func TestMaskedFindContiguous(t *testing.T) {
	assert := assert.New(t)
	T := NewDense(Int, []int{1, 100})
	T.ResetMask(false)
	retSL := T.FlatNotMaskedContiguous()
	assert.Equal(1, len(retSL))
	assert.Equal(rs{0, 100, 1}, retSL[0].(rs))

	// test ability to find unmasked regions
	sliceList := make([]Slice, 0, 4)
	sliceList = append(sliceList, makeRS(3, 9), makeRS(14, 27), makeRS(51, 72), makeRS(93, 100))
	T.ResetMask(true)
	for i := range sliceList {
		tt, _ := T.Slice(nil, sliceList[i])
		ts := tt.(*Dense)
		ts.ResetMask(false)
	}
	retSL = T.FlatNotMaskedContiguous()
	assert.Equal(sliceList, retSL)

	retSL = T.ClumpUnmasked()
	assert.Equal(sliceList, retSL)

	// test ability to find masked regions
	T.ResetMask(false)
	for i := range sliceList {
		tt, _ := T.Slice(nil, sliceList[i])
		ts := tt.(*Dense)
		ts.ResetMask(true)
	}
	retSL = T.FlatMaskedContiguous()
	assert.Equal(sliceList, retSL)

	retSL = T.ClumpMasked()
	assert.Equal(sliceList, retSL)
}

func TestMaskedFindEdges(t *testing.T) {
	assert := assert.New(t)
	T := NewDense(Int, []int{1, 100})

	sliceList := make([]Slice, 0, 4)
	sliceList = append(sliceList, makeRS(0, 9), makeRS(14, 27), makeRS(51, 72), makeRS(93, 100))

	// test ability to find unmasked edges
	T.ResetMask(false)
	for i := range sliceList {
		tt, _ := T.Slice(nil, sliceList[i])
		ts := tt.(*Dense)
		ts.ResetMask(true)
	}
	start, end := T.FlatNotMaskedEdges()
	assert.Equal(9, start)
	assert.Equal(92, end)

	// test ability to find masked edges
	T.ResetMask(true)
	for i := range sliceList {
		tt, _ := T.Slice(nil, sliceList[i])
		ts := tt.(*Dense)
		ts.ResetMask(false)
	}
	start, end = T.FlatMaskedEdges()
	assert.Equal(9, start)
	assert.Equal(92, end)
}
