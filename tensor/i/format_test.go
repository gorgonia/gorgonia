package tensori

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestTFormat(t *testing.T) {
	assert := assert.New(t)
	var T *Tensor
	var res, expected string

	// Scalar
	T = NewTensor(AsScalar(3))
	res = fmt.Sprintf("%3.3d", T)
	assert.Equal("003", res)

	// short vector
	T = NewTensor(WithShape(4))
	res = fmt.Sprintf("%v", T)
	assert.Equal("[  0    0    0    0]", res)

	T = NewTensor(WithShape(3, 3), WithBacking(RangeInt(0, 9)))

	res = fmt.Sprintf("\n%v", T)
	expected = `
⎡  0    1    2⎤
⎢  3    4    5⎥
⎣  6    7    8⎦
`
	assert.Equal(expected, res, res)

	// different bases
	res = fmt.Sprintf("\n%x", T)
	expected = `
⎡  0    1    2⎤
⎢  3    4    5⎥
⎣  6    7    8⎦
`
	assert.Equal(expected, res, res)

	// with metadata
	res = fmt.Sprintf("\n%+0.2v", T)
	expected = `
Matrix (3, 3) [3 1]
⎡0  1  2⎤
⎢3  4  5⎥
⎣6  7  8⎦
`
	assert.Equal(expected, res, res)

	// many columns
	T = NewTensor(WithShape(16, 14), WithBacking(RangeInt(0, 16*14)))
	res = fmt.Sprintf("\n%v", T)
	expected = `
⎡  0    1    2    3  ...  10   11   12   13⎤
⎢ 14   15   16   17  ...  24   25   26   27⎥
⎢ 28   29   30   31  ...  38   39   40   41⎥
⎢ 42   43   44   45  ...  52   53   54   55⎥
.
.
.
⎢168  169  170  171  ... 178  179  180  181⎥
⎢182  183  184  185  ... 192  193  194  195⎥
⎢196  197  198  199  ... 206  207  208  209⎥
⎣210  211  212  213  ... 220  221  222  223⎦
`
	assert.Equal(expected, res, res)

	// many cols, rows, compressed
	T = NewTensor(WithShape(16, 14), WithBacking(RangeInt(0, 16*14)))
	res = fmt.Sprintf("\n%s", T)
	expected = `
⎡  0    1  ⋯  12   13⎤
⎢ 14   15  ⋯  26   27⎥
  ⋮  
⎢196  197  ⋯ 208  209⎥
⎣210  211  ⋯ 222  223⎦
`
	assert.Equal(expected, res, res)

	// many cols, full
	T = NewTensor(WithShape(8, 9), WithBacking(RangeInt(0, 8*9)))
	res = fmt.Sprintf("\n%#v", T)
	expected = `
⎡ 0   1   2   3   4   5   6   7   8⎤
⎢ 9  10  11  12  13  14  15  16  17⎥
⎢18  19  20  21  22  23  24  25  26⎥
⎢27  28  29  30  31  32  33  34  35⎥
⎢36  37  38  39  40  41  42  43  44⎥
⎢45  46  47  48  49  50  51  52  53⎥
⎢54  55  56  57  58  59  60  61  62⎥
⎣63  64  65  66  67  68  69  70  71⎦
`
	assert.Equal(expected, res, res)

	// vectors
	T = NewTensor(WithShape(3, 1))
	res = fmt.Sprintf("%v", T)
	expected = `C[0  0  0]`
	assert.Equal(expected, res)

	T = NewTensor(WithShape(1, 3))
	res = fmt.Sprintf("%v", T)
	expected = `R[0  0  0]`
	assert.Equal(expected, res)

	// 3+ Dimensional Tensors - super janky for now
	T = NewTensor(WithShape(2, 3, 2), WithBacking(RangeInt(0, 2*3*2)))
	res = fmt.Sprintf("\n%v", T)
	expected = `
⎡ 0   1⎤
⎢ 2   3⎥
⎣ 4   5⎦

⎡ 6   7⎤
⎢ 8   9⎥
⎣10  11⎦

`
	assert.Equal(expected, res, res)

	// checking metadata + compression
	res = fmt.Sprintf("\n%+s", T)
	expected = `
3-Tensor (2, 3, 2) [6 2 1]
⎡ 0   1⎤
⎢ 2   3⎥
⎣ 4   5⎦

⎡ 6   7⎤
⎢ 8   9⎥
⎣10  11⎦

`
	assert.Equal(expected, res, res)

	// check flat + compress
	res = fmt.Sprintf("%-s", T)
	expected = `[0 1 2 3 4 ⋯ ]`
	assert.Equal(expected, res, res)

	// check flat
	res = fmt.Sprintf("%-3.3f", T)
	expected = `[0 1 2 3 4 5 6 7 8 9 ... ]`
	assert.Equal(expected, res, res)

	// check flat + extended
	res = fmt.Sprintf("%-#v", T)
	expected = `[0 1 2 3 4 5 6 7 8 9 10 11]`
	assert.Equal(expected, res, res)

	/* Test Views and Sliced Tensors */

	var V *Tensor
	var err error

	V, err = T.Slice(rangedSlice{1, 2})
	if err != nil {
		t.Error(err)
	}
	res = fmt.Sprintf("\n%+s", V)
	expected = `
Matrix (3, 2) [2 1]
⎡  6    7⎤
⎢  8    9⎥
⎣ 10   11⎦
`
	assert.Equal(expected, res, res)

	// T[:, 1]
	V, err = T.Slice(nil, singleSlice(1))
	res = fmt.Sprintf("\n%+s", V)
	expected = `
Matrix (2, 2) [6 1]
⎡2  3⎤
⎣8  9⎦
`
	assert.Equal(expected, res, res)

	// T[1, :, 1]
	V, err = T.Slice(singleSlice(1), nil, singleSlice(1))
	if err != nil {
		t.Error(err)
	}
	expected = `Vector (3, 1) [2]
C[ 111  1001  1011]`
	res = fmt.Sprintf("%+b", V)
	assert.Equal(expected, res)

	// T[1, 1, 1] - will result in a scalar
	V, err = T.Slice(singleSlice(1), singleSlice(1), singleSlice(1))
	if err != nil {
		t.Error(err)
	}

	// on regular matrices
	T = NewTensor(WithShape(3, 5), WithBacking(RangeInt(0, 3*5)))
	V, err = T.Slice(singleSlice(1))
	if err != nil {
		t.Error(err)
	}
	expected = `R[  5    6    7    8    9]`
	res = fmt.Sprintf("%v", V)
	assert.Equal(expected, res)

}
