package tensor

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestDense_Format(t *testing.T) {
	// if os.Getenv("TRAVISTEST") == "true" {
	// 	t.Skip("skipping format test; This is being run on TravisCI")
	// }

	assert := assert.New(t)
	var T *Dense
	var res, expected string

	// Scalar
	T = New(Of(Float64), FromScalar(3.14))
	res = fmt.Sprintf("%3.3f", T)
	assert.Equal("3.140", res)

	// short vector
	T = New(Of(Float64), WithShape(4))
	res = fmt.Sprintf("%v", T)
	expected = "[0  0  0  0]"
	assert.Equal(expected, res)
	T = New(WithShape(2, 2), WithBacking([]float64{3.141515163242, 20, 5.15, 6.28}))

	res = fmt.Sprintf("\n%v", T)
	expected = `
⎡3.141515163242              20⎤
⎣          5.15            6.28⎦
`
	assert.Equal(expected, res, res)

	// precision
	res = fmt.Sprintf("\n%0.2v", T)
	expected = `
⎡3.1   20⎤
⎣5.2  6.3⎦
`
	assert.Equal(expected, res, res)

	// with metadata
	res = fmt.Sprintf("\n%+0.2v", T)
	expected = `
Matrix (2, 2) [2 1]
⎡3.1   20⎤
⎣5.2  6.3⎦
`
	assert.Equal(expected, res, res)

	// many columns
	T = New(WithShape(16, 14), WithBacking(Range(Float32, 0, 16*14)))
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
	assert.Equal(expected, res, "expected %v. Got %v", expected, res)

	// many cols, rows, compressed
	T = New(WithShape(16, 14), WithBacking(Range(Float64, 0, 16*14)))
	res = fmt.Sprintf("\n%s", T)
	expected = `
⎡  0    1  ⋯  12   13⎤
⎢ 14   15  ⋯  26   27⎥
  ⋮  
⎢196  197  ⋯ 208  209⎥
⎣210  211  ⋯ 222  223⎦
`
	assert.Equal(expected, res, "expected %v. Got %v", expected, res)

	// many cols, full
	T = New(WithShape(8, 9), WithBacking(Range(Float64, 0, 8*9)))
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
	T = New(Of(Int), WithShape(3, 1))
	res = fmt.Sprintf("%v", T)
	expected = `C[0  0  0]`
	assert.Equal(expected, res)

	T = New(Of(Int32), WithShape(1, 3))
	res = fmt.Sprintf("%v", T)
	expected = `R[0  0  0]`
	assert.Equal(expected, res)

	// 3+ Dimensional Tensors - super janky for now
	T = New(WithShape(2, 3, 2), WithBacking(Range(Float64, 0, 2*3*2)))
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
Tensor-3 (2, 3, 2) [6 2 1]
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
	expected = `[0.000 1.000 2.000 3.000 4.000 5.000 6.000 7.000 8.000 9.000 ... ]`
	assert.Equal(expected, res, res)

	// check flat + extended
	res = fmt.Sprintf("%-#v", T)
	expected = `[0 1 2 3 4 5 6 7 8 9 10 11]`
	assert.Equal(expected, res, res)

	/* Test Views and Sliced Tensors */

	var V Tensor
	var err error

	V, err = T.Slice(makeRS(1, 2))
	if err != nil {
		t.Error(err)
	}

	// flat mode for view
	res = fmt.Sprintf("\n%-s", V)
	expected = "\n[6 7 8 9 10 ⋯ ]"
	assert.Equal(expected, res, res)

	// standard
	res = fmt.Sprintf("\n%+s", V)
	expected = `
Matrix (3, 2) [2 1]
⎡ 6   7⎤
⎢ 8   9⎥
⎣10  11⎦
`
	assert.Equal(expected, res, res)

	// T[:, 1]
	V, err = T.Slice(nil, ss(1))
	res = fmt.Sprintf("\n%+s", V)
	expected = `
Matrix (2, 2) [6 1]
⎡2  3⎤
⎣8  9⎦
`
	assert.Equal(expected, res, res)

	// transpose a view
	V.T()
	expected = `
Matrix (2, 2) [1 6]
⎡2  8⎤
⎣3  9⎦
`

	res = fmt.Sprintf("\n%+s", V)
	assert.Equal(expected, res, res)

	// T[1, :, 1]
	V, err = T.Slice(ss(1), nil, ss(1))
	if err != nil {
		t.Error(err)
	}
	expected = `Vector (3) [2]
[7881299347898368p-50  5066549580791808p-49  6192449487634432p-49]`
	res = fmt.Sprintf("%+b", V)
	assert.Equal(expected, res)

	// T[1, 1, 1] - will result in a scalar
	V, err = T.Slice(ss(1), ss(1), ss(1))
	if err != nil {
		t.Error(err)
	}
	res = fmt.Sprintf("%#3.3E", V)
	expected = `9.000E+00`
	assert.Equal(expected, res)

	// on regular matrices
	T = New(WithShape(3, 5), WithBacking(Range(Float64, 0, 3*5)))
	V, err = T.Slice(ss(1))
	if err != nil {
		t.Error(err)
	}
	expected = `[5  6  7  8  9]`
	res = fmt.Sprintf("%v", V)
	assert.Equal(expected, res)
}

var basicFmtTests = []struct {
	a      interface{}
	format string

	correct string
}{
	{Range(Float64, 0, 4), "%1.1f", "[0.0  1.0  2.0  3.0]"},
	{Range(Float32, 0, 4), "%1.1f", "[0.0  1.0  2.0  3.0]"},
	{Range(Int, 0, 4), "%b", "[ 0   1  10  11]"},
	{Range(Int, 0, 4), "%d", "[0  1  2  3]"},
	{Range(Int, 6, 10), "%o", "[ 6   7  10  11]"},
	{Range(Int, 14, 18), "%x", "[ e   f  10  11]"},
	{Range(Int, 0, 4), "%f", "[0  1  2  3]"},

	{Range(Int32, 0, 4), "%b", "[ 0   1  10  11]"},
	{Range(Int32, 0, 4), "%d", "[0  1  2  3]"},
	{Range(Int32, 6, 10), "%o", "[ 6   7  10  11]"},
	{Range(Int32, 14, 18), "%x", "[ e   f  10  11]"},
	{Range(Int32, 0, 4), "%f", "[0  1  2  3]"},

	{Range(Int64, 0, 4), "%b", "[ 0   1  10  11]"},
	{Range(Int64, 0, 4), "%d", "[0  1  2  3]"},
	{Range(Int64, 6, 10), "%o", "[ 6   7  10  11]"},
	{Range(Int64, 14, 18), "%x", "[ e   f  10  11]"},
	{Range(Int64, 0, 4), "%f", "[0  1  2  3]"},

	{Range(Byte, 0, 4), "%b", "[ 0   1  10  11]"},
	{Range(Byte, 0, 4), "%d", "[0  1  2  3]"},
	{Range(Byte, 6, 10), "%o", "[ 6   7  10  11]"},
	{Range(Byte, 14, 18), "%x", "[ e   f  10  11]"},
	{Range(Byte, 0, 4), "%f", "[0  1  2  3]"},

	{[]bool{true, false, true, false}, "%f", "[ true  false   true  false]"},
	{[]bool{true, false, true, false}, "%s", "[ true  false   true  false]"},
}

func TestDense_Format_basics(t *testing.T) {
	for _, v := range basicFmtTests {
		T := New(WithBacking(v.a))
		s := fmt.Sprintf(v.format, T)

		if s != v.correct {
			t.Errorf("Expected %q. Got %q", v.correct, s)
		}
	}
}

func TestDense_Format_Masked(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Int), WithShape(1, 12))
	data := T.ints()
	for i := 0; i < len(data); i++ {
		data[i] = i
	}
	T.ResetMask(false)
	for i := 0; i < 12; i += 2 {
		T.mask[i] = true
	}

	s := fmt.Sprintf("%d", T)
	assert.Equal(`R[--   1  --   3  ... --   9  --  11]`, s)

	T = New(Of(Int), WithShape(2, 4, 16))
	data = T.ints()
	for i := 0; i < len(data); i++ {
		data[i] = i
	}
	T.ResetMask(false)
	for i := 0; i < len(data); i += 2 {
		T.mask[i] = true
	}

	s = fmt.Sprintf("%d", T)
	assert.Equal(`⎡ --    1   --    3  ...  --   13   --   15⎤
⎢ --   17   --   19  ...  --   29   --   31⎥
⎢ --   33   --   35  ...  --   45   --   47⎥
⎣ --   49   --   51  ...  --   61   --   63⎦

⎡ --   65   --   67  ...  --   77   --   79⎤
⎢ --   81   --   83  ...  --   93   --   95⎥
⎢ --   97   --   99  ...  --  109   --  111⎥
⎣ --  113   --  115  ...  --  125   --  127⎦

`, s)

}
