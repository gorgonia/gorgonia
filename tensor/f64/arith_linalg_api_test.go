package tensorf64

import (
	"testing"

	"github.com/chewxy/gorgonia/tensor/types"
	"github.com/stretchr/testify/assert"
)

func TestDot(t *testing.T) {
	assert := assert.New(t)
	var a, b, c, r *Tensor
	var A, B, R, R2 *Tensor
	var err error
	var expectedShape types.Shape
	var expectedData []float64

	// vector-vector
	t.Log("Vec⋅Vec")
	a = NewTensor(WithShape(3, 1), WithBacking(RangeFloat64(0, 3)))
	b = NewTensor(WithShape(3, 1), WithBacking(RangeFloat64(0, 3)))
	r, err = Dot(a, b)

	expectedShape = types.Shape{1}
	expectedData = []float64{5}
	assert.Nil(err)
	assert.Equal(expectedData, r.data)
	assert.True(types.ScalarShape().Eq(r.Shape()))

	// vector-mat (which is the same as matᵀ*vec)
	t.Log("Vec⋅Mat dot, should be equal to Aᵀb")
	A = NewTensor(WithShape(3, 2), WithBacking(RangeFloat64(0, 6)))
	R, err = Dot(b, A)

	expectedShape = types.Shape{2, 1}
	expectedData = []float64{10, 13}
	assert.Nil(err)
	assert.Equal(expectedData, R.data)
	assert.Equal(expectedShape, R.Shape())

	t.Log("Mat⋅Vec, should be equal to b⋅A")
	R2, err = Dot(A, b)
	assert.Equal(R, R2)
	assert.Equal(expectedData, R2.data)
	assert.Equal(expectedShape, R2.Shape())

	// mat-mat
	t.Log("Mat⋅Mat")
	A = NewTensor(WithShape(4, 5), WithBacking(RangeFloat64(0, 20)))
	B = NewTensor(WithShape(5, 10), WithBacking(RangeFloat64(2, 52)))
	R, err = Dot(A, B)
	expectedShape = types.Shape{4, 10}
	expectedData = []float64{
		320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 870,
		905, 940, 975, 1010, 1045, 1080, 1115, 1150, 1185, 1420, 1480,
		1540, 1600, 1660, 1720, 1780, 1840, 1900, 1960, 1970, 2055, 2140,
		2225, 2310, 2395, 2480, 2565, 2650, 2735,
	}
	assert.Nil(err)
	assert.Equal(expectedData, R.data)
	assert.Equal(expectedShape, R.Shape())

	// T-T
	t.Log("3T⋅3T")
	A = NewTensor(WithShape(2, 3, 4), WithBacking(RangeFloat64(0, 24)))
	B = NewTensor(WithShape(3, 4, 2), WithBacking(RangeFloat64(0, 24)))
	R, err = Dot(A, B)
	expectedShape = types.Shape{2, 3, 3, 2}
	expectedData = []float64{
		28, 34,
		76, 82,
		124, 130,

		76, 98,
		252, 274,
		428, 450,

		124, 162,
		428, 466,
		732, 770,

		//

		172, 226,
		604, 658,
		1036, 1090,

		220, 290,
		780, 850,
		1340, 1410,

		268, 354,
		956, 1042,
		1644, 1730,
	}
	assert.Nil(err)
	assert.Equal(expectedData, R.data)
	assert.Equal(expectedShape, R.Shape())

	// T-T
	t.Log("3T⋅4T")
	A = NewTensor(WithShape(2, 3, 4), WithBacking(RangeFloat64(0, 24)))
	B = NewTensor(WithShape(2, 3, 4, 5), WithBacking(RangeFloat64(0, 120)))
	R, err = Dot(A, B)
	expectedShape = types.Shape{2, 3, 2, 3, 5}
	expectedData = []float64{
		70, 76, 82, 88, 94, 190, 196, 202, 208, 214, 310,
		316, 322, 328, 334, 430, 436, 442, 448, 454, 550, 556,
		562, 568, 574, 670, 676, 682, 688, 694, 190, 212, 234,
		256, 278, 630, 652, 674, 696, 718, 1070, 1092, 1114, 1136,
		1158, 1510, 1532, 1554, 1576, 1598, 1950, 1972, 1994, 2016, 2038,
		2390, 2412, 2434, 2456, 2478, 310, 348, 386, 424, 462, 1070,
		1108, 1146, 1184, 1222, 1830, 1868, 1906, 1944, 1982, 2590, 2628,
		2666, 2704, 2742, 3350, 3388, 3426, 3464, 3502, 4110, 4148, 4186,
		4224, 4262, 430, 484, 538, 592, 646, 1510, 1564, 1618, 1672,
		1726, 2590, 2644, 2698, 2752, 2806, 3670, 3724, 3778, 3832, 3886,
		4750, 4804, 4858, 4912, 4966, 5830, 5884, 5938, 5992, 6046, 550,
		620, 690, 760, 830, 1950, 2020, 2090, 2160, 2230, 3350, 3420,
		3490, 3560, 3630, 4750, 4820, 4890, 4960, 5030, 6150, 6220, 6290,
		6360, 6430, 7550, 7620, 7690, 7760, 7830, 670, 756, 842, 928,
		1014, 2390, 2476, 2562, 2648, 2734, 4110, 4196, 4282, 4368, 4454,
		5830, 5916, 6002, 6088, 6174, 7550, 7636, 7722, 7808, 7894, 9270,
		9356, 9442, 9528, 9614,
	}
	assert.Nil(err)
	assert.Equal(expectedData, R.data)
	assert.Equal(expectedShape, R.Shape())

	// T-v
	t.Log("3T⋅Vec")
	b = NewTensor(WithShape(4), WithBacking(RangeFloat64(0, 4)))
	R, err = Dot(A, b)
	expectedShape = types.Shape{2, 3}
	expectedData = []float64{
		14, 38, 62,
		86, 110, 134,
	}
	assert.Nil(err)
	assert.Equal(expectedData, R.data)
	assert.Equal(expectedShape, R.Shape())

	// 3T-vec with reuse
	t.Log("3T⋅vec with reuse")
	R.Zero()
	R, err = Dot(A, b, types.WithReuse(R))
	assert.Nil(err)
	assert.Equal(expectedData, R.data)
	assert.Equal(expectedShape, R.Shape())

	// v-m
	t.Log("vec⋅Mat with reuse")
	R.Zero()
	A = NewTensor(WithShape(4, 2), WithBacking(RangeFloat64(0, 8)))
	R, err = Dot(b, A, types.WithReuse(R))
	expectedShape = types.Shape{2}
	expectedData = []float64{28, 34}
	assert.Nil(err)
	assert.Equal(expectedData, R.data)
	assert.Equal(expectedShape, R.Shape())

	/* HERE BE STUPIDS */

	// different sizes of vectors
	c = NewTensor(WithShape(1, 4))
	_, err = Dot(a, c)
	assert.NotNil(err)

	// vector mat, but with shape mismatch
	B = NewTensor(WithShape(2, 3), WithBacking(RangeFloat64(0, 6)))
	_, err = Dot(b, B)
	assert.NotNil(err)

	// mat-mat but wrong reuse size
	A = NewTensor(WithShape(2, 2))
	R = NewTensor(WithShape(5, 10))
	_, err = Dot(A, B, types.WithReuse(R))
	assert.NotNil(err)

	// T-T but misaligned shape
	A = NewTensor(WithShape(2, 3, 4))
	B = NewTensor(WithShape(4, 2, 3))
	_, err = Dot(A, B)
	assert.NotNil(err)

	// v-T
	a = NewTensor(WithShape(4))
	B = NewTensor(WithShape(2, 3, 4))
	_, err = Dot(a, B)
	assert.NotNil(err)
}
