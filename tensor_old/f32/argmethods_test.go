package tensorf32

import (
	"testing"

	ti "github.com/chewxy/gorgonia/tensor/i"
	"github.com/chewxy/gorgonia/tensor/types"
	"github.com/chewxy/math32"
	"github.com/stretchr/testify/assert"
)

var basicArgT = NewTensor(WithShape(2, 3, 4, 5, 2), WithBacking([]float32{
	9.46384501e-01, 3.99576106e-01, 4.01371511e-01,
	2.76123881e-01, 6.61266621e-01, 1.20752237e+00,
	-1.90380901e-01, -1.07060986e+00, 1.69514654e+00,
	-4.96724874e-01, 2.86233490e-01, -4.41745317e-01,
	-1.59209301e+00, 3.73386054e-01, -9.50425200e-01,
	-2.89315702e-01, -1.72317444e-01, 1.33859648e+00,
	5.07639877e-01, 1.49833415e+00, 5.70929674e-01,
	3.00589219e+00, -6.05960792e-01, 4.40067343e-01,
	3.34369970e-02, 9.18335188e-01, 1.54918173e-01,
	-9.15680088e-01, 1.11862334e+00, 7.42162060e-01,
	-1.81103376e+00, 3.83660781e+00, 9.85253246e-02,
	-4.03920387e-01, 2.34620817e+00, -8.04983694e-01,
	-1.81096004e+00, 2.23711647e+00, -1.17968928e+00,
	1.49492298e+00, 3.10818792e-01, -1.50686056e+00,
	2.18313637e-01, 3.96213929e-02, -4.84482684e-01,
	1.64934816e+00, -2.57447564e-01, -9.91756696e-01,
	-9.45340490e-01, 3.13188937e-01, 1.34961331e+00,
	5.78517508e-01, 1.08034108e+00, 7.94414364e-01,
	2.76050262e-01, 5.38689278e-02, -1.40104075e+00,
	1.54288680e-01, -1.13238687e+00, 1.81449575e+00,
	-3.13002579e-01, -9.15880718e-01, 6.11056396e-01,
	4.16551110e-01, -4.33164772e-01, 3.13964618e-01,
	-7.47722089e-01, -2.41060117e-01, 1.16977901e-01,
	-3.55668581e-01, -2.63082850e-02, -4.08363294e-01,
	-7.66794220e-01, 5.49480039e-01, 7.99975402e-01,
	1.47018192e+00, -6.21941595e-01, -1.38082660e+00,
	5.26174919e-01, -2.14118275e-02, 2.97353067e-01,
	-9.55843985e-01, 5.85992800e-01, 2.18708773e+00,
	2.92560142e+00, -2.19508063e-01, -1.96902950e+00,
	-1.54806255e+00, 6.88957161e-01, 1.02011890e+00,
	6.95781020e-01, 2.84771634e-01, 2.72811005e-01,
	1.07613138e+00, -1.39683677e-01, -1.00102787e+00,
	6.74509652e-01, -1.64403821e+00, -4.27171355e-01,
	-1.18301603e+00, 5.60665113e-02, -5.44865970e-01,
	5.54159083e-02, 1.33746517e+00, -1.60279180e+00,
	1.12626392e+00, 1.58301784e+00, 1.03902296e+00,
	3.17507410e-01, -1.40258797e-01, -1.56827285e-01,
	4.76796700e-01, -4.31394436e-01, -5.39486767e-01,
	-5.11722053e-01, -3.80032540e+00, -2.01012072e+00,
	-2.02521465e+00, 8.17793774e-01, -1.35665304e+00,
	-1.80119934e+00, 1.00317724e+00, 2.81175825e-01,
	3.77362925e-01, 1.00943148e+00, 9.83977962e-01,
	-7.97205246e-01, 8.37790247e-01, -1.66961544e-01,
	-6.19878474e-01, -4.88402408e-01, -2.86075118e-01,
	-1.54427115e+00, 3.41379769e-03, -2.60353316e-01,
	-1.33796977e+00, -1.00040389e+00, 5.40014618e-01,
	2.44741986e+00, -7.33537867e-01, -1.41483963e-01,
	2.96195441e+00, -1.27274075e+00, -9.45607258e-01,
	1.22463313e+00, -2.31914473e-01, -2.20718669e+00,
	-8.41126472e-02, -1.25739523e-01, 8.34496432e-01,
	-2.56269941e+00, -1.22130832e+00, -1.28120911e+00,
	-9.13805354e-01, -3.63065168e-01, 6.76685936e-01,
	-1.07429073e+00, 1.09596898e+00, 1.39332066e+00,
	6.47932021e-01, 5.00989335e-01, -1.90550413e+00,
	2.83643553e-01, 7.66496520e-01, 2.33083228e-01,
	-2.00844678e-01, 1.32992161e+00, -5.05727782e-01,
	-9.41628611e-01, -1.01402102e+00, -6.80625972e-01,
	5.37752492e-01, -3.80538094e-01, 4.23380077e-02,
	-1.92070653e+00, 7.64182130e-01, 1.77471539e+00,
	-1.05689180e+00, -1.76446882e-01, 5.35090784e-01,
	-3.87393017e-01, -1.59211642e-01, 3.43843328e-01,
	-2.74293343e+00, -3.62028582e-01, -1.54440944e+00,
	4.50571475e-01, 1.71346683e+00, -7.77770653e-02,
	-2.46840580e-01, 6.14454298e-01, 2.29180499e+00,
	-7.23791598e-01, 6.34391877e-01, -1.58628875e-01,
	-1.22199267e-01, -1.49201890e+00, 6.16500317e-01,
	-7.57158163e-01, 3.53440359e-01, 5.08734589e-01,
	1.40689680e+00, -3.99692567e-02, 9.72539929e-01,
	1.22642727e+00, -1.05588803e-01, -8.80732946e-01,
	-9.78338091e-01, -6.09638759e-01, -2.32506449e+00,
	-1.04079431e+00, 2.53328794e-01, -9.42362599e-01,
	-4.58890221e-02, 3.50358667e-01, 3.23566193e-01,
	1.22103962e+00, 1.21358384e+00, 1.27441174e-01,
	5.77249715e-01, -8.04855341e-01, -1.36758911e-03,
	-4.00103138e-01, -8.14700072e-01, 5.72819553e-01,
	1.79205521e+00, -8.62631088e-01, -6.62383619e-01,
	-4.58028871e-01, 2.41264833e-01, 1.45800558e-03,
	4.14076552e-01, -1.90934626e+00, -6.91605343e-01,
	1.17159870e+00, 1.13732759e+00, 7.89015938e-01,
	6.09622480e-01, 2.13060304e-01, 8.90292105e-01,
}))

var argmaxCorrect = []struct {
	correctShape types.Shape
	correctData  []int
}{
	{types.Shape{3, 4, 5, 2}, []int{
		0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,
		0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0,
		1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0,
		1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
		0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1,
		1, 1, 1, 0, 1,
	}},
	{types.Shape{2, 4, 5, 2}, []int{
		0, 0, 2, 2, 2, 1, 0, 1, 0, 2, 1, 1, 1, 2, 1, 1, 2, 0, 0, 1, 0, 0, 1,
		2, 0, 2, 2, 2, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 2, 0, 2, 2, 1, 2, 2, 0,
		1, 0, 0, 0, 0, 1, 1, 1, 2, 1, 1, 2, 0, 2, 0, 0, 1, 2, 0, 2, 1, 1, 1,
		0, 1, 1, 1, 1, 2, 2, 2, 0, 0, 2,
	}},
	{types.Shape{2, 3, 5, 2}, []int{
		0, 3, 0, 2, 3, 0, 2, 3, 0, 1, 1, 1, 1, 1, 3, 0, 0, 1, 3, 1, 1, 3, 0,
		0, 0, 2, 2, 2, 3, 0, 2, 2, 0, 0, 2, 0, 0, 3, 1, 2, 3, 3, 2, 0, 0, 1,
		1, 2, 2, 1, 0, 0, 0, 0, 0, 2, 1, 1, 3, 3,
	}},
	{types.Shape{2, 3, 4, 2}, []int{
		4, 2, 4, 4, 4, 0, 2, 0, 0, 2, 0, 4, 1, 1, 2, 2, 2, 1, 0, 1, 3, 1, 4,
		0, 2, 0, 4, 3, 2, 0, 4, 3, 3, 1, 3, 2, 3, 3, 0, 0, 2, 0, 3, 3, 2, 2,
		2, 2,
	}},
	{types.Shape{2, 3, 4, 5}, []int{
		0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1,
		0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0,
		1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1,
		0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0,
		0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1,
		1, 1, 0, 0, 1,
	}},
}

func TestArgmax(t *testing.T) {
	assert := assert.New(t)
	var T *Tensor
	var argmax *ti.Tensor
	var err error

	T = basicArgT.Clone()
	for i := 0; i < T.Dims(); i++ {
		if argmax, err = T.Argmax(i); err != nil {
			t.Error(err)
			continue
		}

		assert.True(argmaxCorrect[i].correctShape.Eq(argmax.Shape()), "Argmax(%d) error. Want shape %v, got %v", i, argmaxCorrect[i].correctShape, argmax.Shape())
		assert.Equal(argmaxCorrect[i].correctData, argmax.Data(), "Argmax(%d) error. Want data %v, got %v", i, argmaxCorrect[i].correctData, argmax.Data())
	}

	// test all axes
	if argmax, err = T.Argmax(types.AllAxes); err != nil {
		t.Error(err)
		return
	}

	assert.True(argmax.IsScalar())
	assert.Equal(31, argmax.ScalarValue())

	// test with NaN
	T = NewTensor(WithShape(4), WithBacking([]float32{1, 2, math32.NaN(), 4}))
	if argmax, err = T.Argmax(types.AllAxes); err != nil {
		t.Error(err)
		return
	}
	assert.True(argmax.IsScalar())
	assert.Equal(2, argmax.ScalarValue())

	// test with +Inf
	T = NewTensor(WithShape(4), WithBacking([]float32{1, 2, math32.Inf(1), 4}))
	if argmax, err = T.Argmax(types.AllAxes); err != nil {
		t.Error(err)
		return
	}
	assert.True(argmax.IsScalar())
	assert.Equal(2, argmax.ScalarValue())

	// test with +Inf
	T = NewTensor(WithShape(4), WithBacking([]float32{1, 2, math32.Inf(-1), 4}))
	if argmax, err = T.Argmax(types.AllAxes); err != nil {
		t.Error(err)
		return
	}
	assert.True(argmax.IsScalar())
	assert.Equal(3, argmax.ScalarValue())

	// idiotsville
	_, err = T.Argmax(10)
	assert.NotNil(err)
}

var argminCorrect = []struct {
	correctShape types.Shape
	correctData  []int
}{
	{types.Shape{3, 4, 5, 2}, []int{
		1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1,
		1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1,
		0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1,
		0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1,
		1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0,
		0, 0, 0, 1, 0,
	}},
	{types.Shape{2, 4, 5, 2}, []int{
		2, 1, 1, 1, 1, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 2, 1, 2, 1, 2, 1, 1, 0,
		1, 2, 1, 1, 0, 1, 1, 0, 1, 1, 2, 2, 2, 2, 2, 0, 2, 0, 1, 2, 0, 1, 1,
		2, 2, 1, 2, 2, 0, 0, 2, 1, 0, 0, 1, 1, 0, 2, 1, 0, 1, 1, 1, 0, 2, 2,
		1, 0, 0, 2, 0, 0, 1, 1, 2, 1, 1,
	}},
	{types.Shape{2, 3, 5, 2}, []int{
		3, 1, 1, 3, 1, 3, 3, 0, 3, 0, 2, 0, 3, 0, 0, 1, 1, 3, 1, 2, 3, 0, 3,
		3, 2, 3, 3, 3, 1, 3, 3, 3, 1, 2, 3, 1, 2, 2, 0, 1, 1, 0, 3, 2, 1, 2,
		3, 1, 0, 0, 1, 2, 3, 2, 1, 0, 0, 0, 0, 0,
	}},
	{types.Shape{2, 3, 4, 2}, []int{
		3, 3, 1, 0, 1, 3, 0, 2, 4, 0, 3, 2, 3, 0, 1, 3, 3, 3, 4, 3, 2, 0, 3,
		2, 0, 4, 1, 2, 3, 1, 0, 0, 4, 0, 2, 3, 0, 1, 3, 2, 3, 4, 0, 1, 3, 1,
		1, 1,
	}},
	{types.Shape{2, 3, 4, 5}, []int{
		1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0,
		1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1,
		0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0,
		1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1,
		1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0,
		0, 0, 1, 1, 0,
	}},
}

func TestArgmin(t *testing.T) {
	assert := assert.New(t)
	var T *Tensor
	var argmin *ti.Tensor
	var err error

	T = basicArgT.Clone()
	for i := 0; i < T.Dims(); i++ {
		if argmin, err = T.Argmin(i); err != nil {
			t.Error(err)
			continue
		}

		assert.True(argminCorrect[i].correctShape.Eq(argmin.Shape()), "Argmax(%d) error. Want shape %v, got %v", i, argminCorrect[i].correctShape, argmin.Shape())
		assert.Equal(argminCorrect[i].correctData, argmin.Data(), "Argmax(%d) error. Want data %v, got %v", i, argminCorrect[i].correctData, argmin.Data())
	}

	// test all axes
	if argmin, err = T.Argmin(types.AllAxes); err != nil {
		t.Error(err)
		return
	}

	assert.True(argmin.IsScalar())
	assert.Equal(115, argmin.ScalarValue())

	// test with NaN
	T = NewTensor(WithShape(4), WithBacking([]float32{1, 2, math32.NaN(), 4}))
	if argmin, err = T.Argmin(types.AllAxes); err != nil {
		t.Error(err)
		return
	}
	assert.True(argmin.IsScalar())
	assert.Equal(2, argmin.ScalarValue())

	// test with +Inf
	T = NewTensor(WithShape(4), WithBacking([]float32{1, 2, math32.Inf(1), 4}))
	if argmin, err = T.Argmin(types.AllAxes); err != nil {
		t.Error(err)
		return
	}
	assert.True(argmin.IsScalar())
	assert.Equal(0, argmin.ScalarValue())

	// test with -Inf
	T = NewTensor(WithShape(4), WithBacking([]float32{1, 2, math32.Inf(-1), 4}))
	if argmin, err = T.Argmin(types.AllAxes); err != nil {
		t.Error(err)
		return
	}
	assert.True(argmin.IsScalar())
	assert.Equal(2, argmin.ScalarValue())

	// idiotsville
	_, err = T.Argmin(10)
	assert.NotNil(err)
}
