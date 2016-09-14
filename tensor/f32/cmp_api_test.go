package tensorf32

import (
	"testing"

	tb "github.com/chewxy/gorgonia/tensor/b"
	"github.com/chewxy/gorgonia/tensor/types"
	"github.com/stretchr/testify/assert"
)

func TestCmp(t *testing.T) {
	var correctBack []bool
	var expected, got types.Tensor
	var err error
	assert := assert.New(t)

	backA := []float32{1, 2, 3, 4, 5}
	backB := []float32{5, 4, 3, 2, 1}

	Ta := NewTensor(WithBacking(backA))
	Tb := NewTensor(WithBacking(backB))

	t.Logf("Lt T-T")
	if got, err = Lt(Ta, Tb); err != nil {
		t.Error(err)
	}
	correctBack = []bool{true, true, false, false, false}
	expected = tb.NewTensor(tb.WithBacking(correctBack))
	assert.Equal(expected, got)
	assert.NotNil(got)

	t.Logf("Gt T-T")
	if got, err = Gt(Ta, Tb); err != nil {
		t.Error(err)
	}
	correctBack = []bool{false, false, false, true, true}
	expected = tb.NewTensor(tb.WithBacking(correctBack))
	assert.Equal(expected, got)
	assert.NotNil(got)

	t.Logf("Lte T-T")
	if got, err = Lte(Ta, Tb); err != nil {
		t.Error(err)
	}
	correctBack = []bool{true, true, true, false, false}
	expected = tb.NewTensor(tb.WithBacking(correctBack))
	assert.Equal(expected, got)
	assert.NotNil(got)

	t.Logf("Gte T-T")
	if got, err = Gte(Ta, Tb); err != nil {
		t.Error(err)
	}
	correctBack = []bool{false, false, true, true, true}
	expected = tb.NewTensor(tb.WithBacking(correctBack))
	assert.Equal(expected, got)
	assert.NotNil(got)

	t.Logf("Eq T-T")
	if got, err = Eq(Ta, Tb); err != nil {
		t.Error(err)
	}
	correctBack = []bool{false, false, true, false, false}
	expected = tb.NewTensor(tb.WithBacking(correctBack))
	assert.Equal(expected, got)
	assert.NotNil(got)

	t.Logf("Ne T-T")
	if got, err = Ne(Ta, Tb); err != nil {
		t.Error(err)
	}
	correctBack = []bool{true, true, false, true, true}
	expected = tb.NewTensor(tb.WithBacking(correctBack))
	assert.Equal(expected, got)
	assert.NotNil(got)

	/* TENSOR-SCALAR TEST */

	t.Logf("Lt T-S")
	if got, err = Lt(Ta, float32(3)); err != nil {
		t.Error(err)
	}
	correctBack = []bool{true, true, false, false, false}
	expected = tb.NewTensor(tb.WithBacking(correctBack))
	assert.Equal(expected, got)
	assert.NotNil(got)

	t.Logf("Gt T-S")
	if got, err = Gt(Ta, float32(3)); err != nil {
		t.Error(err)
	}
	correctBack = []bool{false, false, false, true, true}
	expected = tb.NewTensor(tb.WithBacking(correctBack))
	assert.Equal(expected, got)
	assert.NotNil(got)

	t.Logf("Lte T-S")
	if got, err = Lte(Ta, float32(3)); err != nil {
		t.Error(err)
	}
	correctBack = []bool{true, true, true, false, false}
	expected = tb.NewTensor(tb.WithBacking(correctBack))
	assert.Equal(expected, got)
	assert.NotNil(got)

	t.Logf("Gte T-S")
	if got, err = Gte(Ta, float32(3)); err != nil {
		t.Error(err)
	}
	correctBack = []bool{false, false, true, true, true}
	expected = tb.NewTensor(tb.WithBacking(correctBack))
	assert.Equal(expected, got)
	assert.NotNil(got)

	t.Logf("Eq T-S")
	if got, err = Eq(Ta, float32(3)); err != nil {
		t.Error(err)
	}
	correctBack = []bool{false, false, true, false, false}
	expected = tb.NewTensor(tb.WithBacking(correctBack))
	assert.Equal(expected, got)
	assert.NotNil(got)

	t.Logf("Ne T-S")
	if got, err = Ne(Ta, float32(3)); err != nil {
		t.Error(err)
	}
	correctBack = []bool{true, true, false, true, true}
	expected = tb.NewTensor(tb.WithBacking(correctBack))
	assert.Equal(expected, got)
	assert.NotNil(got)

	/* SCALAR-TENSOR TEST */

	t.Logf("Lt S-T")
	if got, err = Lt(float32(3), Tb); err != nil {
		t.Error(err)
	}
	correctBack = []bool{true, true, false, false, false}
	expected = tb.NewTensor(tb.WithBacking(correctBack))
	assert.Equal(expected, got)
	assert.NotNil(got)

	t.Logf("Gt S-T")
	if got, err = Gt(float32(3), Tb); err != nil {
		t.Error(err)
	}
	correctBack = []bool{false, false, false, true, true}
	expected = tb.NewTensor(tb.WithBacking(correctBack))
	assert.Equal(expected, got)
	assert.NotNil(got)

	t.Logf("Lte S-T")
	if got, err = Lte(float32(3), Tb); err != nil {
		t.Error(err)
	}
	correctBack = []bool{true, true, true, false, false}
	expected = tb.NewTensor(tb.WithBacking(correctBack))
	assert.Equal(expected, got)
	assert.NotNil(got)

	t.Logf("Gte S-T")
	if got, err = Gte(float32(3), Tb); err != nil {
		t.Error(err)
	}
	correctBack = []bool{false, false, true, true, true}
	expected = tb.NewTensor(tb.WithBacking(correctBack))
	assert.Equal(expected, got)
	assert.NotNil(got)

	t.Logf("Eq S-T")
	if got, err = Eq(float32(3), Tb); err != nil {
		t.Error(err)
	}
	correctBack = []bool{false, false, true, false, false}
	expected = tb.NewTensor(tb.WithBacking(correctBack))
	assert.Equal(expected, got)
	assert.NotNil(got)

	t.Logf("Ne S-T")
	if got, err = Ne(float32(3), Tb); err != nil {
		t.Error(err)
	}
	correctBack = []bool{true, true, false, true, true}
	expected = tb.NewTensor(tb.WithBacking(correctBack))
	assert.Equal(expected, got)
	assert.NotNil(got)

	/* IDIOT TEST */

	t.Logf("Lt idiots")
	if got, err = Lt(float32(3), float32(3)); err == nil {
		t.Error("Expected error")
	}

	t.Logf("Gt idiots")
	if got, err = Gt(float32(3), float32(3)); err == nil {
		t.Error("Expected error")
	}

	t.Logf("Lte idiots")
	if got, err = Lte(float32(3), float32(3)); err == nil {
		t.Error("Expected error")
	}

	t.Logf("Gte idiots")
	if got, err = Gte(float32(3), float32(3)); err == nil {
		t.Error("Expected error")
	}

	t.Logf("Eq idiots")
	if got, err = Eq(float32(3), float32(3)); err == nil {
		t.Error("Expected error")
	}

	t.Logf("Ne idiots")
	if got, err = Ne(float32(3), float32(3)); err == nil {
		t.Error("Expected error")
	}

}
