package gorgonia

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestAddcmul(t *testing.T) {
	assert := assert.New(t)
	recv := tensor.New(tensor.WithShape(3, 4), tensor.WithBacking([]float64{
		1, 1, 1, 1,
		1, 1, 1, 1,
		1, 1, 1, 1,
	}))
	a := tensor.New(tensor.WithShape(3, 4), tensor.WithBacking([]float64{
		1, 2, 3, 4,
		5, 6, 1, 2,
		3, 4, 5, 6,
	}))
	b := tensor.New(tensor.WithShape(3, 4), tensor.WithBacking([]float64{
		10, 10, 10, 10,
		100, 100, 100, 100,
		1000, 1000, 1000, 1000,
	}))
	var scalar interface{} = float64(2)

	if err := addcmul(recv, a, b, scalar); err != nil {
		t.Fatal(err)
	}

	correctRecv := []float64{
		21, 41, 61, 81,
		1001, 1201, 201, 401,
		6001, 8001, 10001, 12001,
	}
	correctA := []float64{
		10, 20, 30, 40,
		500, 600, 100, 200,
		3000, 4000, 5000, 6000,
	}
	correctB := []float64{
		10, 10, 10, 10,
		100, 100, 100, 100,
		1000, 1000, 1000, 1000,
	}

	assert.Equal(correctRecv, recv.Data())
	assert.Equal(correctA, a.Data())
	assert.Equal(correctB, b.Data())

	// squaring
	recv.Zero()
	newA := []float64{
		1, 2, 3, 4,
		5, 6, 1, 2,
		3, 4, 5, 6,
	}
	copy(a.Data().([]float64), newA)
	copy(correctA, newA)
	for i := range correctA {
		correctA[i] *= correctA[i]
	}
	correctRecv = []float64{
		2, 8, 18, 32,
		50, 72, 2, 8,
		18, 32, 50, 72,
	}
	if err := addcmul(recv, a, a, scalar); err != nil {
		t.Fatal(err)
	}
	assert.Equal(correctRecv, recv.Data())
	assert.Equal(correctA, a.Data())

}
