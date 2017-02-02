package tensorf32

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestVecTrans(t *testing.T) {
	assert := assert.New(t)
	backing := []float32{1, 2, 3, 4}
	T := NewTensor(WithShape(4, 1), WithBacking(backing))
	correct := make([]float32, 4)
	for i := range correct {
		correct[i] = T.data[i] + float32(1)
	}

	vecTrans(1, T.data)
	assert.Equal(correct, T.data)
}

func TestVecTransFrom(t *testing.T) {
	assert := assert.New(t)
	backing := []float32{1, 2, 3, 4}

	correct := make([]float32, len(backing))
	copy(correct, backing)
	for i, v := range correct {
		correct[i] = float32(1) - v
	}

	vecTransFrom(1, backing)
	assert.Equal(correct, backing)
	t.Logf("%v", backing)

}
