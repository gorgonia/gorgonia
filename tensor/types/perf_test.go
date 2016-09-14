package types

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestIntsPool(t *testing.T) {
	assert := assert.New(t)

	ints := BorrowInts(2)
	assert.Equal(2, len(ints))

	// modify
	ints[0] = 111
	ints[1] = 222
	ints = ints[:1]

	ReturnInts(ints)
	ints = BorrowInts(2)

	assert.Equal(2, len(ints))
	assert.Equal(111, ints[0])
	assert.Equal(222, ints[1])

}
