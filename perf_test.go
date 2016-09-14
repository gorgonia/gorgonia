package gorgonia

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNodePool(t *testing.T) {
	assert := assert.New(t)
	n := borrowNode()
	assert.NotNil(n)
}
