package shapes

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestAbstract_T(t *testing.T) {
	assert := assert.New(t)
	abstract := Abstract{Size(1), BinOp{Add, Size(1), Size(2)}}

	// noop
	a2, err := abstract.T(0, 1)
	if err == nil {
		t.Errorf("Expected a noop error")
	}
	if _, ok := err.(NoOpError); !ok {
		t.Errorf("Expected a noop error. Got %v instead", err)
	}
	assert.Equal(a2, abstract)

	a2, err = abstract.T(1, 0)
	if err != nil {
		t.Fatal(err)
	}
	correct := Abstract{abstract[1], abstract[0]}
	assert.Equal(correct, a2)
}
