package types

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

var normtests = []struct {
	val   float64
	valid bool
}{
	{float64(UnorderedNorm()), true},
	{float64(FrobeniusNorm()), true},
	{float64(NuclearNorm()), true},
	{float64(InfNorm()), true},
	{float64(NegInfNorm()), true},
	{0.0, true},
	{1.0, true},
	{-1.0, true},
	{3.14, false},
	{-3.14, false},
	{math.Float64frombits(0x7ff8000000000004), false},
	{math.Float64frombits(0x7ff8000000000004), false},
	{math.Float64frombits(0x7ff8000000000004), false},
}

func TestNorms(t *testing.T) {
	assert := assert.New(t)
	for _, nts := range normtests {
		n := NormOrder(nts.val)
		valid := n.Valid()
		assert.Equal(nts.valid, valid, "Uh oh. Val: %v should be %v, Got %v instead", nts.val, nts.valid, valid)
	}

	// test constructors, because why not
	assert.Equal(NormOrder(1.0), Norm(1))
	assert.Equal(NormOrder(-1.0), Norm(-1))

	// testing equivalence of floats is a fraught endeavour. Best to tet them
	assert.True(FrobeniusNorm().IsFrobenius())
	assert.True(NuclearNorm().IsNuclear())
	assert.True(InfNorm().IsInf(1))
	assert.True(NegInfNorm().IsInf(-1))
}
