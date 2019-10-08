package gorgonia

import (
	"testing"

	"github.com/chewxy/hm"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func Test_dvBind0(t *testing.T) {
	var x, y, z Value
	var xT, yT, zT hm.Type
	x, xT = anyToScalar(2.0)
	y, yT = anyToScalar(3.0)
	z, zT = anyToScalar(0.0)

	op := newEBOByType(addOpType, xT, yT)
	xdv := constantDV(x)
	ydv := constantDV(y)
	zdv := constantDV(z)
	dvBind0(op, zdv, []*dualValue{xdv, ydv})

	t.Logf("%v %v", zdv, zT)

}

func TestDVBindVar(t *testing.T) {
	var x, y Value
	var xT, yT hm.Type
	x, xT = anyToScalar(2.0)
	y, yT = anyToScalar(3.0)

	op := newEBOByType(addOpType, xT, yT)
	xdv := constantDV(x)
	ydv := constantDV(y)
	retVal, err := dvBindVar(op, []*dualValue{xdv, ydv})
	if err != nil {
		t.Error(err)
	}
	assert.Equal(t, 1.0, retVal.d.Data())

	x = tensor.New(tensor.WithBacking([]float64{4, 3, 2, 1}))
	op = newEBOByType(addOpType, TypeOf(x), TypeOf(y))
	xdv = constantDV(x)
	ydv = constantDV(y)
	if retVal, err = dvBindVar(op, []*dualValue{xdv, ydv}); err != nil {
		t.Error(err)
	}
	assert.Equal(t, []float64{1, 1, 1, 1}, retVal.d.Data())
}
