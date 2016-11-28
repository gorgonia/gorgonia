package gorgonia

import (
	"testing"

	tf64 "github.com/chewxy/gorgonia/tensor/f64"
	"github.com/chewxy/hm"
	"github.com/stretchr/testify/assert"
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
	assert.Equal(t, 1.0, retVal.d.(Scalar).Any())

	x = tf64.NewTensor(tf64.WithBacking([]float64{1, 2, 3, 4}))
	x = tf64.NewTensor(tf64.WithBacking([]float64{4, 3, 2, 1}))
	op = newEBOByType(addOpType, TypeOf(x), TypeOf(y))
	xdv = constantDV(x)
	ydv = constantDV(y)
	retVal, err = dvBindVar(op, []*dualValue{xdv, ydv})
	if err != nil {
		t.Error(err)
	}
	assert.Equal(t, []float64{1, 1, 1, 1}, retVal.d.Data())
}
