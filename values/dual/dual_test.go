package dual

import (
	"testing"

	"github.com/chewxy/hm"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/tensor"
)

func Test_dvBind0(t *testing.T) {
	var x, y, z values.Value
	var xT, yT, zT hm.Type
	x, xT = values.AnyToScalar(2.0)
	y, yT = values.AnyToScalar(3.0)
	z, zT = values.AnyToScalar(0.0)

	op := newEBOByType(addOpType, xT, yT)
	xdv := constantDV(x)
	ydv := constantDV(y)
	zdv := constantDV(z)
	dvBind0(op, zdv, []*Dual{xdv, ydv})

	t.Logf("%v %v", zdv, zT)

}

func TestDVBindVar(t *testing.T) {
	var x, y values.Value
	var xT, yT hm.Type
	x, xT = values.AnyToScalar(2.0)
	y, yT = values.AnyToScalar(3.0)

	op := newEBOByType(addOpType, xT, yT)
	xdv := constantDV(x)
	ydv := constantDV(y)
	retVal, err := dvBindVar(op, []*Dual{xdv, ydv})
	if err != nil {
		t.Error(err)
	}
	assert.Equal(t, 1.0, retVal.d.Data())

	x = tensor.New(tensor.WithBacking([]float64{4, 3, 2, 1}))
	op = newEBOByType(addOpType, types.TypeOf(x), types.TypeOf(y))
	xdv = constantDV(x)
	ydv = constantDV(y)
	if retVal, err = dvBindVar(op, []*Dual{xdv, ydv}); err != nil {
		t.Error(err)
	}
	assert.Equal(t, []float64{1, 1, 1, 1}, retVal.d.Data())
}
