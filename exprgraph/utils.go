package exprgraph

import (
	"fmt"
	"strconv"

	"github.com/chewxy/hm"
	"gorgonia.org/gorgonia/types"
	"gorgonia.org/gorgonia/values/dual"
	"gorgonia.org/tensor"
)

// T2T tries to find a `tensor.Tensor` from a Tensor
// it returns nil if no tensor is found
func T2T[DT tensor.Num](a Tensor) tensor.Basic[DT] {
	switch t := a.(type) {
	case Node:
		if v, ok := t.(valuelifter); ok {
			return v.Value().(tensor.Basic[DT])
		}
		return nil
	case *dual.Dual[DT]:
		return t
	case tensor.Basic[DT]:
		return t
	default:
		return nil
	}
}

func typeof(n Node) hm.Type {
	dt := n.Dtype()
	shp := n.Shape()
	if shp.IsScalar() {
		return dt
	}
	return types.MakeTensorType(shp.Dims(), dt)
}

func consFmtStr(a fmt.State, c rune) string {
	retVal := "%"
	acceptable := []rune{'+', '-', ' ', '#', '0'}
	for _, f := range acceptable {
		if a.Flag(int(f)) {
			retVal = retVal + string(f)
		}
	}
	width, wok := a.Width()
	prec, pok := a.Precision()
	if wok {
		retVal = retVal + strconv.Itoa(width)
	}
	if pok {
		retVal = retVal + "." + strconv.Itoa(prec)
	}
	retVal = retVal + string(c)
	return retVal
}

func in(l []int64, want int64) bool {
	for _, v := range l {
		if v == want {
			return true
		}
	}
	return false
}
