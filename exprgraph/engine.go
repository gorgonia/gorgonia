package exprgraph

// NOTE: this is copied directly from engine.go in package tensor/dense. If that changes, this needs to change too

import (
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/dense"
	stdeng "gorgonia.org/tensor/engines"
)

func defaultEngine[DT any]() tensor.Engine {
	var e tensor.Engine
	var v DT

	switch any(v).(type) { // TODO fill in rest of numeric engine
	case int:
		e = stdeng.StdOrderedNumEngine[int, *dense.Dense[int]]{}
	case int8:
		e = stdeng.StdOrderedNumEngine[int8, *dense.Dense[int8]]{}
	case int16:
		e = stdeng.StdOrderedNumEngine[int16, *dense.Dense[int16]]{}
	case int32:
		e = stdeng.StdOrderedNumEngine[int32, *dense.Dense[int32]]{}
	case int64:
		e = stdeng.StdOrderedNumEngine[int64, *dense.Dense[int64]]{}
	case uint8:
		e = stdeng.StdOrderedNumEngine[uint8, *dense.Dense[uint8]]{}
	case uint16:
		e = stdeng.StdOrderedNumEngine[uint16, *dense.Dense[uint16]]{}
	case uint32:
		e = stdeng.StdOrderedNumEngine[uint32, *dense.Dense[uint32]]{}
	case uint64:
		e = stdeng.StdOrderedNumEngine[uint64, *dense.Dense[uint64]]{}
	case float32:
		e = dense.StdFloat32Engine[*dense.Dense[float32]]{}
	case float64:
		e = dense.StdFloat64Engine[*dense.Dense[float64]]{}
	default:
		// rv := reflect.ValueOf(v)
		// if rv.Comparable() {
		// 	return stdeng.StdComparableEngine[DT, *dense.Dense[DT]]
		// }
		e = stdeng.StdEng[DT, *dense.Dense[DT]]{}
	}
	return e
}
