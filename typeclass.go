package gorgonia

type typeClass interface {
	addInstance(t Type)
}

type simpleTC struct {
	instances typeSet
}

func (tc *simpleTC) addInstance(t Type) {
	tc.instances = tc.instances.Add(t)
}

/* CONSTANTS */

var arithable *simpleTC
var floats *simpleTC
var summable *simpleTC
var scalarOrTensor *simpleTC

func init() {
	arithable = new(simpleTC)
	arithable.addInstance(Int)
	arithable.addInstance(Int32)
	arithable.addInstance(Int64)
	arithable.addInstance(Float64)
	arithable.addInstance(Float32)

	floats = new(simpleTC)
	floats.addInstance(Float64)
	floats.addInstance(Float32)

	summable = new(simpleTC)
	summable.addInstance(Int)
	summable.addInstance(Int32)
	summable.addInstance(Int64)
	summable.addInstance(Float64)
	summable.addInstance(Float32)

	scalarOrTensor = new(simpleTC)
	scalarOrTensor.addInstance(Float64)
	scalarOrTensor.addInstance(Float32)
}
