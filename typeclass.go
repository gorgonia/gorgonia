package gorgonia

import "github.com/chewxy/hm"

/* CONSTANTS */

var arithable *hm.SimpleTypeClass
var floats *hm.SimpleTypeClass
var summable *hm.SimpleTypeClass
var scalarOrTensor *hm.SimpleTypeClass

func init() {
	arithable = new(hm.SimpleTypeClass)
	arithable.AddInstance(Int)
	arithable.AddInstance(Int32)
	arithable.AddInstance(Int64)
	arithable.AddInstance(Float64)
	arithable.AddInstance(Float32)

	floats = new(hm.SimpleTypeClass)
	floats.AddInstance(Float64)
	floats.AddInstance(Float32)

	summable = new(hm.SimpleTypeClass)
	summable.AddInstance(Int)
	summable.AddInstance(Int32)
	summable.AddInstance(Int64)
	summable.AddInstance(Float64)
	summable.AddInstance(Float32)

	scalarOrTensor = new(hm.SimpleTypeClass)
	scalarOrTensor.AddInstance(Float64)
	scalarOrTensor.AddInstance(Float32)
}
