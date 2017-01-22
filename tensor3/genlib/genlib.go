package main

import (
	"reflect"
	"strings"
	"text/template"
)

var parameterizedKinds = [...]reflect.Kind{
	reflect.Array,
	reflect.Chan,
	reflect.Func,
	reflect.Interface,
	reflect.Map,
	reflect.Ptr,
	reflect.Slice,
	reflect.Struct,
}

var funcs = template.FuncMap{
	"lower":           strings.ToLower,
	"title":           strings.Title,
	"isParameterized": isParameterized,
	"short":           short,
	"clean":           clean,
	"strip":           strip,
}

func isParameterized(a reflect.Kind) bool {
	for _, v := range parameterizedKinds {
		if v == a {
			return true
		}
	}
	return false
}

var shortNames = map[reflect.Kind]string{
	reflect.Invalid:       "Invalid",
	reflect.Bool:          "B",
	reflect.Int:           "I",
	reflect.Int8:          "I8",
	reflect.Int16:         "I16",
	reflect.Int32:         "I32",
	reflect.Int64:         "I64",
	reflect.Uint:          "U",
	reflect.Uint8:         "U8",
	reflect.Uint16:        "U16",
	reflect.Uint32:        "U32",
	reflect.Uint64:        "U64",
	reflect.Uintptr:       "Uintptr",
	reflect.Float32:       "F32",
	reflect.Float64:       "F64",
	reflect.Complex64:     "C64",
	reflect.Complex128:    "C128",
	reflect.Array:         "Array",
	reflect.Chan:          "Chan",
	reflect.Func:          "Func",
	reflect.Interface:     "Interface",
	reflect.Map:           "Map",
	reflect.Ptr:           "Ptr",
	reflect.Slice:         "Slice",
	reflect.String:        "Str",
	reflect.Struct:        "Struct",
	reflect.UnsafePointer: "UnsafePointer",
}

func short(a reflect.Kind) string {
	return shortNames[a]
}

func clean(a string) string {
	if a == "unsafe.pointer" {
		return "unsafe.Pointer"
	}
	return a
}

func strip(a string) string {
	return strings.Replace(a, ".", "", -1)
}
