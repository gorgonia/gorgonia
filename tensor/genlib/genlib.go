package main

import (
	"fmt"
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

var rangeable = [...]reflect.Kind{
	reflect.Int,
	reflect.Int8,
	reflect.Int16,
	reflect.Int32,
	reflect.Int64,
	reflect.Uint,
	reflect.Uint8,
	reflect.Uint16,
	reflect.Uint32,
	reflect.Uint64,
	reflect.Float32,
	reflect.Float64,
	reflect.Complex64,
	reflect.Complex128,
}

var specialized = [...]reflect.Kind{
	reflect.Bool,
	reflect.Int,
	reflect.Int8,
	reflect.Int16,
	reflect.Int32,
	reflect.Int64,
	reflect.Uint,
	reflect.Uint8,
	reflect.Uint16,
	reflect.Uint32,
	reflect.Uint64,
	reflect.Float32,
	reflect.Float64,
	reflect.Complex64,
	reflect.Complex128,
	reflect.String,
}

var number = [...]reflect.Kind{
	reflect.Int,
	reflect.Int8,
	reflect.Int16,
	reflect.Int32,
	reflect.Int64,
	reflect.Uint,
	reflect.Uint8,
	reflect.Uint16,
	reflect.Uint32,
	reflect.Uint64,
	reflect.Float32,
	reflect.Float64,
	reflect.Complex64,
	reflect.Complex128,
}

var signedNumber = [...]reflect.Kind{
	reflect.Int,
	reflect.Int8,
	reflect.Int16,
	reflect.Int32,
	reflect.Int64,
	reflect.Float32,
	reflect.Float64,
	reflect.Complex64,
	reflect.Complex128,
}

var elEq = [...]reflect.Kind{
	reflect.Bool,
	reflect.Int,
	reflect.Int8,
	reflect.Int16,
	reflect.Int32,
	reflect.Int64,
	reflect.Uint,
	reflect.Uint8,
	reflect.Uint16,
	reflect.Uint32,
	reflect.Uint64,
	reflect.Uintptr,
	reflect.Float32,
	reflect.Float64,
	reflect.Complex64,
	reflect.Complex128,
	reflect.String,
	reflect.UnsafePointer,
}

var elOrd = [...]reflect.Kind{
	reflect.Int,
	reflect.Int8,
	reflect.Int16,
	reflect.Int32,
	reflect.Int64,
	reflect.Uint,
	reflect.Uint8,
	reflect.Uint16,
	reflect.Uint32,
	reflect.Uint64,
	// reflect.Uintptr, // comparison of pointers is not that great an idea - it can technically be done but should not be encouraged
	reflect.Float32,
	reflect.Float64,
	// reflect.Complex64,
	// reflect.Complex128,
	reflect.String, // strings are orderable and the assumption is lexicographic sorting
}

var div0panics = [...]reflect.Kind{
	reflect.Int,
	reflect.Int8,
	reflect.Int16,
	reflect.Int32,
	reflect.Int64,
	reflect.Uint,
	reflect.Uint8,
	reflect.Uint16,
	reflect.Uint32,
	reflect.Uint64,
}

var stdTypes = [...]string{
	"Bool",
	"Int",
	"Int8",
	"Int16",
	"Int32",
	"Int64",
	"Uint",
	"Uint8",
	"Uint16",
	"Uint32",
	"Uint64",
	"Float32",
	"Float64",
	"Complex64",
	"Complex128",
	"String",
	"Uintptr",
	"UnsafePointer",
}

var funcs = template.FuncMap{
	"lower":           strings.ToLower,
	"title":           strings.Title,
	"hasPrefix":       strings.HasPrefix,
	"hasSuffix":       strings.HasSuffix,
	"isParameterized": isParameterized,
	"isRangeable":     isRangeable,
	"isSpecialized":   isSpecialized,
	"isNumber":        isNumber,
	"isSignedNumber":  isSignedNumber,
	"isAddable":       isAddable,
	"isFloat":         isFloat,
	"isFloatCmplx":    isFloatCmplx,
	"isEq":            isEq,
	"isOrd":           isOrd,
	"panicsDiv0":      panicsDiv0,

	"short": short,
	"clean": clean,
	"strip": strip,

	"reflectKind": reflectKind,
	"asType":      asType,
	"sliceOf":     sliceOf,
	"getOne":      getOne,
	"setOne":      setOne,

	"mathPkg":   mathPkg,
	"vecPkg":    vecPkg,
	"bitSizeOf": bitSizeOf,

	"isntFloat": isntFloat,
}

func isParameterized(a reflect.Kind) bool {
	for _, v := range parameterizedKinds {
		if v == a {
			return true
		}
	}
	return false
}

func isNotParameterized(a reflect.Kind) bool { return !isParameterized(a) }

func isRangeable(a reflect.Kind) bool {
	for _, v := range rangeable {
		if v == a {
			return true
		}
	}
	return false
}

func isSpecialized(a reflect.Kind) bool {
	for _, v := range specialized {
		if v == a {
			return true
		}
	}
	return false
}

func isNumber(a reflect.Kind) bool {
	for _, v := range number {
		if v == a {
			return true
		}
	}
	return false
}

func isSignedNumber(a reflect.Kind) bool {
	for _, v := range signedNumber {
		if v == a {
			return true
		}
	}
	return false
}

func isAddable(a reflect.Kind) bool {
	if a == reflect.String {
		return true
	}
	return isNumber(a)
}

func isComplex(a reflect.Kind) bool {
	if a == reflect.Complex128 || a == reflect.Complex64 {
		return true
	}
	return false
}

func panicsDiv0(a reflect.Kind) bool {
	for _, v := range div0panics {
		if v == a {
			return true
		}
	}
	return false
}

func isEq(a reflect.Kind) bool {
	for _, v := range elEq {
		if v == a {
			return true
		}
	}
	return false
}

func isOrd(a reflect.Kind) bool {
	for _, v := range elOrd {
		if v == a {
			return true
		}
	}
	return false
}

func mathPkg(a reflect.Kind) string {
	if a == reflect.Float64 {
		return "math."
	}
	if a == reflect.Float32 {
		return "math32."
	}
	if a == reflect.Complex64 || a == reflect.Complex128 {
		return "cmplx."
	}
	return ""
}

func vecPkg(a reflect.Kind) string {
	if a == reflect.Float64 {
		return "vecf64."
	}
	if a == reflect.Float32 {
		return "vecf32."
	}
	return ""
}

func bitSizeOf(a reflect.Kind) string {
	switch a {
	case reflect.Int, reflect.Uint:
		return "0"
	case reflect.Int8, reflect.Uint8:
		return "8"
	case reflect.Int16, reflect.Uint16:
		return "16"
	case reflect.Int32, reflect.Uint32, reflect.Float32:
		return "32"
	case reflect.Int64, reflect.Uint64, reflect.Float64:
		return "64"
	}
	return "UNKNOWN BIT SIZE"
}

func isFloat(a reflect.Kind) bool {
	if a == reflect.Float32 || a == reflect.Float64 {
		return true
	}
	return false
}

func isFloatCmplx(a reflect.Kind) bool {
	if a == reflect.Float32 || a == reflect.Float64 || a == reflect.Complex64 || a == reflect.Complex128 {
		return true
	}
	return false
}

func isntFloat(a reflect.Kind) bool { return !isFloat(a) }

func isntComplex(a reflect.Kind) bool { return !isComplex(a) }

func filter(a []reflect.Kind, is func(reflect.Kind) bool) (retVal []reflect.Kind) {
	for _, k := range a {
		if is(k) {
			retVal = append(retVal, k)
		}
	}
	return
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

func reflectKind(a reflect.Kind) string {
	return strip(strings.Title(a.String()))
}

func asType(a reflect.Kind) string {
	return clean(a.String())
}

func sliceOf(a reflect.Kind) string {
	s := fmt.Sprintf("%ss()", strings.Title(a.String()))
	return strip(clean(s))
}

func getOne(a reflect.Kind) string {
	return fmt.Sprintf("Get%s", short(a))
}

func setOne(a reflect.Kind) string {
	return fmt.Sprintf("Set%s", short(a))
}
