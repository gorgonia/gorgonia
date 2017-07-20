package main

import (
	"fmt"
	"reflect"
	"strings"
)

type TypeClass func(a reflect.Kind) bool

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

func isNonComplexNumber(a reflect.Kind) bool {
	for _, v := range nonComplexNumber {
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

func getalias(name string) string {
	if nice, ok := nameMaps[name]; ok {
		return nice
	}
	return name
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
