package main

import (
	"reflect"
	"strings"
	"text/template"
)

var arithSymbolTemplates = [...]string{
	"+",
	"-",
	"*",
	"/",
	"{{mathPkg .}}Pow",
	"{{if isFloatCmplx .}}{{mathPkg .}}Mod{{else}}%{{end}}",
}

var cmpSymbolTemplates = [...]string{
	">",
	">=",
	"<",
	"<=",
	"==",
	"!=",
}

var nonFloatConditionalUnarySymbolTemplates = [...]*template.Template{
	template.Must(template.New("abs").Funcs(funcs).Parse(`if a[i] < 0 {
		a[i] = -a[i]
	}`)),
	template.Must(template.New("sign").Funcs(funcs).Parse(`if a[i] < 0 {
		a[i] = -1
	} else if a[i] > 0 {
		a[i] = 1
	}`)),
}

var unconditionalNumUnarySymbolTemplates = [...]*template.Template{
	template.Must(template.New("Neg").Funcs(funcs).Parse("-")),
	template.Must(template.New("Inv").Funcs(funcs).Parse("1/")),
	template.Must(template.New("Square").Funcs(funcs).Parse("a[i]*")),
	template.Must(template.New("Cube").Funcs(funcs).Parse("a[i]*a[i]*")),
}

var unconditionalFloatUnarySymbolTemplates = [...]*template.Template{
	template.Must(template.New("Exp").Funcs(funcs).Parse("{{mathPkg .}}.Exp")),
	template.Must(template.New("Log").Funcs(funcs).Parse("{{mathPkg .}}.Log")),
	template.Must(template.New("Log2").Funcs(funcs).Parse("{{mathPkg .}}.Log2")),
	template.Must(template.New("Log10").Funcs(funcs).Parse("{{mathPkg .}}.Log10")),
	template.Must(template.New("Sqrt").Funcs(funcs).Parse("{{mathPkg .}}.Sqrt")),
	template.Must(template.New("Cbrt").Funcs(funcs).Parse("{{mathPkg .}}.Cbrt")),
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

var nonComplexNumber = [...]reflect.Kind{
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

var boolRepr = [...]reflect.Kind{
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

var funcs = template.FuncMap{
	"lower":              strings.ToLower,
	"title":              strings.Title,
	"hasPrefix":          strings.HasPrefix,
	"hasSuffix":          strings.HasSuffix,
	"isParameterized":    isParameterized,
	"isRangeable":        isRangeable,
	"isSpecialized":      isSpecialized,
	"isNumber":           isNumber,
	"isSignedNumber":     isSignedNumber,
	"isNonComplexNumber": isNonComplexNumber,
	"isAddable":          isAddable,
	"isFloat":            isFloat,
	"isFloatCmplx":       isFloatCmplx,
	"isEq":               isEq,
	"isOrd":              isOrd,
	"panicsDiv0":         panicsDiv0,

	"short": short,
	"clean": clean,
	"strip": strip,

	"reflectKind": reflectKind,
	"asType":      asType,
	"sliceOf":     sliceOf,
	"getOne":      getOne,
	"setOne":      setOne,
	"trueValue":   trueValue,
	"falseValue":  falseValue,

	"mathPkg":   mathPkg,
	"vecPkg":    vecPkg,
	"bitSizeOf": bitSizeOf,
	"getalias":  getalias,

	"isntFloat": isntFloat,
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

var nameMaps = map[string]string{
	"AddVS": "Trans",
	"AddSV": "TransR",
	"SubVS": "TransInv",
	"SubSV": "TransInvR",
	"MulVS": "Scale",
	"MulSV": "ScaleR",
	"DivVS": "ScaleInv",
	"DivSV": "ScaleInvR",
	"PowVS": "PowOf",
	"PowSV": "PowOfR",

	"AddIncr": "IncrAdd",
	"SubIncr": "IncrSub",
	"MulIncr": "IncrMul",
	"DivIncr": "IncrDiv",
	"PowIncr": "IncrPow",
	"ModIncr": "IncrMod",

	"AddIncrVS": "IncrTrans",
	"AddIncrSV": "IncrTransR",
	"SubIncrVS": "IncrTransInv",
	"SubIncrSV": "IncrTransInvR",
}

var arithBinOps []basicBinOp
var cmpBinOps []basicBinOp
var typedAriths []TypedBinOp
var typedCmps []TypedBinOp
var allKinds []reflect.Kind

func init() {
	arithBinOps = []basicBinOp{
		{"", "Add", false, isAddable},
		{"", "Sub", false, isNumber},
		{"", "Mul", false, isNumber},
		{"", "Div", false, isNumber},
		{"", "Pow", true, isFloatCmplx},
		{"", "Mod", false, isNonComplexNumber},
	}
	for i := range arithBinOps {
		arithBinOps[i].symbol = arithSymbolTemplates[i]
	}

	cmpBinOps = []basicBinOp{
		{"", "Gt", false, isOrd},
		{"", "Gte", false, isOrd},
		{"", "Lt", false, isOrd},
		{"", "Lte", false, isOrd},
		{"", "Eq", false, isEq},
		{"", "Ne", false, isEq},
	}
	for i := range cmpBinOps {
		cmpBinOps[i].symbol = cmpSymbolTemplates[i]
	}

	for k := reflect.Invalid + 1; k < reflect.UnsafePointer+1; k++ {
		allKinds = append(allKinds, k)
	}

	for _, bo := range arithBinOps {
		for _, k := range allKinds {
			tb := TypedBinOp{
				BinOp: bo,
				k:     k,
			}
			typedAriths = append(typedAriths, tb)
		}
	}

	for _, bo := range cmpBinOps {
		for _, k := range allKinds {
			tb := TypedBinOp{
				BinOp: bo,
				k:     k,
			}
			typedCmps = append(typedCmps, tb)
		}
	}
}
