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

var nonFloatConditionalUnarySymbolTemplates = [...]string{
	`{{if isFloat .Kind -}} 
	{{.Range}}[{{.Index0}}] = {{mathPkg .Kind}}Abs({{.Range}}[{{.Index0}}]) {{else -}}
	if {{.Range}}[{{.Index0}}] < 0 {
		{{.Range}}[{{.Index0}}] = -{{.Range}}[{{.Index0}}]
	}{{end -}}`, // abs

	`if {{.Range}}[{{.Index0}}] < 0 {
		{{.Range}}[{{.Index0}}] = -1
	} else if {{.Range}}[{{.Index0}}] > 0 {
		{{.Range}}[{{.Index0}}] = 1
	}`, // sign
}

var unconditionalNumUnarySymbolTemplates = [...]string{
	"-",                            // neg
	"1/",                           // inv
	"{{.Range}}[i]*",               // square
	"{{.Range}}[i]*{{.Range}}[i]*", // cube
}

var unconditionalFloatUnarySymbolTemplates = [...]string{
	"{{mathPkg .Kind}}Exp",
	"{{mathPkg .Kind}}Tanh",
	"{{mathPkg .Kind}}Log",
	"{{mathPkg .Kind}}Log2",
	"{{mathPkg .Kind}}Log10",
	"{{mathPkg .Kind}}Sqrt",
	"{{mathPkg .Kind}}Cbrt",
	`{{asType .Kind}}(1)/{{mathPkg .Kind}}Sqrt`,
}

var funcOptUse = map[string]string{
	"reuse":  ",WithReuse(reuse)",
	"incr":   ",WithIncr(incr)",
	"unsafe": ",UseUnsafe()",
}

var funcOptCheck = map[string]string{
	"reuse": `if reuse != ret {
			t.Errorf("Expected reuse to be the same as retVal")
			return false
	}

	`,

	"incr": "",

	"unsafe": `if ret != a {
		t.Errorf("Expected ret to be the same as a")
		return false
	}

	`,
}

var funcOptDecl = map[string]string{
	"reuse":  "reuse := New(Of(a.t), WithShape(a.Shape().Clone()...))\n",
	"incr":   "incr := New(Of(a.t), WithShape(a.Shape().Clone()...))\n",
	"unsafe": "",
}

var funcOptCorrect = map[string]string{
	"reuse": "",
	"incr": `incr.Memset(identityVal(100, a.t))
	correct.Add(incr, UseUnsafe())
	`,
	"unsafe": "",
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
	"unexport":           unexport,
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
	"isBoolRepr":         isBoolRepr,
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

	"mathPkg":       mathPkg,
	"vecPkg":        vecPkg,
	"bitSizeOf":     bitSizeOf,
	"getalias":      getalias,
	"interfaceName": interfaceName,

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
	"VecAdd": "Add",
	"VecSub": "Sub",
	"VecMul": "Mul",
	"VecDiv": "Div",
	"VecPow": "Pow",
	"VecMod": "Mod",

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

var arithBinOps []arithOp
var cmpBinOps []basicBinOp
var typedAriths []TypedBinOp
var typedCmps []TypedBinOp

var conditionalUnaries []unaryOp
var unconditionalUnaries []unaryOp
var specialUnaries []UnaryOp
var typedCondUnaries []TypedUnaryOp
var typedUncondUnaries []TypedUnaryOp
var typedSpecialUnaries []TypedUnaryOp

var allKinds []reflect.Kind

func init() {
	// kinds

	for k := reflect.Invalid + 1; k < reflect.UnsafePointer+1; k++ {
		allKinds = append(allKinds, k)
	}

	// ops

	arithBinOps = []arithOp{
		{basicBinOp{"", "Add", false, isAddable}, "numberTypes", true, 0, false, "", true, false},
		{basicBinOp{"", "Sub", false, isNumber}, "numberTypes", false, 0, true, "Add", false, true},
		{basicBinOp{"", "Mul", false, isNumber}, "numberTypes", true, 1, false, "", true, false},
		{basicBinOp{"", "Div", false, isNumber}, "numberTypes", false, 1, true, "Mul", false, false},
		{basicBinOp{"", "Pow", true, isFloatCmplx}, "floatcmplxTypes", true, 1, false, "", false, false},
		{basicBinOp{"", "Mod", false, isNonComplexNumber}, "nonComplexNumberTypes", false, 0, false, "", false, false},
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

	conditionalUnaries = []unaryOp{
		{"", "Abs", false, isSignedNumber},
		{"", "Sign", false, isSignedNumber},
	}
	for i := range conditionalUnaries {
		conditionalUnaries[i].symbol = nonFloatConditionalUnarySymbolTemplates[i]
	}

	unconditionalUnaries = []unaryOp{
		{"", "Neg", false, isNumber},
		{"", "Inv", false, isNumber},
		{"", "Square", false, isNumber},
		{"", "Cube", false, isNumber},

		{"", "Exp", true, isFloatCmplx},
		{"", "Tanh", true, isFloatCmplx},
		{"", "Log", true, isFloatCmplx},
		{"", "Log2", true, isFloat},
		{"", "Log10", true, isFloatCmplx},
		{"", "Sqrt", true, isFloatCmplx},
		{"", "Cbrt", true, isFloat},
		{"", "InvSqrt", true, isFloat}, // TODO: cmplx requires to much finagling to the template. Come back to it later
	}
	nonF := len(unconditionalNumUnarySymbolTemplates)
	for i := range unconditionalNumUnarySymbolTemplates {
		unconditionalUnaries[i].symbol = unconditionalNumUnarySymbolTemplates[i]
	}
	for i := range unconditionalFloatUnarySymbolTemplates {
		unconditionalUnaries[i+nonF].symbol = unconditionalFloatUnarySymbolTemplates[i]
	}

	specialUnaries = []UnaryOp{
		specialUnaryOp{unaryOp{clampBody, "Clamp", false, isNonComplexNumber}, []string{"min", "max"}},
	}

	// typed operations

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

	for _, uo := range conditionalUnaries {
		for _, k := range allKinds {
			tu := TypedUnaryOp{
				UnaryOp: uo,
				k:       k,
			}
			typedCondUnaries = append(typedCondUnaries, tu)
		}
	}

	for _, uo := range unconditionalUnaries {
		for _, k := range allKinds {
			tu := TypedUnaryOp{
				UnaryOp: uo,
				k:       k,
			}
			typedUncondUnaries = append(typedUncondUnaries, tu)
		}
	}

	for _, uo := range specialUnaries {
		for _, k := range allKinds {
			tu := TypedUnaryOp{
				UnaryOp: uo,
				k:       k,
			}
			typedSpecialUnaries = append(typedSpecialUnaries, tu)
		}
	}
}
