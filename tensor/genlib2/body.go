package main

// generic loop templates

type LoopBody struct {
	TypedOp
	Range    string
	Left     string
	Right    string
	IterName string
}

const (
	genericLoopRaw = `for i := range {{.Range}} {
		{{template "check" . -}}
		{{template "loopbody" .}}
	}`

	genericIncrLoopRaw = `for i := range {{.Range}}{
		{{template "check" . -}}
		{{template "loopbody" . -}}
	}`

	genericBinaryIterLoopRaw = `var i, j int
	var validi, validj bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, validj, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi && validj {
			{{template "check" . -}}
			{{template "loopbody" . -}}
		}
	}`

	genericBinaryIncrIterLoopRaw = `var i, j, k int
	var validi, validj, validk bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, validj, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if k, validk, err = iit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi && validj && validk {
			{{template "check" . -}}
			{{template "loopbody" . -}}
		}
	}`

	genericUnaryIterLoopRaw = `var i int
	var validi bool
	for {
		if i, validi, err = {{.IterName}}.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			{{template "check" . -}}
			{{template "loopbody" . -}}
		}
	}`

	genericUnaryIterIncrLoopRaw = `var i, k int
	var validi, validk bool
	for {
		if i, validi, err = {{.IterName}}.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if k, validk, err = iit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi && validk {
			{{template "check" . -}}
			{{template "loopbody" . -}}
		}
	}
	`

	// ALL THE SYNTACTIC ABSTRACTIONS!
	// did I mention how much I hate C-style macros? Now I'm doing them instead

	basicSet = `{{if .IsFunc -}}
			{{.Range}}[i] = {{ template "callFunc" . -}}
		{{else -}}
			{{.Range}}[i] = {{template "opDo" . -}}
		{{end -}}`

	basicIncr = `{{if .IsFunc -}}
			{{.Range}}[i] += {{template "callFunc" . -}}
		{{else -}}
			{{.Range}}[i] += {{template "opDo" . -}}
		{{end -}}`

	iterIncrLoopBody = `{{if .IsFunc -}}
			{{.Range}}[k] += {{template "callFunc" . -}}
		{{else -}}
			{{.Range}}[k] += {{template "opDo" . -}}
		{{end -}}`

	binOpCallFunc = `{{if eq "complex64" .Kind.String -}}
		complex64({{template "symbol" .Kind}}(complex128({{.Left}}), complex128({{.Right}})))
		{{else -}}
		{{template "symbol" .Kind}}({{.Left}}, {{.Right}})
		{{end -}}`

	binOpDo = `{{.Left}} {{template "symbol" .Kind}} {{.Right}}`

	unaryOpDo = `{{template "symbol" .Kind}}{{.Left}}`

	unaryOpCallFunc = `{{template "symbol" .Kind}}({{.Left}})`

	check0 = `if {{.Right}} == 0 {
		errs = append(errs, i)
		{{.Range}}[i] = 0
		continue
	}
	`
)

// renamed
const (
	vvLoopRaw         = genericLoopRaw
	vvIncrLoopRaw     = genericIncrLoopRaw
	vvIterLoopRaw     = genericBinaryIterLoopRaw
	vvIterIncrLoopRaw = genericBinaryIncrIterLoopRaw

	mixedLoopRaw         = genericLoopRaw
	mixedIncrLoopRaw     = genericIncrLoopRaw
	mixedIterLoopRaw     = genericUnaryIterLoopRaw
	mixedIterIncrLoopRaw = genericUnaryIterIncrLoopRaw
)
