package main

type LoopBody struct {
	TypedBinOp
	Range string
	Left  string
	Right string
}

const (
	vvLoopRaw = `for i := range {{.Range}} {
		{{template "check" . -}}
		{{if .IsFunc -}}
			{{.Left}} = {{template "symbol" .Kind}}({{.Left}}, {{.Right}})
		{{else -}}
			{{.Left}} {{template "symbol" .Kind}}= {{.Right}}
		{{end -}}
	}`

	vvIncrLoopRaw = `for i := range {{.Range}}{
		{{template "check" . -}}
		{{if .IsFunc -}}
			{{.Range}}[i] += {{template "symbol" .Kind}}({{.Left}}, {{.Right}})
		{{else -}}
			{{.Range}}[i] += {{.Left}} {{template "symbol" .Kind}} {{.Right}}
		{{end -}}
	}`

	vvIterLoopRaw = `var i, j int
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

			{{if .IsFunc -}}
				{{.Left}} = {{template "symbol" .Kind}}({{.Left}}, {{.Right}})
			{{else -}}
				{{.Left}} {{template "symbol" .Kind}}= {{.Right}}
			{{end -}}		
		}
	}`

	vvIterIncrLoopRaw = `var i, j, k int
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
			{{if .IsFunc -}}
				{{.Range}}[i] += {{template "symbol" .Kind}}({{.Left}}, {{.Right}})
			{{else -}}
				{{.Range}}[i] += {{.Left}} {{template "symbol" .Kind}}  {{.Right}}
			{{end -}}		
		}
	}`

	mixLoopRaw = `for i := range {{.Range}}{
		{{template "check" . -}}

		{{if .IsFunc -}}
			{{.Range}}[i] = {{template "symbol" .Kind}}({{.Left}}, {{.Right}})
		{{else -}}
			{{.Range}}[i]  {{template "symbol" .Kind}}=  {{.Right}}
		{{end -}}
	}`

	mixIncrRaw = `for i := range {{.Range}}{
		{{template "check" . -}}

		{{if .IsFunc -}}
			{{.Range}}[i] += {{template "symbol" .Kind}}({{.Left}}, {{.Right}})
		{{else -}}
			{{.Range}}[i]  += {{.Left}} {{template "symbol" .Kind}}  {{.Right}}
		{{end -}}	
	}`

	check0 = `if {{.Right}} == 0 {
		errs = append(errs, i)
		{{.Range}}[i] = 0
		continue
	}
	`
)
