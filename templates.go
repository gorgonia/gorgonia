package gorgonia

import (
	"fmt"
	"strings"
	"text/template"
)

const exprNodeTemplText = `<
<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" PORT="anchor" {{if isLeaf .}} COLOR="#00FF00;"{{else if isRoot . }} COLOR="#FF0000;" {{else if isMarked .}} COLOR="#0000FF;" {{end}}{{if isInput .}} BGCOLOR="lightyellow"{{else if isStmt .}} BGCOLOR="lightblue"{{end}}>

<TR><TD>{{printf "%x" .ID}}</TD><TD>{{printf "%v" .Name | html | dotEscape}} :: {{nodeType . | html | dotEscape }}</TD></TR>
{{if printOp . }}<TR><TD>Op</TD><TD>{{ opStr . | html | dotEscape }} :: {{ opType . | html | dotEscape }}</TD></TR>{{end}}
{{if hasShape .}}<TR><TD>Shape</TD><TD>{{ getShape .}}</TD></TR>{{end}}
<TR><TD>Overwrites Input {{overwritesInput . }}</TD><TD>Data On: {{.Device}}</TD></TR>
{{if hasGrad .}}<TR><TD>Value</TD><TD>Grad</TD></TR>
<TR><TD>{{printf "%+3.3s" .Value | dotEscape}}</TD><TD>{{getGrad . | dotEscape }} </TD></TR>
<TR><TD>Ptr: {{getValPtr . | dotEscape}} </TD><TD>Ptr: {{getGradPtr . | dotEscape}} </TD></TR>
{{else}}
<TR><TD>Value</TD><TD>{{printf "%+3.3s" .Value | dotEscape}}</TD></TR>
{{end}}

</TABLE>
>`

func dotEscape(s string) string {
	s = strings.Replace(s, "\n", "<BR />", -1)
	s = strings.Replace(s, "<nil>", "NIL", -1)
	return s
}

func printOp(n *Node) bool  { return n.op != nil && !n.isStmt }
func isLeaf(n *Node) bool   { return len(n.children) == 0 }
func isInput(n *Node) bool  { return n.isInput() }
func isMarked(n *Node) bool { return n.ofInterest }
func isRoot(n *Node) bool   { return n.isRoot() }
func isStmt(n *Node) bool   { return n.isStmt }
func hasShape(n *Node) bool { return n.shape != nil }
func hasGrad(n *Node) bool  { _, err := n.Grad(); return err == nil }
func opStr(n *Node) string  { return n.op.String() }
func opType(n *Node) string { return n.op.Type().String() }

func nodeType(n *Node) string {
	if n.t == nil {
		return "NIL"
	}
	return n.t.String()
}

func overwritesInput(n *Node) int {
	if n.op == nil {
		return -1
	}
	return n.op.OverwritesInput()
}

func getShape(n *Node) string {
	if !n.inferredShape {
		return fmt.Sprintf("%v", n.shape)
	}
	return fmt.Sprintf("<U>%v</U>", n.shape) // graphviz 2.38+ only supports <O>
}

func getGrad(n *Node) string {
	grad, err := n.Grad()
	if err == nil {
		return fmt.Sprintf("%+3.3s", grad)
	}
	return ""
}

func getGradPtr(n *Node) string {
	grad, err := n.Grad()
	if err == nil && grad != nil {
		return fmt.Sprintf("0x%x", grad.Uintptr())
	}
	return ""
}

func getValPtr(n *Node) string {
	if n.Value() == nil {
		return "<nil>"
	}
	return fmt.Sprintf("0x%dx", n.Value().Uintptr())
}

var funcMap = template.FuncMap{
	"dotEscape":       dotEscape,
	"printOp":         printOp,
	"isRoot":          isRoot,
	"isLeaf":          isLeaf,
	"isInput":         isInput,
	"isStmt":          isStmt,
	"isMarked":        isMarked,
	"hasShape":        hasShape,
	"hasGrad":         hasGrad,
	"getShape":        getShape,
	"getValPtr":       getValPtr,
	"getGrad":         getGrad,
	"getGradPtr":      getGradPtr,
	"overwritesInput": overwritesInput,
	"opStr":           opStr,
	"opType":          opType,
	"nodeType":        nodeType,
}

var (
	exprNodeTempl     *template.Template
	exprNodeJSONTempl *template.Template
)

func init() {
	exprNodeTempl = template.Must(template.New("node").Funcs(funcMap).Parse(exprNodeTemplText))
}
