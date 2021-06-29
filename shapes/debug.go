// +build debug

package shapes

import (
	"fmt"
	"io"
	"log"
)

// logstate prints the current state in a tab separated table that looks like this
// 	| current token | stack  | infix stack |
// 	|---------------|--------|-------------|
func (p *parser) logstate(name ...interface{}) {
	if p.log == nil {
		return
	}
	var cur tok = tok{}
	if p.qptr < len(p.queue) {
		cur = p.queue[p.qptr]
	}

	// print current token if no name given
	if len(name) > 0 {
		n := fmt.Sprintf(name[0].(string), name[1:]...)
		fmt.Fprintf(p.log, "%v\t[", n)
	} else {
		fmt.Fprintf(p.log, "%v\t[", cur)
	}

	// print stack
	for _, item := range p.stack {
		fmt.Fprintf(p.log, "%v;", item)
	}

	// print infix stack
	fmt.Fprintf(p.log, "]\t[")
	for _, item := range p.infixStack {
		fmt.Fprintf(p.log, "%q ", item.v)
	}
	fmt.Fprintf(p.log, "]\n")
}

func (p *parser) printTab(w io.Writer) {
	if p.log == nil {
		return
	}
	if w == nil {
		w = log.Default().Writer()
	}
	w.Write([]byte("Current Token\tStack\tInfix Stack\n"))
	w.Write([]byte(p.log.String()))
}
