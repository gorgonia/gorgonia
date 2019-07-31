package dot

import (
	"fmt"
	"strings"

	"gonum.org/v1/gonum/graph/encoding"
	"gorgonia.org/gorgonia"
)

type node struct {
	n *gorgonia.Node
}

func (n *node) ID() int64 {
	return n.n.ID()
}

// DOTID is used for the graphviz output. It fulfils the gonum encoding interface
func (n *node) DOTID() string {
	return fmt.Sprintf("Node_%p", n.n)
}

// Attributes is for graphviz output. It specifies the "label" of the node (a table)
func (n *node) Attributes() []encoding.Attribute {
	var htmlEscaper = strings.NewReplacer(
		`&`, "&amp;",
		`'`, "&#39;", // "&#39;" is shorter than "&apos;" and apos was not in HTML until HTML5.
		`<`, "&lt;",
		`>`, "&gt;",
		`{`, "\\n",
		`}`, "\\n",
		`"`, "&#34;", // "&#34;" is shorter than "&quot;".
		`const`, "const|", // "&#34;" is shorter than "&quot;".
	)
	attrs := []encoding.Attribute{
		encoding.Attribute{
			Key:   "id",
			Value: fmt.Sprintf(`"%p"`, n.n),
		},
		encoding.Attribute{
			Key:   "shape",
			Value: "Mrecord",
		},
		encoding.Attribute{
			Key:   "label",
			Value: fmt.Sprintf(`"{{%s|%#x}|{Op|%s}|{Shape|%v}}"`, n.n.Name(), n.ID(), htmlEscaper.Replace(fmt.Sprintf("%s", n.n.Op())), n.n.Shape()),
		},
	}
	return attrs
}
