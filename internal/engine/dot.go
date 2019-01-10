package engine

import (
	"fmt"
	"net/http"
	"strings"

	"gonum.org/v1/gonum/graph/encoding"
)

// DOTID is used for the graphviz output. It fulfils the gonum encoding interface
func (n *Node) DOTID() string {
	//	return strconv.Itoa(int(n.ID()))
	return fmt.Sprintf("Node_%v_%p", n.id, n)
}

// Attributes is for graphviz output. It specifies the "label" of the node (a table)
func (n *Node) Attributes() []encoding.Attribute {
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
			Value: fmt.Sprintf(`"%p"`, n),
		},
		encoding.Attribute{
			Key:   "href",
			Value: fmt.Sprintf(`"/nodes/%p"`, n),
		},
		encoding.Attribute{
			Key:   "shape",
			Value: "Mrecord",
		},
		encoding.Attribute{
			Key:   "label",
			Value: fmt.Sprintf(`"{{%s|%#x}|{Op|%s}|{Shape|%v}}"`, n.name, n.ID(), htmlEscaper.Replace(fmt.Sprintf("%s", n.Op())), n.Shape()),
		},
	}
	return attrs
}

/*
type attributer []encoding.Attribute

func (a attributer) Attributes() []encoding.Attribute { return a }

// DOTAttributers to specify the top-level graph attributes for the graphviz generation
func (g *ExprGraph) DOTAttributers() (graph, node, edge encoding.Attributer) {
	// Create a special attribute "rank" to place the input at the same level in the graph

	graphAttributes := attributer{
		encoding.Attribute{
			Key:   "nodesep",
			Value: "1",
		},
		encoding.Attribute{
			Key:   "rankdir",
			Value: "TB",
		},
		encoding.Attribute{
			Key:   "ranksep",
			Value: `"1.5 equally"`,
		},
	}
	nodeAttributes := attributer{
		encoding.Attribute{
			Key:   "style",
			Value: "rounded",
		},
		encoding.Attribute{
			Key:   "fontname",
			Value: "monospace",
		},
		encoding.Attribute{
			Key:   "shape",
			Value: "none",
		},
	}
	return graphAttributes, nodeAttributes, attributer{}
}
*/

// ServeHTTP to get the value of the node via http
func (n *Node) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "%i", n.Value())

	fmt.Fprintf(w, "%#s", n.Value())
}
