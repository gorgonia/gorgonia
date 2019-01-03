//go:generate statik -src=./htdocs

package tracer

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"net/http"
	"os"
	"os/exec"
	"strconv"

	"github.com/rakyll/statik/fs"
	"gonum.org/v1/gonum/graph"

	"gorgonia.org/gorgonia"
	"gorgonia.org/gorgonia/debugger/dot"
	_ "gorgonia.org/gorgonia/debugger/tracer/statik" // Initialize the FS for static files
	"gorgonia.org/tensor"
)

// StartDebugger runs a http webserver
func StartDebugger(g graph.Directed, listenAddress string) error {
	statikFS, err := fs.New()
	if err != nil {
		return err
	}

	b, err := dot.Marshal(g)
	if err != nil {
		return err
	}
	svg, err := generateSVG(b)
	if err != nil {
		return err
	}
	handler := http.NewServeMux()
	nodes := g.Nodes()
	nodes.Reset()
	for nodes.Next() {
		n := nodes.Node()
		_, ok := n.(http.Handler)
		if ok {
			//  handler.Handle(fmt.Sprintf("/nodes/%p", n), n.(http.Handler))
			handler.HandleFunc(fmt.Sprintf("/nodes/%p", n), nodeHandler(n))
			handler.HandleFunc(fmt.Sprintf("/nodes/%p/images/pic.png", n), nodeHandlerPic(n))
		}
	}
	handler.HandleFunc("/graph.dot", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain; charset=UTF-8")
		io.Copy(w, bytes.NewReader(b))
	})
	handler.HandleFunc("/graph.svg", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "image/svg+xml; charset=UTF-8")
		io.WriteString(w, string(svg))
	})
	handler.Handle("/", http.FileServer(statikFS))

	return http.ListenAndServe(listenAddress, handler)
}

func generateSVG(b []byte) ([]byte, error) {

	dotProcess := exec.Command("dot", "-Tsvg")

	// Set the stdin stdout and stderr of the dot subprocess
	stdinOfDotProcess, err := dotProcess.StdinPipe()
	if err != nil {
		return nil, err
	}
	defer stdinOfDotProcess.Close() // the doc says subProcess.Wait will close it, but I'm not sure, so I kept this line
	readCloser, err := dotProcess.StdoutPipe()
	if err != nil {
		return nil, err
	}
	dotProcess.Stderr = os.Stderr

	// Actually run the dot subprocess
	if err = dotProcess.Start(); err != nil { //Use start, not run
		return nil, err //replace with logger, or anything you want
	}
	fmt.Fprintf(stdinOfDotProcess, "%v", string(b))
	stdinOfDotProcess.Close()
	// Read from stdout and store it in the correct structure
	var buf bytes.Buffer
	buf.ReadFrom(readCloser)

	dotProcess.Wait()
	if err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func nodeHandlerPic(n graph.Node) http.HandlerFunc {

	return func(w http.ResponseWriter, r *http.Request) {
		n, ok := n.(*gorgonia.Node)
		if !ok {
			http.Error(w, "Node is not a Gorgonia node", 500)
			return
		}
		layers, ok := r.URL.Query()["layer"]
		if !ok || len(layers) != 1 {
			http.Error(w, "expeced a 'layer' argument", 500)
			return
		}
		layer, err := strconv.Atoi(layers[0])
		if err != nil {
			http.Error(w, err.Error(), 500)
			return
		}

		v, ok := n.Value().(*tensor.Dense)
		if !ok {
			http.Error(w, "can only decode a Dense", 500)
			return

		}
		if len(v.Shape()) != 4 {
			http.Error(w, "Cannot draw a tensor that has not 4 dimension", 500)
			return
		}

		width := v.Shape()[2]
		height := v.Shape()[3]
		im := image.NewGray(image.Rectangle{Max: image.Point{X: width, Y: height}})
		for w := 0; w < width; w++ {
			for h := 0; h < height; h++ {
				v, err := v.At(0, layer, w, h)
				if err != nil {
					panic(err)
				}
				im.Set(w, h, color.Gray{uint8(v.(float32))})
			}
		}
		enc := png.Encoder{}
		w.Header().Set("Content-Type", "image/png")
		err = enc.Encode(w, im)
		if err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
	}
}
func nodeHandler(n graph.Node) http.HandlerFunc {

	return func(w http.ResponseWriter, r *http.Request) {
		n, ok := n.(*gorgonia.Node)
		if !ok {
			http.Error(w, "Node is not a Gorgonia node", 500)
			return
		}

		image := true
		v, ok := n.Value().(*tensor.Dense)
		if !ok {
			image = false
			//http.Error(w, "can only decode a Dense", 500)
			//return

		}
		if len(v.Shape()) != 4 {
			image = false
			//http.Error(w, "Cannot draw a tensor that has not 4 dimension", 500)
			//return
		}
		w.Header().Set("Content-Type", "text/html; charset=UTF-8")
		fmt.Fprint(w, "<html><body>\n")

		if image {
			for i := 0; i < v.Shape()[1]; i++ {
				fmt.Fprintf(w, `<img src="%v/images/pic.png?layer=%v" width=50></img>&nbsp;`, r.URL.Path, i)
			}
		}
		fmt.Fprintf(w, `<br><pre>%i</pre>`, n.Value())
		fmt.Fprintf(w, `<br><pre>%#s"</pre>`, n.Value())

		fmt.Fprint(w, "</body></html>\n")
	}
}
