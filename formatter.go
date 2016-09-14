package gorgonia

import (
	"fmt"
	"io"
	"reflect"
)

type mapFmt struct {
	m interface{} // map
}

// FmtNodeMap is a convenience function to print map[*Node]<T>
//
// The fmt flag that makes it all nicely formatted is "-". Because a map consists of two types (key's type
// and val's type), and the Go fmt verb doesn't quite allow us to do something like "%ds", a hack is introduced
// to enable nicer printing of map[*Node]<T>
//
// Here's the hack:
// The "#" flag is used to indicate if the map will use the Node's ID or Name when formatting the map.
//		%-v 	nodeName:%v
//		%-#v	nodeID:%v
//		%-d 	nodeName:%x
//		%-#d 	nodeID: %x
//		%-p 	nodeName:%p
// 		%-#p	nodeID:%p
//
// If the "-" flag is not found, then the formatter returns the default Go format for map[<T>]<T2>
func FmtNodeMap(m interface{}) mapFmt {
	refVal := reflect.ValueOf(m)
	if refVal.Kind() != reflect.Map {
		panic("Only expect maps in FmtNodeMap")
	}

	t := refVal.Type()
	keyType := t.Key()

	var n *Node
	if keyType != reflect.TypeOf(n) {
		panic("Only expected map[*Node]<T>")
	}

	return mapFmt{m}
}

func (mf mapFmt) defaultFmt(s fmt.State, c rune) {
	str := "%"
	for i := 0; i < 128; i++ {
		if s.Flag(i) {
			str += string(i)
		}
	}
	if w, ok := s.Width(); ok {
		str += fmt.Sprintf("%d", w)
	}
	if p, ok := s.Precision(); ok {
		str += fmt.Sprintf(".%d", p)
	}
	str += string(c)
	fmt.Fprintf(s, str, mf.m)
}

func (mf mapFmt) Format(s fmt.State, c rune) {
	w := s.(io.Writer)
	refVal := reflect.ValueOf(mf.m)
	var n *Node
	t := refVal.Type()
	keyType := t.Key()
	if keyType != reflect.TypeOf(n) {
		panic("Only map[*Node]<T> is expected")
	}

	var tmpl string
	switch {
	case c == 'v':
		if s.Flag('#') {
			tmpl = "\t%x: %v\n"
		} else {
			tmpl = "\t%s: %v\n"
		}
	case c == 'd':
		if s.Flag('#') {
			tmpl = "\t%x: %x\n"
		} else {
			tmpl = "\t%s: %x\n"
		}
	case c == 'p':
		if s.Flag('#') {
			tmpl = "\t%x: %p\n"
		} else {
			tmpl = "\t%s: %p\n"
		}
	default:
		tmpl = "\t%s: %s\n"
	}

	keys := refVal.MapKeys()
	if s.Flag('-') {
		if s.Flag('#') {
			// then key, will try its best to be a number

			fmt.Fprintf(w, "map[Node.ID]%s {\n", t.Elem())
			for i := 0; i < refVal.Len(); i++ {
				key := keys[i]
				val := refVal.MapIndex(key)

				meth := key.MethodByName("ID")
				id := meth.Call(nil)[0]

				valType := val.Type()
				if valType == reflect.TypeOf(n) {
					switch c {
					case 'd':
						valMeth := val.MethodByName("ID")
						valID := valMeth.Call(nil)[0]
						fmt.Fprintf(w, tmpl, id, valID)
					default:
						strMeth := val.MethodByName("String")
						str := strMeth.Call(nil)[0]
						fmt.Fprintf(w, tmpl, id, str)

					}
				} else {
					if _, ok := valType.MethodByName("Format"); ok {
						fmt.Fprintf(w, "\t%x: ", id)
						fmtMeth := val.MethodByName("Format")
						fmtMeth.Call([]reflect.Value{reflect.ValueOf(s), reflect.ValueOf(c)})
						fmt.Fprintf(w, "\n")
					} else if _, ok := valType.MethodByName("String"); ok {
						strMeth := val.MethodByName("String")
						str := strMeth.Call(nil)[0]
						fmt.Fprintf(w, tmpl, id, str)
					} else {
						fmt.Fprintf(w, tmpl, id, val)
					}
				}
			}
			w.Write([]byte("}"))

		} else {
			fmt.Fprintf(w, "map[Node.Name]%s {\n", t.Elem())
			for i := 0; i < refVal.Len(); i++ {
				key := keys[i]
				val := refVal.MapIndex(key)

				meth := key.MethodByName("String")
				id := meth.Call(nil)[0]

				valType := val.Type()
				if valType == reflect.TypeOf(n) {
					switch c {
					case 'd':
						valMeth := val.MethodByName("ID")
						valID := valMeth.Call(nil)[0]
						fmt.Fprintf(w, tmpl, id, valID)
					default:
						strMeth := val.MethodByName("String")
						str := strMeth.Call(nil)[0]
						fmt.Fprintf(w, tmpl, id, str)

					}
				} else {
					if _, ok := valType.MethodByName("Format"); ok {
						fmt.Fprintf(w, "\t%s: ", id)
						fmtMeth := val.MethodByName("Format")
						fmtMeth.Call([]reflect.Value{reflect.ValueOf(s), reflect.ValueOf(c)})
						fmt.Fprintf(w, "\n")
					} else if _, ok := valType.MethodByName("String"); ok {
						strMeth := val.MethodByName("String")
						str := strMeth.Call(nil)[0]
						fmt.Fprintf(w, tmpl, id, str)
					} else {
						fmt.Fprintf(w, tmpl, id, val)
					}
				}
			}
			w.Write([]byte("}"))
		}
		return
	}
	mf.defaultFmt(s, c)
}
