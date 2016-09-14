package tensori

import (
	"bytes"
	"fmt"
	"strconv"

	"github.com/chewxy/gorgonia/tensor/types"
)

var fmtFlags = [...]rune{'+', '-', '#', ' ', '0'}

// Format pretty prints a *Tensor. Valid flags are:
//		'#' - prints the whole length without eliding
// 		'+' - prints the metadata as well
//		'-' - prints out the *Tensor as a flat slice (essentially printing out t.data). By default it will only print 10 elements, or 5 if the verb is 's'. To print fully, use the '#' flag
// These verbs defines what will be printed:
//		'v' - default
// 		's' - compressed. Everything will try to fit into as few lines as possible
// 		'f', 'g', 'x'... - the *Tensor values will be printed as if %f, %g... were called
//
// Since a *Tensor is a representation that extends beyond a 2D matrix, and monitors are only sadly 2D surfaces,
// any additional dimension higher than 2 will be notated by a new line. For example, in a (2, 2, 3) Tensor, this will be shown:
// 		⎡0   1   2⎤
// 		⎣3   4   5⎦
//
//		⎡6   7   8⎤
// 		⎣9  10  11⎦
//
// For vectors, Format just prints the array as is. A 'R' or 'C' is added in front to indicate if it's a row vector or column vector
func (t *Tensor) Format(state fmt.State, c rune) {
	if t.IsScalar() {
		var formatBuf bytes.Buffer
		formatBuf.WriteRune('%')
		for _, flag := range fmtFlags {
			if state.Flag(int(flag)) {
				formatBuf.WriteRune(flag)
			}
		}
		if width, ok := state.Width(); ok {
			formatBuf.WriteString(strconv.Itoa(width))
		}
		if prec, ok := state.Precision(); ok {
			formatBuf.WriteRune('.')
			formatBuf.WriteString(strconv.Itoa(prec))
		}
		formatBuf.WriteRune(c)

		fmt.Fprintf(state, formatBuf.String(), t.data[0])
		return
	}

	var rows, cols int
	if t.IsVector() {
		rows = 1
		cols = t.Size()
	} else {
		rows = t.Shape()[t.Dims()-2]
		cols = t.Shape()[t.Dims()-1]
	}

	metadata := state.Flag('+')
	flat := state.Flag('-')
	extended := state.Flag('#')
	compress := c == 's'
	hElision := "... "
	vElision := ".\n.\n.\n"
	width, _ := state.Width()
	buf := make([]byte, 0, 10)

	var base int
	switch c {
	case 'b':
		base = 2
	case 'd':
		base = 10
	case 'o':
		base = 8
	case 'x', 'X':
		base = 16
	default:
		base = 10
	}

	// use the elements from the first half to determine the max width needed
	consideration := t.data
	if len(t.data) > 100 {
		consideration = t.data[:len(t.data)/2]
	}
	for _, v := range consideration {
		buf = strconv.AppendInt(buf, int64(v), base)
		if len(buf) > width {
			width = len(buf)
		}
		// empty buffer
		buf = buf[:0]
	}

	pad := make([]byte, types.MaxInt(width, 2))
	for i := range pad {
		pad[i] = ' '
	}

	var printedCols, printedRows int

	switch {
	case flat && extended:
		printedCols = len(t.data)
	case flat && compress:
		printedCols = 5
		hElision = "⋯ "
	case flat:
		printedCols = 10
	case extended:
		printedCols = cols
		printedRows = rows
	case compress:
		printedCols = types.MinInt(cols, 4)
		printedRows = types.MinInt(rows, 4)
		hElision = "⋯ "
		vElision = "  ⋮  \n"
	default:
		printedCols = types.MinInt(cols, 8)
		printedRows = types.MinInt(rows, 8)
	}

	// start printing
	if metadata {
		var userFriendly string
		switch {
		case t.IsScalar():
			userFriendly = "Scalar"
		case t.IsVector():
			userFriendly = "Vector"
		case t.Dims() == 2:
			userFriendly = "Matrix"
		default:
			userFriendly = fmt.Sprintf("%d-Tensor", t.Dims())
		}
		fmt.Fprintf(state, "%s %v %v\n", userFriendly, t.Shape(), t.Strides())
	}

	if flat {
		fmt.Fprintf(state, "[")
		switch {
		case extended:
			for i, v := range t.data {
				buf = strconv.AppendInt(buf[:0], int64(v), base)
				state.Write(buf)
				if i < len(t.data)-1 {
					state.Write(pad[:1])
				}
			}
		case t.viewOf != nil:
			it := newIterator(t)
			var c, i int
			var err error
			for i, err = it.next(); err == nil; i, err = it.next() {

				buf = strconv.AppendInt(buf[:0], int64(t.data[i]), base)
				state.Write(buf)
				state.Write(pad[:1])

				c++
				if c >= printedCols {
					fmt.Fprintf(state, hElision)
					break
				}
			}
			if err != nil {
				if _, noop := err.(NoOpError); !noop {
					fmt.Fprintf(state, "ERROR ITERATING: %v", err)

				}
			}
		default:
			for i := 0; i < printedCols; i++ {
				buf = strconv.AppendInt(buf[:0], int64(t.data[i]), base)
				state.Write(buf)
				state.Write(pad[:1])

			}

			if printedCols < len(t.data) {
				fmt.Fprintf(state, hElision)
			}
		}
		fmt.Fprintf(state, "]")
		return
	}

	var rowStride int
	var colStride int
	switch {
	case t.IsColVec():
		if t.Strides()[0] != 1 {
			colStride = t.Strides()[0]
		} else {
			colStride = 1
		}
		rowStride = len(t.data)
	case t.IsRowVec():
		colStride = 1
		rowStride = len(t.data)
	case t.IsVector() && !t.IsColVec() && !t.IsColVec():
		colStride = 1
		rowStride = len(t.data)
	default:
		rowStride = t.Strides()[t.Dims()-2]
		colStride = t.Strides()[t.Dims()-1]
	}

	first := true

	var last string
	for row := 0; row*rowStride < len(t.data); row++ {
		switch {
		case t.IsColVec():
			fmt.Fprintf(state, "C[")
			last = "]"
		case t.IsRowVec():
			fmt.Fprintf(state, "R[")
			last = "]"
		case t.IsVector() && !t.IsColVec() && !t.IsRowVec():
			fmt.Fprintf(state, "[")
			last = "]"
		case first:
			fmt.Fprintf(state, "⎡")
			last = "⎤\n"
			first = false
		case ((row+1)%rows == 0):
			fmt.Fprint(state, "⎣")

			var lastBuf bytes.Buffer
			lastBuf.WriteString("⎦\n")
			for i := t.Dims(); i > 2; i-- {
				lastBuf.WriteString("\n") // one new newline for each dimension above 2
			}
			if t.Dims() > 2 {
				first = true
			}

			last = lastBuf.String()
		default:
			fmt.Fprintf(state, "⎢")
			last = "⎥\n"
		}

		if cols > printedCols {
			for col := 0; col < printedCols/2; col++ {
				idx := row*rowStride + col*colStride
				v := t.data[idx]
				buf = strconv.AppendInt(buf[:0], int64(v), base)

				state.Write(pad[:width-len(buf)]) // prepad
				state.Write(buf)                  // write the number
				state.Write(pad[:2])              // pad with a space
			}
			fmt.Fprintf(state, hElision)
			for col := cols - (printedCols / 2); col < cols; col++ {
				idx := row*rowStride + col*colStride
				v := t.data[idx]

				buf = strconv.AppendInt(buf[:0], int64(v), base)
				state.Write(pad[:width-len(buf)])
				state.Write(buf)
				if col < cols-1 {
					state.Write(pad[:2])
				}
			}
		} else {
			for col := 0; col < cols; col++ {
				idx := row*rowStride + col*colStride
				v := t.data[idx]
				buf = strconv.AppendInt(buf[:0], int64(v), base)
				state.Write(pad[:width-len(buf)])
				state.Write(buf)

				if col < cols-1 {
					state.Write(pad[:2])
				}
			}
		}

		fmt.Fprintf(state, last)

		if rows > printedRows && row+1 == printedRows/2 {
			row = rows - (printedRows / 2) - 1
			fmt.Fprintf(state, vElision)
		}
	}
}

func (t *Tensor) String() string {
	return fmt.Sprintf("%v", t)
}
