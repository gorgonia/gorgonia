package shapes

import (
	"bytes"
	"fmt"
	"io"
	"log"
	"strconv"
	"strings"
	"unicode"

	"github.com/pkg/errors"
)

// Parse parses a string and returns a shape expression.
func Parse(a string) (retVal Expr, err error) {
	q, err := lex(a)
	if err != nil {
		return nil, err
	}

	p := new(parser)
	err = p.parse(q)
	retVal = p.stack[0].(Expr)
	log.Printf("%v", p.stack)
	return
}

type parser struct {
	stack      []substitutable
	infixStack []int // quick hack to allow for parsing of infix operators. The integers point to the existing infix structures in stack
	buf        strings.Builder
}

func (p *parser) pop() substitutable {
	if len(p.stack) == 0 {
		panic("cannot pop")
	}

	retVal := p.stack[len(p.stack)-1]
	p.stack = p.stack[:len(p.stack)-1]
	return retVal
}

func (p *parser) push(a substitutable) {
	p.stack = append(p.stack, a)
}

func (p *parser) popExpr() (Expr, error) {
	s := p.pop()
	e, ok := s.(Expr)
	if !ok {
		return nil, errors.Errorf("Expected an Expr. Got %v of %v instead", s, s)
	}
	return e, nil
}
func (p *parser) logstack(buf io.Writer) {
	var uselog bool
	if buf == nil {
		buf = new(bytes.Buffer)
		uselog = true
	}
	for i, item := range p.stack {
		fmt.Fprintf(buf, "%d: %v\n", i, item)
	}
	if uselog {
		log.Println(buf.(*bytes.Buffer).String())
	}
}

func (p *parser) parse(q []tok) (err error) {
	for i := 0; i < len(q); i++ {
		t := q[i]
		switch t.t {
		case parenL:
			p.push(Abstract{})
		case parenR:
			p.resolveA()
		case brackL:
			p.push(Sli{})
		case brackR:
			p.resolveSlice()
		case braceL:
			p.push(Compound{})
		case braceR:
			p.resolveCompound()
		case pipe:
			if err := p.resolveTop(); err != nil {
				return errors.Wrapf(err, "Parse Error: Unable to parse %c (Location %d).", t.v, t.l)
			}
			p.push(SubjectTo{})
		case digit:
			p.push(Size(int(t.v))) // we'll treat all intlike things to be Size in the context of the parser.

			log.Printf("digit %v infix stack %v", t.v, p.infixStack)
			p.logstack(nil)
			if len(p.infixStack) > 0 && len(p.stack)-2 == p.infixStack[len(p.infixStack)-1] {
				if err := p.resolveTop(); err != nil {
					return errors.Wrapf(err, "Failed to parse %v after parsing %d", p.stack[p.infixStack[len(p.infixStack)-1]], t)
				}
				p.infixStack = p.infixStack[:len(p.infixStack)-1]
			}
		case letter:
			p.push(Var(t.v))

			log.Printf("leter %c infix stack %v | %d", t.v, p.infixStack, len(p.stack))
			p.logstack(nil)
			if len(p.infixStack) > 0 && len(p.stack)-2 == p.infixStack[len(p.infixStack)-1] {
				if err := p.resolveTop(); err != nil {
					return errors.Wrapf(err, "Failed to parse %v after parsing %d", p.stack[p.infixStack[len(p.infixStack)-1]], t)
				}
				p.infixStack = p.infixStack[:len(p.infixStack)-1]
			}
		case arrow:
			e, err := p.popExpr()
			if err != nil {
				return errors.Wrap(err, "Failed to parse Arrow. Left expression error:")
			}
			p.push(Arrow{A: e})
			p.infixStack = append(p.infixStack, len(p.stack)-1)
		case unop:
			e, err := p.popExpr()
			if err != nil {
				return errors.Wrapf(err, "Failed to parse Unary Operation %c. Left expression error:", t.v)
			}
			o, err := parseOpType(t.v)
			if err != nil {
				return errors.Wrapf(err, "Failed to parse Unary Operator %c.", t.v)
			}
			p.push(UnaryOp{Op: o, A: e})
			p.infixStack = append(p.infixStack, len(p.stack)-1)
		case binop:
			e, err := p.popExpr()
			if err != nil {
				return errors.Wrapf(err, "Failed to parse Binary Operation %c. Left expression error:", t.v)
			}
			o, err := parseOpType(t.v)
			if err != nil {
				return errors.Wrapf(err, "Failed to parse Binary Operator %c.", t.v)
			}
			p.push(BinOp{Op: o, A: e})
			p.infixStack = append(p.infixStack, len(p.stack)-1)
		case logop:
			e, err := p.popExpr()
			if err != nil {
				return errors.Wrapf(err, "Failed to parse Logical Operation %c. Left expression error:", t.v)
			}
			o, err := parseOpType(t.v)
			if err != nil {
				return errors.Wrapf(err, "Failed to parse Logical Operator %c.", t.v)
			}
			p.push(BinOp{Op: o, A: e})
			p.infixStack = append(p.infixStack, len(p.stack)-1)

		}
	}
	for i := len(p.infixStack) - 1; i >= 0; i-- {
		if err := p.resolveTop(); err != nil {
			return errors.Wrap(err, "Unable to resolve top")
		}
	}

	return nil
}

func (p *parser) resolveA() error {
	var fw Abstract
	var bw []substitutable
	var ok bool
	// repeatedly pop stuff off the stack
	for i := len(p.stack) - 1; i >= 0; i-- {
		s := p.pop()
		if fw, ok = s.(Abstract); ok {
			break
		}
		bw = append(bw, s)
	}
	if fw == nil {
		return errors.Errorf("Popped every item of the stack, found no Abstract")
	}

	for i := len(bw) - 1; i >= 0; i-- {
		s := bw[i]
		switch st := s.(type) {
		case Size:
			fw = append(fw, st)
		case Var:
			fw = append(fw, st)
		case UnaryOp:
			fw = append(fw, st)
		case BinOp:
			fw = append(fw, st)
		default:
			return errors.Errorf("Unable to parse %v of %T as a Sizelike", s, s)
		}
	}

	shp, ok := fw.ToShape()
	if ok {
		p.push(shp)
		return nil
	}
	p.push(fw)
	return nil

}

// resolveCompound expects the stack to look like this:
// 	[..., Compound{}, Expr, SubjectTo{...}]
// The result will look like this
// 	[..., Compound{...}] (the Compound{} now has data)
func (p *parser) resolveCompound() error {
	// first check
	var st SubjectTo
	var e Expr
	var c Compound
	var ok bool

	top := p.pop() // SubjectTo
	if st, ok = top.(SubjectTo); !ok {
		return errors.Errorf("Expected Top of Stack to be a SubjectTo is %v of %T. Stack: %v", top, top, p.stack)
	}
	snd := p.pop() // Expr
	if e, ok = snd.(Expr); !ok {
		return errors.Errorf("Expected Second of Stack to be a Expr is %v of %T. Stack: %v", snd, snd, p.stack)
	}
	thd := p.pop() // Compound{} (should be empty)
	if c, ok = thd.(Compound); !ok {
		return errors.Errorf("Expected Third of Stack to be a Compound is %v of %T. Stack: %v", thd, thd, p.stack)
	}
	c.Expr = e
	c.SubjectTo = st
	p.push(c)
	return nil
}

func (p *parser) resolveSlice() error {
	// three cases:
	// 1. single slice
	//	- look at top 2
	// 2. range
	//	- look at top 3
	// 3. range + step
	//	- look at top 4

	top := p.pop()
	snd := p.pop()

	topN, ok := substToInt(top)
	if !ok {
		return errors.Errorf("Expected the top to be a Size. Got %v of %T instead", top, top)
	}
	if s, ok := snd.(Sli); ok {
		// case 1
		s.start = topN
		s.end = topN + 1
		s.step = 1
		p.push(s)
		return nil
	}

	thd := p.pop()
	if s, ok := thd.(Sli); ok {
		// case 2
		sndN, ok := substToInt(snd)
		if !ok {
			return errors.Errorf("Expected the second of stack to be an intlike. Got %v of %T instead", snd, snd)
		}
		s.start = topN
		s.end = sndN
		s.step = 1
		p.push(s)
		return nil
	}

	fth := p.pop()
	if s, ok := fth.(Sli); ok {
		// case 3
		sndN, ok := substToInt(snd)
		if !ok {
			return errors.Errorf("Expected the second of stack to be an intlike. Got %v of %T instead", snd, snd)
		}

		thdN, ok := substToInt(thd)
		if !ok {
			return errors.Errorf("Expected the third of stack to be an intlike. Got %v of %T instead", thd, thd)
		}

		s.start = topN
		s.end = sndN
		s.step = thdN
		p.push(s)
		return nil
	}
	panic(fmt.Sprintf("Unreachable case. Got top %v of %T;\nSecond %v of %T;\nThird: %v of %v;\nFourth: %v of %T", top, top, snd, snd, thd, thd, fth, fth))
}

func (p *parser) resolveTop() (err error) {
	top := p.pop()
	snd := p.pop()
	var ok bool
	switch s := snd.(type) {
	case Compound:
		if s.Expr == nil {
			if s.Expr, ok = top.(Expr); !ok {
				return errors.Errorf("Expected top of the stack to be an expression. Got %v of %T instead", top, top)
			}
			p.push(s)
			return nil
		}
		if s.SubjectTo, ok = top.(SubjectTo); !ok {
			return errors.Errorf("Expected top of the stack to be a SubjectTo. Got %v of %T instead", top, top)
		}
		p.push(s)
		return nil
	case UnaryOp:
		if s.A, ok = top.(Expr); !ok {
			return errors.Errorf("Expected top of the stack to be an expression. Got %v of %T instead", top, top)
		}
		p.push(s)
		return nil
	case BinOp:
		if s.B, ok = top.(Expr); !ok {
			return errors.Errorf("Expected top of the stack to be an expression. Got %v of %T instead", top, top)
		}
		p.push(s)
		return nil
	case Arrow:
		if s.B, ok = top.(Expr); !ok {
			return errors.Errorf("Expected top of the stack to be an expression. Got %v of %T instead", top, top)
		}
		p.push(s)
		return nil
	default:
		p.push(snd)
		p.push(top)
		log.Printf("top: %v of %v | snd %v of %T | stack %v", top, top, snd, snd, p.stack)
		return nil

	}
}

type tokentype int

const (
	ws tokentype = iota
	parenL
	parenR
	brackL
	brackR
	braceL
	braceR
	digit
	letter
	comma
	arrow
	colon
	pipe
	unop
	binop
	cmpop
	logop
)

type tok struct {
	t tokentype // type
	v rune      // value of the token
	l int       // location
}

const eoserr = "Lex Error: sudden end of string found in %q. Position %d"

// lex is a function that takes a string and returns a slice of tokens.
//
// it's fundamentally a giant state table in a for loop. Since the grammar of the language is very strict, it is a fairly straight forwards parse.
func lex(a string) (retVal []tok, err error) {
	rs := []rune(a)
	for i := 0; i < len(rs); i++ {
		r := rs[i]
		switch {
		case r == '(':
			retVal = append(retVal, tok{parenL, r, i})
		case r == ')':
			retVal = append(retVal, tok{parenR, r, i})
		case r == '[':
			retVal = append(retVal, tok{brackL, r, i})
		case r == ']':
			retVal = append(retVal, tok{brackR, r, i})
		case r == '{':
			retVal = append(retVal, tok{braceL, r, i})
		case r == '}':
			retVal = append(retVal, tok{braceR, r, i})
		case r == '→':
			retVal = append(retVal, tok{arrow, r, i})
		case r == '-':
			if i+1 >= len(rs) {
				return nil, errors.Errorf(eoserr, a, i)
			}
			if rs[i+1] == '>' {
				i++
				retVal = append(retVal, tok{arrow, '→', i})
				continue
			}
			retVal = append(retVal, tok{binop, r, i})
		case r == '+', r == '*', r == '×', r == '/', r == '∕', r == '÷':
			rr := r
			if r == '*' {
				rr = '×'
			}
			if r == '/' || r == '∕' {
				rr = '÷'
			}
			retVal = append(retVal, tok{binop, rr, i})
		case r == '=', r == '≠', r == '≥', r == '≤', r == '≥', r == '≤': // single symbol cmp op (note there are TWO acceptable unicode symbols for gte and lte)
			retVal = append(retVal, tok{cmpop, r, i})
		case r == '!':
			if i+1 >= len(rs) {
				return nil, errors.Errorf(eoserr, a, i)
			}
			if rs[i+1] == '=' {
				i++
				retVal = append(retVal, tok{cmpop, '≠', i})
			}
		case r == '>', r == '<':
			if i+1 >= len(rs) {
				return nil, errors.Errorf(eoserr, a, i)
			}
			if rs[i+1] == '=' {
				i++
				var rr rune
				switch r {
				case '>':
					rr = '≥'
				case '<':
					rr = '≤'
				}
				retVal = append(retVal, tok{cmpop, rr, i})
				continue
			}
			retVal = append(retVal, tok{cmpop, r, i})
		case r == '∧', r == '∨', r == '⋀', r == '⋁': // single symbol logical op
			retVal = append(retVal, tok{logop, r, i})
		case r == '&':
			if i+1 >= len(rs) {
				return nil, errors.Errorf(eoserr, a, i)
			}
			if rs[i+1] == '&' { // for people who think AND is written "&&"
				i++
				retVal = append(retVal, tok{logop, '∧', i})
				continue
			}
		case r == '|':
			if i+1 >= len(rs) {
				return nil, errors.Errorf(eoserr, a, i)
			}
			if rs[i+1] == '|' { // for people who think OR is written "||"
				i++
				retVal = append(retVal, tok{logop, '∨', i})
				continue
			}
			retVal = append(retVal, tok{pipe, r, i})
		case r == 'Π', r == 'Σ', r == '∀', r == 'D', r == 'P', r == 'S':
			rr := r
			if r == 'P' {
				rr = 'Π'
			}
			if r == 'S' {
				rr = 'Σ'
			}
			retVal = append(retVal, tok{unop, rr, i})
		case r == ':', r == ',', unicode.IsSpace(r):
			continue // we ignore spaces and delimiters
		case unicode.IsDigit(r):
			var rs2 = []rune{r}
			for j := i + 1; j < len(rs); j++ {
				r2 := rs[j]
				if !unicode.IsDigit(r2) {
					i = j - 1
					break
				}
				rs2 = append(rs2, r2)
			}
			s := string(rs2)
			num, err := strconv.Atoi(s)
			if err != nil {
				return nil, errors.Wrapf(err, "Unable to parse %q as a number", s)
			}

			retVal = append(retVal, tok{digit, rune(int32(num)), i})
		case unicode.IsLetter(r):
			retVal = append(retVal, tok{letter, r, i})
			// check if next is a letter. if it is  then error out
			if i+1 >= len(rs) {
				return retVal, nil
			}
			if unicode.IsLetter(rs[i+1]) {
				return nil, errors.Errorf("Only single letters are allowed as variables.")
			}

		}

	}
	return retVal, nil
}
