package shapes

import (
	"bytes"
	"fmt"
	"io"
	lg "log"
	"strconv"
	"strings"
	"unicode"

	log "github.com/chewxy/InkHuffer"
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
	return
}

type parser struct {
	stack      []substitutable
	infixStack []tok

	buf strings.Builder
}

func (p *parser) pop() substitutable {
	if len(p.stack) == 0 {
		panic("cannot pop")
	}

	retVal := p.stack[len(p.stack)-1]
	p.stack = p.stack[:len(p.stack)-1]
	return retVal
}

func (p *parser) popExpr() (Expr, error) {
	s := p.pop()
	e, ok := s.(Expr)
	if !ok {
		return nil, errors.Errorf("Expected an Expr. Got %v of %v instead", s, s)
	}
	return e, nil
}

func (p *parser) push(a substitutable) {
	p.stack = append(p.stack, a)
}

func (p *parser) pushInfix(t tok) {
	p.infixStack = append(p.infixStack, t)
}

func (p *parser) popInfix() tok {
	if len(p.infixStack) == 0 {
		panic("Cannot pop infix stack")
	}
	retVal := p.infixStack[len(p.infixStack)-1]
	p.infixStack = p.infixStack[:len(p.infixStack)-1]
	return retVal
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
		lg.Printf("%v", buf.(*bytes.Buffer).String())
	}
}

func (p *parser) loginfixStack(buf io.Writer) {
	var uselog bool
	if buf == nil {
		buf = new(bytes.Buffer)
		uselog = true
	}
	for i, item := range p.infixStack {
		fmt.Fprintf(buf, "%d: {%v %c %d}\n", i, item.t, item.v, item.l)
	}
	if uselog {
		lg.Printf("%v", buf.(*bytes.Buffer).String())
	}
}

func (p *parser) parse(q []tok) (err error) {
	for i := 0; i < len(q); i++ {
		t := q[i]
		switch t.t {
		case parenL:
			log.Logf("ParenL\nStack\n")
			p.logstack(nil)
			log.Logf("InfixStack\n")
			p.loginfixStack(nil)

			// special case to handle ()
			t2 := q[i+1]
			if t2.t == parenR {
				p.push(Shape{})
				i++
				continue
			}
			p.push(Abstract{})

		case parenR:
			log.Logf("parenR")
			// before we close any ')', we need to resolve the last infix if there are any.
			// Now, because of this, the special case of () has to be handled elsewhere
			p.logstack(nil)
			p.loginfixStack(nil)
			log.EnterScope()
			for len(p.infixStack) > 0 {
				log.Logf("len stack %v", len(p.stack))
				p.logstack(nil)
				log.Logf("infix stack ")
				p.loginfixStack(nil)
				if err = p.condResolveInfix(t); err != nil {
					_, ok := err.(noop)
					if ok {
						break
					}
					return errors.Wrapf(err, "Unable to resolve the last infix before resolving a Shape at %d", t.l)
				}
			}
			log.LeaveScope()
			if err = p.resolveA(); err != nil {
				return errors.Wrapf(err, "Unable to resolve an abstract shape at %d", t.l)
			}
		case brackL:
			p.push(Sli{})
		case brackR:
			// before we close any ']', we need to resolve the last infix if there are any.
			if err = p.maybeResolveInfix(); err != nil {
				return errors.Wrapf(err, "Unable to resolve the last infix before resolving a slice at %d", t.l)
			}
			if err = p.resolveSlice(); err != nil {
				return errors.Wrapf(err, "Unable to resolve slice at %d", t.l)
			}
		case braceL:
			p.push(Compound{})
		case braceR:
			// before we clsoe any '}', we need to resolve the last infix if there are any.
			if err = p.maybeResolveInfix(); err != nil {
				return errors.Wrapf(err, "Unable to resolve the last infix before resolving a Compound at %d", t.l)
			}
			if err = p.resolveCompound(); err != nil {
				return errors.Wrapf(err, "Unable to resolve Compound at %d", t.l)
			}
		case pipe:
			log.Logf("Pipe:")
			p.logstack(nil)
			p.loginfixStack(nil)
			// before we close any expression with a '|' to go into the clauses, we need to resolve all the infixes.
			for len(p.infixStack) > 0 {
				if err = p.resolveInfix(); err != nil {
					return errors.Wrapf(err, "Unable to resolve the last infix before resolving a slice at %d", t.l)
				}
			}
		case digit:
			p.logstack(nil)
			p.push(Size(int(t.v))) // we'll treat all intlike things to be Size in the context of the parser.
			p.logstack(nil)
		case letter:
			p.logstack(nil)
			p.push(Var(t.v))
			p.logstack(nil)
		case comma:
			p.logstack(nil)
			p.loginfixStack(nil)
			if err := p.maybeResolveInfix(); err != nil {
				return errors.Wrapf(err, "Cannot resolve all the infixes in a shape at %d", t.l)
			}
		case arrow:
			p.logstack(nil)
			p.condResolveInfix(t)
			p.pushInfix(t)
		case unop:
			log.Logf("Unop %c", t.v)
			p.logstack(nil)
			p.condResolveInfix(t)
			p.pushInfix(t)
			p.logstack(nil)
			p.loginfixStack(nil)
			log.Logf("----")
		case binop:
			p.condResolveInfix(t)
			p.pushInfix(t)
		case cmpop:
			p.condResolveInfix(t)
			p.pushInfix(t)
		case logop:
			p.condResolveInfix(t)
			p.pushInfix(t)
		default:
			panic(fmt.Sprintf("%v with type %v unhandled", t, t.t))

		}
	}
	log.Logf("AT THE END")
	for len(p.infixStack) > 0 {
		if err := p.resolveInfix(); err != nil {
			return errors.Wrap(err, "Unable to resolve final infixes")
		}
	}
	return nil
}

func (p *parser) maybeResolveInfix() error {
	log.Logf("Maybe Resolve Infix")
	log.EnterScope()
	defer log.LeaveScope()
	if len(p.infixStack) > 0 {
		return p.resolveInfix()
	}
	return nil
}

// condResolveInfix conditionally resolves previous infixes given a "current" token. This uses operator precedence (which is not coded)
func (p *parser) condResolveInfix(cur tok) error {
	if len(p.infixStack) == 0 {
		return nil
	}
	last := p.infixStack[len(p.infixStack)-1] // peek
	switch cur.t {
	case arrow:
		// if the current token is an arrow, check previous infixes. Only fix them if they are not arrow, because arrow has the lowest operator precedence.

		switch last.t {
		case arrow:
			// do nothing
			return nil
		default:
			return p.resolveInfix()
		}

	case binop:
		switch last.t {
		case unop:
			return p.resolveInfix()
		case binop:
			// check for the operator precedence of other binop TODO
			panic("NYI")
		default:
			return nil
		}

	case unop:
		switch last.t {
		case unop:
			return p.resolveInfix()
		default:
			return nil
		}
	case cmpop:
		switch last.t {
		case unop:
			return p.resolveInfix()
		case cmpop:
			return p.resolveInfix()
		default:
			return nil
		}
	case logop:
		switch last.t {
		case unop:
			return p.resolveInfix()
		case binop:
			return p.resolveInfix()
		case cmpop:
			return p.resolveInfix()
		case logop:
			return p.resolveInfix()
		default:
			return nil
		}
	case parenR:
		switch last.t {
		case unop:
			return p.resolveInfix()
		case binop:
			return p.resolveInfix()
		case cmpop, logop:
			return p.resolveInfix()
		case arrow:
			return noop{}
		}
		p.logstack(nil)
		panic(fmt.Sprintf("parenR. Last %c of %v", last.v, last.t))
	default:
		panic(fmt.Sprintf("tok %v of %v is unsupported", cur, cur.t))

	}
	panic("Unreachable")
}

func (p *parser) resolveInfix() error {
	log.Logf("Resolve Infix")
	log.EnterScope()
	defer log.LeaveScope()

	last := p.popInfix()
	var err error
	switch last.t {
	case unop:
		if err = p.resolveUnOp(last); err != nil {
			return errors.Wrapf(err, "Unable to resolve unop %v.", last)
		}
	case binop:
		if err = p.resolveBinOp(last); err != nil {
			return errors.Wrapf(err, "Unable to resolve binop %v.", last)
		}
	case cmpop:
		if err = p.resolveCmpOp(last); err != nil {
			return errors.Wrapf(err, "Unable to resolve cmpop %v.", last)
		}
	case logop:
		if err = p.resolveLogOp(last); err != nil {
			return errors.Wrapf(err, "Unable to resolve logop %v.", last)
		}

	case arrow:
		if err := p.resolveArrow(last); err != nil {
			return errors.Wrapf(err, "Cannot resolve arrow %v.", last)
		}
	}

	return nil
}

func (p *parser) resolveA() error {
	log.Logf("ResolveA")
	log.EnterScope()
	p.logstack(nil)
	defer log.LeaveScope()

	var abs Abstract
	var ok bool
	var bw []substitutable

	// repeatedly pop stuff off the stack
	for i := len(p.stack) - 1; i >= 0; i-- {
		s := p.pop()
		if abs, ok = s.(Abstract); ok && len(abs) == 0 {
			break
		}
		bw = append(bw, s)
	}
	if abs == nil {
		return errors.Errorf("Popped every item of the stack, found no Abstract")
	}
	var allSizelike bool = true
	for _, v := range bw {
		if _, ok = v.(Sizelike); !ok {
			allSizelike = false
		}
	}
	log.Logf("all sizelike %v", allSizelike)
	log.Logf("bw %v", bw)
	if allSizelike {
		for i := len(bw) - 1; i >= 0; i-- {
			s := bw[i].(Sizelike)
			abs = append(abs, s)
		}
		shp, ok := abs.ToShape()
		if ok {
			p.push(shp)
			return nil
		}
		p.push(abs)
		return nil
	}
	p.push(bw[0])
	log.Logf("...")
	p.logstack(nil)
	return nil
}

func (p *parser) resolveArrow(t tok) error {
	snd, err := p.popExpr()
	if err != nil {
		return errors.Wrapf(err, "Cannot resolve snd of arrow at %d as Expr.", t.l)
	}
	fst, err := p.popExpr()
	if err != nil {
		return errors.Wrapf(err, "Cannot resolve fst of arrow at %d as Expr.", t.l)
	}
	arr := Arrow{
		A: fst,
		B: snd,
	}
	p.push(arr)
	return nil
}

func (p *parser) resolveUnOp(t tok) error {
	log.Logf("Resolve UnaryOp")
	log.EnterScope()
	defer log.LeaveScope()
	expr, err := p.popExpr()
	if err != nil {
		return errors.Wrapf(err, "Cannot resolve expr of unop %c at %d.", t.v, t.l)
	}
	op, err := parseOpType(t.v)
	if err != nil {
		return errors.Wrapf(err, "Unable to parse UnOp OpType %v", t)
	}
	o := UnaryOp{
		Op: op,
		A:  expr,
	}
	log.Logf("UnaryOP to be pushed %v", o)
	p.push(o)
	return nil
}

func (p *parser) resolveBinOp(t tok) error {
	snd, err := p.popExpr()
	if err != nil {
		return errors.Wrapf(err, "Unable to resolve snd of BinOp at %d as Expr.", t.l)
	}
	fst, err := p.popExpr()
	if err != nil {
		return errors.Wrapf(err, "Unable to resolve fst of BinOp at %d as Expr.", t.l)
	}

	op, err := parseOpType(t.v)
	if err != nil {
		return errors.Wrapf(err, "Unable to parse BinOp OpType %v", t)
	}

	o := BinOp{
		Op: op,
		A:  fst,
		B:  snd,
	}
	p.push(o)
	return nil
}

func (p *parser) resolveCmpOp(t tok) error {
	snd := p.pop()
	sndOp, ok := snd.(Operation)
	if !ok {
		return errors.Errorf("Cannot resolve snd of CmpOp %c at %d as Operation. Got %T instead ", t.v, t.l, snd)
	}
	fst := p.pop()
	fstOp, ok := fst.(Operation)
	if !ok {
		return errors.Errorf("Cannot resolve fst of CmpOp %c at %d as Operation.", t.v, t.l)
	}

	op, err := parseOpType(t.v)
	if err != nil {
		return errors.Wrapf(err, "Unable to parse CmpOp OpType %v", t)
	}

	o := SubjectTo{
		OpType: op,
		A:      fstOp,
		B:      sndOp,
	}
	p.push(o)
	return nil

}

func (p *parser) resolveLogOp(t tok) error {
	snd := p.pop()
	sndOp, ok := snd.(Operation)
	if !ok {
		return errors.Errorf("Cannot resolve snd of LogOp %c at %d as Operation. Got %T instead ", t.v, t.l, snd)
	}
	fst := p.pop()
	fstOp, ok := fst.(Operation)
	if !ok {
		return errors.Errorf("Cannot resolve fst of LogOp %c at %d as Operation. Got %v of %T instead", t.v, t.l, fst, fst)
	}

	op, err := parseOpType(t.v)
	if err != nil {
		return errors.Wrapf(err, "Unable to parse LogOp OpType %v", t)
	}

	o := SubjectTo{
		OpType: op,
		A:      fstOp,
		B:      sndOp,
	}
	p.push(o)
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
		case r == ',':
			retVal = append(retVal, tok{comma, r, i})
		case r == ':', unicode.IsSpace(r):
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
