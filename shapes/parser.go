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

	p := newParser(true)
	defer func() {
		if r := recover(); r != nil {
			p.printTab(nil)
			panic(r)
		}
	}()
	err = p.parse(q)
	p.logstate()
	if len(p.stack) <= 0 {
		p.printTab(nil)
		return nil, errors.Errorf("WTF?")
	}

	p.printTab(nil)
	retVal = p.stack[0].(Expr)
	return
}

type parser struct {
	queue      []tok           // incoming string of tokens
	stack      []substitutable // "working" stack
	infixStack []tok           // stack of operators
	qptr       int             // queue pointer

	buf strings.Builder

	log *bytes.Buffer
}

func newParser(log bool) *parser {
	var l *bytes.Buffer
	if log {
		l = new(bytes.Buffer)
	}
	return &parser{
		log: l,
	}
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

func (p *parser) cur() (tok, error) {
	if p.qptr < 0 || p.qptr >= len(p.queue) {
		return tok{}, errors.Errorf("Cannot get current token. Pointer: %d. Queue: %v", p.qptr, len(p.queue))
	}
	return p.queue[p.qptr], nil
}

func (p *parser) compose(g, f func() error, fname, gname string) func() error {
	return func() error {
		if err := f(); err != nil {
			return errors.Wrapf(err, "In composed functions %v ∘ %v. %v failed", fname, gname, fname)
		}
		if err := g(); err != nil {
			return errors.Wrapf(err, "In composed functions %v ∘ %v. %v failed", fname, gname, gname)
		}
		return nil
	}
}

// compareCur compares the cur token with  the top of the infix stack. It returns a function that the parser should take
func (p *parser) compareCur() func() error {
	p.logstate()
	t, err := p.cur()
	if err != nil {
		return func() error { return errors.Wrap(err, "Cannot compareCur()") }
	}
	log.Printf("CUR %v", t)
	switch t.t {
	case digit:
		return p.pushNum
	case letter:
		return p.pushVar
	default:
		if len(p.infixStack) == 0 {
			// push the current to infixStack
			return p.pushCurTok
		}
		// check if current token has greater op prec than top of stack
		top := p.infixStack[len(p.infixStack)-1]
		topPrec := opprec[top.v]
		curPrec := opprec[t.v]
		log.Printf("cur %q top %q || %d %d", t, top, curPrec, topPrec)

		// if current is negative, we need to resolve until the infixStack has len 0 then pushCurTok
		if curPrec < 0 {
			if err := p.pushCurTok(); err != nil {
				return func() error { return errors.Wrap(err, "curPrec < 0") }
			}
			return p.resolveAllInfix
		}

		if curPrec > topPrec {
			return p.pushCurTok
		}

		// check special case of arrows (which are right assoc)
		if top.t == arrow && t.t == arrow {
			return p.pushCurTok
		}

		// otherwise resolve first then pushcurtok
		return p.compose(p.pushCurTok, p.resolveInfix, "resolveInfix", "pushCurTok")
	}
}

func (p *parser) incrQPtr() error { p.qptr++; return nil }

// pushVar pushes a var on to the values stack.
func (p *parser) pushVar() error {
	t, err := p.cur()
	if err != nil {
		return err
	}
	p.push(Var(t.v))
	return nil
}

// pushNum pushes a number (typed as a Size) onto the values stack.
func (p *parser) pushNum() error {
	t, err := p.cur()
	if err != nil {
		return err
	}
	p.push(Size(int(t.v)))
	return nil
}

// pushCurTok pushes the current token into the infixStack
func (p *parser) pushCurTok() error {
	t, err := p.cur()
	if err != nil {
		return err
	}
	p.pushInfix(t)
	return nil
}

func (p *parser) parse(q []tok) (err error) {
	p.queue = q
	for p.qptr < len(p.queue) {
		if err = p.parseOne(); err != nil {
			p.printTab(nil)
			return err
		}
		p.incrQPtr()
	}
	if len(p.infixStack) > 0 {
		if err := p.resolveAllInfix(); err != nil {
			return errors.Wrap(err, "Unable to resolve all infixes while in .parse")
		}
	}
	log.Printf("q %v", q)
	p.printTab(nil)
	return nil
}

func (p *parser) parseOne() error {
	t, err := p.cur()
	if err != nil {
		return errors.Wrap(err, "Unable to parseOne")
	}

	// special cases: ()
	if t.v == '(' {
		if p.qptr == len(p.queue)-1 {
			return errors.Errorf("Dangling open paren '(' at %v", t.l)
		}
		if p.queue[p.qptr+1].v == ')' {
			p.push(Shape{})
			p.incrQPtr()
			return nil
		}
	}

	fn := p.compareCur()
	if err := fn(); err != nil {
		return errors.Wrap(err, "Unable to parseOne")
	}
	return nil
}

// resolveAllInfix resolves all the infixes in the infixStack. The result will be an empty infix stack.
func (p *parser) resolveAllInfix() error {
	var count int
	for len(p.infixStack) > 0 {
		if err := p.resolveInfix(); err != nil {
			return errors.Wrapf(err, "Unable to resolve all infixes. %d processed.", count)
		}
		count++
	}
	return nil
}

func (p *parser) resolveInfix() error {
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
	case parenR:
		if err := p.resolveA(); err != nil {
			return errors.Wrap(err, "Cannot resolve A.")
		}
	case comma:
		if err := p.resolveComma(last); err != nil {
			return errors.Wrapf(err, "Cannot resolve comma %v", last)
		}
	case braceR:
		if err := p.resolveCompound(); err != nil {
			return errors.Wrapf(err, "Cannot resolve compound %v", last)
		}
	default:
		log.Printf("last {%v %c %v} is unhandled", last.t, last.v, last.l)
	}

	return nil
}

func (p *parser) resolveA() error {
	var bw []tok
	for i := len(p.infixStack) - 1; i >= 0; i-- {
		t := p.popInfix()
		// keep going until you find the first '('
		if t.v == '(' {
			break
		}
		bw = append(bw, t)
	}
	reverse(bw)
	backup := p.infixStack
	p.infixStack = bw

	if err := p.resolveAllInfix(); err != nil {
		return errors.Wrap(err, "Unable to resolveA")
	}
	if len(p.infixStack) > 0 {
		// do something
	}
	p.infixStack = backup

	last := p.pop()
	if abs, ok := last.(Abstract); ok {
		if shp, ok := abs.ToShape(); ok {
			p.push(shp)
			return nil
		}
	}
	p.push(last)
	return nil
}

func (p *parser) resolveArrow(t tok) error {
	p.logstate("resolveArrow")
	defer p.logstate("resolveArrow END")
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

func (p *parser) resolveComma(t tok) error {
	// comma is a binary option. However if it's a trailing comma, then there is no need.
	snd := p.pop()

	if len(p.stack) == 0 {
		switch s := snd.(type) {
		case Abstract:
			p.push(s)
		case Sizelike:
			p.push(Abstract{s})
		}
		return nil
	}

	fst := p.pop()
	log.Printf("resolveComma %v. fst %v snd %v", t, fst, snd)
	switch f := fst.(type) {
	case Abstract:
		switch s := snd.(type) {
		case Sizelike:
			f = append(f, s)
			p.push(f)
			return nil
		case Conser:
			ret := f.Cons(s).(substitutable)
			p.push(ret)
			return nil
		}
	case Sizelike:
		switch s := snd.(type) {
		case Sizelike:
			p.push(Abstract{f, s})
			return nil
		case Shape:
			abs := Abstract{f}
			ret := abs.Cons(s).(substitutable)
			p.push(ret)
			return nil
		case Abstract:
			s = append(s, f)
			copy(s[1:], s[0:])
			s[0] = f
			p.push(s)
			return nil
		}
	}
	panic("Unreachable")
}

func (p *parser) resolveUnOp(t tok) error {
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
		log.Printf("p.stack %v", p.stack)
		log.Printf("snd %v, fst %v", snd, fst)

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
	var ok bool

	top := p.pop() // SubjectTo
	if st, ok = top.(SubjectTo); !ok {
		return errors.Errorf("Expected Top of Stack to be a SubjectTo is %v of %T. Stack: %v", top, top, p.stack)
	}
	snd := p.pop() // Expr
	if e, ok = snd.(Expr); !ok {
		return errors.Errorf("Expected Second of Stack to be a Expr is %v of %T. Stack: %v", snd, snd, p.stack)
	}
	c := Compound{
		Expr:      e,
		SubjectTo: st,
	}

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

// operator precedence table
var opprec = map[rune]int{
	'(': 80,
	')': -1,
	'[': 10,
	']': 10,
	'{': 70,
	'}': -1,
	',': 2,
	':': 1,
	'|': -1,
	'→': 0,

	// unop
	'K': 60,
	'D': 60,
	'Π': 60,
	'Σ': 60,
	'∀': 60,

	// binop
	'+': 40,
	'-': 40,
	'×': 50,
	'÷': 50,

	// cmpop
	'=': 30,
	'≠': 30,
	'<': 30,
	'>': 30,
	'≤': 30,
	'≥': 30,

	// logop
	'∧': 20,
	'∨': 10,
}

type tokentype int

const (
	eos tokentype = iota
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

func (t tok) Format(s fmt.State, c rune) {
	switch t.t {
	case letter:
		fmt.Fprintf(s, "{%c %d}", t.v, t.l)
	case digit:
		fmt.Fprintf(s, "{%d %d}", t.v, t.l)
	case eos:
		fmt.Fprintf(s, "{EOS %d}", t.l)
	default:
		fmt.Fprintf(s, "{%c %d}", t.v, t.l)
	}
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
