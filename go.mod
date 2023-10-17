module gorgonia.org/gorgonia

go 1.20

replace (
	gorgonia.org/dtype => ../dtype
	gorgonia.org/shapes => ../shapes
	gorgonia.org/tensor => ../tensor
)

require (
	github.com/chewxy/hm v1.0.0
	github.com/chewxy/math32 v1.10.1
	github.com/go-gota/gota v0.10.1
	github.com/pkg/errors v0.9.1
	github.com/stretchr/testify v1.8.3
	github.com/xtgo/set v1.0.0
	gonum.org/v1/gonum v0.13.0
	gonum.org/v1/netlib v0.0.0-20200317120129-c5a04cffd98a
	gopkg.in/cheggaaa/pb.v1 v1.0.27
	gorgonia.org/cu v0.9.2
	gorgonia.org/dawson v1.2.0
	gorgonia.org/dtype v0.10.0
	gorgonia.org/internal v0.0.0-20210804133310-438f7f1f5027
	gorgonia.org/shapes v0.0.0-20220805023001-db33330e8e09
	gorgonia.org/tensor v0.9.20
)

require (
	github.com/davecgh/go-spew v1.1.1 // indirect
	github.com/fatih/color v1.7.0 // indirect
	github.com/google/gofuzz v1.2.0 // indirect
	github.com/mattn/go-colorable v0.1.4 // indirect
	github.com/mattn/go-runewidth v0.0.4 // indirect
	github.com/pmezard/go-difflib v1.0.0 // indirect
	golang.org/x/exp v0.0.0-20230522175609-2e198f4a06a1 // indirect
	golang.org/x/sys v0.6.0 // indirect
	google.golang.org/protobuf v1.30.0 // indirect
	gopkg.in/check.v1 v1.0.0-20180628173108-788fd7840127 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
	gorgonia.org/vecf32 v0.9.0 // indirect
	gorgonia.org/vecf64 v0.9.0 // indirect
)
