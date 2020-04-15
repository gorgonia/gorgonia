package gorgonnx

import (
	"fmt"
	"os"
	"sort"
	"testing"

	"github.com/owulveryck/onnx-go/backend/testbackend"
	_ "github.com/owulveryck/onnx-go/backend/testbackend/onnx"
	"github.com/owulveryck/onnx-go/backend/testbackend/testreport"
)

type report struct {
	info    string
	failed  bool
	skipped bool
}

// TestONNX run the onnx's backend testConstuctors against all registered operatos
func TestONNX(t *testing.T) {
	var testConstructors []func() *testbackend.TestCase
	if testing.Short() {
		for optype := range operators {
			testConstructors = append(testConstructors, testbackend.GetOpTypeTests(optype)...)
		}
	} else {
		testConstructors = testbackend.GetAllRegisteredTests()
	}
	var tests []*testbackend.TestCase
	for i := 0; i < len(testConstructors); i++ {
		tc := testConstructors[i]() // capture range variable
		tests = append(tests, tc)
		t.Run(tc.GetInfo(), tc.RunTest(NewGraph(), false))
	}
	file, ok := os.LookupEnv("ONNX_COVERAGE")
	if ok {
		// TODO write the coverate to a file
		f, err := os.Create(file)
		if err != nil {
			t.Fatal("cannot write report", err)
		}
		defer f.Close()
		sort.Sort(testreport.ByStatus(tests))
		fmt.Fprintf(f, "Covering %.1f%% of the onnx integration tests\n", testreport.Coverage(tests))
		testreport.WriteCoverageReport(f, tests, testreport.ReportTable)
	}
}

func runner(t *testing.T, testConstuctors []func() *testbackend.TestCase) []report {
	t.Helper()
	status := make([]report, 0)
	return status
}
