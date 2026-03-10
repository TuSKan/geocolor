package geocolor

import (
	"fmt"
	"math"
	"testing"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"

	_ "github.com/gomlx/gomlx/backends/default"
)

func floatEquals(a, b float64) bool {
	return math.Abs(a-b) <= 1e-3
}

func printRGBFailureRow(t *testing.T, label string, lch [3]float64, rgb [3]float64) {
	t.Errorf("%s -> LCH: [%.3f, %.3f, %.3f] RGB: [%.3f, %.3f, %.3f]", label, lch[0], lch[1], lch[2], rgb[0], rgb[1], rgb[2])
}

// TestConvertToRGB tests the geometric grid generation and the LCH to RGB conversion.
func TestConvertToRGB(t *testing.T) {
	backend := backends.MustNew()
	ctx := context.New()

	resL, resC, resH := 11, 11, 11
	cfg := &GridConfig{
		ResL:      resL,
		ResA:      resC,
		ResB:      resH,
		MinBounds: [3]float64{0.0, 0.0, -math.Pi},
		MaxBounds: [3]float64{100.0, 150.0, math.Pi},
		OriginIndex: [3]int{0, 0, 0},
	}

	execFn := func(ctx *context.Context, g *graph.Graph) []*graph.Node {
		// 1. Generate the Grid (which represents L, C, H coordinates based on cfg Min/MaxBounds)
		lchValsNode := BuildCoordinateGridGraph(g, cfg)
		// 2. Convert grid of LCH into RGB
		rgbValsNode := ConvertToRGB(g, lchValsNode)

		return []*graph.Node{lchValsNode, rgbValsNode}
	}

	e, err := context.NewExec(backend, ctx, execFn)
	if err != nil {
		t.Fatalf("Failed to compile GoMLX pipeline: %v", err)
	}

	results, err := e.Exec()
	if err != nil {
		t.Fatalf("Failed to execute GoMLX pipeline: %v", err)
	}

	// 4D slice: [ResL][ResA][ResB][3]float64
	labVals := results[0].Value().([][][][]float64)
	rgbVals := results[1].Value().([][][][]float64)

	// --- 1. Grid Generation Validation ---
	t.Run("Grid Generation", func(t *testing.T) {
		// Check that C index = 0 has Chroma = 0.0
		for l := 0; l < resL; l++ {
			for h := 0; h < resH; h++ {
				cVal := labVals[l][0][h][1] // Index 1 is Chroma (ResA)
				if !floatEquals(cVal, 0.0) {
					t.Errorf("Expected Chroma=0.0 at C index 0, got %f. Coordinates: %v", cVal, labVals[l][0][h])
				}
			}
		}

		// Check Hue axis spans from -math.Pi to math.Pi across ResB
		for h := 0; h < resH; h++ {
			expectedHue := -math.Pi + float64(h)*(2.0*math.Pi)/float64(resH-1)
			actualHue := labVals[0][0][h][2]
			if !floatEquals(actualHue, expectedHue) {
				t.Errorf("At Hue index %d, expected %f, got %f", h, expectedHue, actualHue)
			}
		}
	})

	// --- 2. Neutral Axis Color Test ---
	t.Run("Neutral Axis", func(t *testing.T) {
		for l := 0; l < resL; l++ {
			for h := 0; h < resH; h++ {
				cIndex := 0 // Chroma is 0
				r := rgbVals[l][cIndex][h][0]
				g := rgbVals[l][cIndex][h][1]
				b := rgbVals[l][cIndex][h][2]

				if !floatEquals(r, g) || !floatEquals(g, b) || !floatEquals(r, b) {
					t.Errorf("Expected neutral R=G=B at Chroma=0.0 (L=%d). Got R=%f, G=%f, B=%f", l, r, g, b)
					printRGBFailureRow(t, "Neutral Fail", [...]float64{labVals[l][cIndex][h][0], labVals[l][cIndex][h][1], labVals[l][cIndex][h][2]}, [...]float64{r, g, b})
				}
			}
		}
	})

	// --- Debug Trace Output ---
	if t.Failed() {
		fmt.Printf("\n--- Debug Trace Output ---\n")
		fmt.Printf("%-20s | %-30s | %-30s\n", "Location", "Expected/Input LCH", "Actual RGB")
		fmt.Printf("--------------------------------------------------------------------------------\n")
		
		// Center
		centerL, centerC, centerH := resL/2, resC/2, resH/2
		lch := labVals[centerL][centerC][centerH]
		rgb := rgbVals[centerL][centerC][centerH]
		fmt.Printf("%-20s | LCH: [%.3f, %.3f, %.3f] | RGB: [%.3f, %.3f, %.3f]\n", "Center", lch[0], lch[1], lch[2], rgb[0], rgb[1], rgb[2])

		// Far Edge
		edgeC := resC - 1
		lch = labVals[centerL][edgeC][centerH]
		rgb = rgbVals[centerL][edgeC][centerH]
		fmt.Printf("%-20s | LCH: [%.3f, %.3f, %.3f] | RGB: [%.3f, %.3f, %.3f]\n", "Far Edge C=150", lch[0], lch[1], lch[2], rgb[0], rgb[1], rgb[2])

		// Mid-Hue
		midH := resH / 4
		lch = labVals[centerL][centerC][midH]
		rgb = rgbVals[centerL][centerC][midH]
		fmt.Printf("%-20s | LCH: [%.3f, %.3f, %.3f] | RGB: [%.3f, %.3f, %.3f]\n", "Mid-Hue Angle", lch[0], lch[1], lch[2], rgb[0], rgb[1], rgb[2])
	}
}

// --- 3. Trigonometry Isolation ---
// Creates a standalone test case to validate Polar-to-Cartesian conversion.
func TestPolarToCartesian(t *testing.T) {
	backend := backends.MustNew()
	ctx := context.New()

	execFn := func(ctx *context.Context, g *graph.Graph) []*graph.Node {
		// Create a small tensor with specific L, C, H
		// We'll test:
		// [L=50, C=100, H=0] -> A=100, B=0
		// [L=50, C=100, H=pi/2] -> A=0, B=100
		// [L=50, C=100, H=pi] -> A=-100, B=0
		// [L=50, C=100, H=3pi/2] -> A=0, B=-100
		
		inputData := [][]float64{
			{50.0, 100.0, 0.0},
			{50.0, 100.0, math.Pi / 2.0},
			{50.0, 100.0, math.Pi},
			{50.0, 100.0, 3.0 * math.Pi / 2.0},
		}

		lchValsNode := graph.Const(g, inputData)
		
		C := graph.Slice(lchValsNode, graph.AxisRange(), graph.AxisRange(1, 2))
		H := graph.Slice(lchValsNode, graph.AxisRange(), graph.AxisRange(2, 3))
		
		A := graph.Mul(C, graph.Cos(H))
		B := graph.Mul(C, graph.Sin(H))
		
		return []*graph.Node{graph.Concatenate([]*graph.Node{A, B}, -1)}
	}

	e, err := context.NewExec(backend, ctx, execFn)
	if err != nil {
		t.Fatalf("Failed to compile GoMLX pipeline: %v", err)
	}

	results, err := e.Exec()
	if err != nil {
		t.Fatalf("Failed to execute GoMLX pipeline: %v", err)
	}

	abVals := results[0].Value().([][]float64)

	expectedPaths := [][]float64{
		{100.0, 0.0},
		{0.0, 100.0},
		{-100.0, 0.0},
		{0.0, -100.0},
	}

	for i, expected := range expectedPaths {
		actualA := abVals[i][0]
		actualB := abVals[i][1]

		if !floatEquals(actualA, expected[0]) || !floatEquals(actualB, expected[1]) {
			t.Errorf("PolarToCartesian test case %d failed: expected [A=%.3f, B=%.3f], got [A=%.3f, B=%.3f]",
				i, expected[0], expected[1], actualA, actualB)
		}
	}
}
