package geocolor_test

import (
	"fmt"
	"math"
	"testing"

	_ "github.com/gomlx/gomlx/backends/default"

	"github.com/TuSKan/geocolor"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"
)

func TestGeocolorPipeline(t *testing.T) {
	backend, err := backends.New()
	if err != nil {
		t.Fatalf("Failed to initialize backend: %v", err)
	}
	fmt.Printf("Using backend: %s\n", backend.Name())
	ctx := context.New()

	// 2. Define grid parameters
	resL, resA, resB := 4, 5, 5
	minBounds := [3]float64{0.0, -10.0, -10.0}
	maxBounds := [3]float64{100.0, 10.0, 10.0}
	originIndex := [3]int{0, 2, 2}

	// 3. Create the execution closure for the end-to-end pipeline
	pipelineFn := func(ctx *context.Context, g *graph.Graph) *graph.Node {
		// a. Build Coordinate Grid
		coords := geocolor.BuildCoordinateGridGraph(g, resL, resA, resB, minBounds, maxBounds)

		// b. Compute Local Metric Tensor
		localMetrics := geocolor.ComputeLocalMetric(g, coords)

		// c. Solve Global Distances via iterative Eikonal solver
		distances := geocolor.SolveGlobalDistances(g, localMetrics, originIndex)

		// d. Apply Diminishing Returns scaling (c = 0.1)
		cParam := graph.Const(g, float32(0.1))
		dampened := geocolor.ApplyDiminishingReturns(g, distances, cParam)

		// e. Extract Neutral Axis
		neutralIndices := geocolor.ExtractNeutralAxis(g, dampened)

		// f. Compute final Geometric Chroma and Hue parameters
		finalGeometry := geocolor.ComputeGeometricColor(g, coords, neutralIndices, minBounds, maxBounds)

		return finalGeometry
	}

	// 4. Execute the graph and capture the resulting tensor
	e, err := context.NewExec(backend, ctx, pipelineFn)
	if err != nil {
		t.Fatalf("Failed to compile GoMLX pipeline: %v", err)
	}
	results, err := e.Exec()
	if err != nil {
		t.Fatalf("Failed to execute GoMLX pipeline: %v", err)
	}

	geometryTensor := results[0]
	
	// 5. Extract underlying float32 slice
	vals, ok := geometryTensor.Value().([][][][]float32)
	if !ok {
		// Use tensor parsing if direct recast fails based on backend shape mapping
		t.Fatalf("Result tensor was not [][][][]float32")
	}

	// 6. Assertions over resulting geometry
	for L := 0; L < resL; L++ {
		minC_geo := float32(math.Inf(1))
		
		for a := 0; a < resA; a++ {
			for b := 0; b < resB; b++ {
				// Structure is [L, C_geo, h_geo]
				C_geo := vals[L][a][b][1]
				if C_geo < minC_geo {
					minC_geo = C_geo
				}
			}
		}

		// The neutral axis should have essentially zero chroma
		if minC_geo >= 1e-4 {
			t.Errorf("At level L=%d, minimum geometric chroma %f is not near zero", L, minC_geo)
		} else {
			fmt.Printf("Level L=%d geometric origin correctly shifted. Min C_geo = %f\n", L, minC_geo)
		}
	}
}
