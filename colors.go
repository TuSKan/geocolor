package geocolor

import "github.com/gomlx/gomlx/pkg/core/graph"

// ConvertToRGB transforms a tensor of LCH coordinates to sRGB (D65).
// Input colorsLCH expects shape [..., 3] where the last dimension is L, C, H.
// Returns a tensor of the exact same shape containing the RGB floats.
func ConvertToRGB(g *graph.Graph, colorsLCH *graph.Node) *graph.Node {
	// 1. Slice out the L, C, and H channels.
	rank := colorsLCH.Rank()
	axes := make([]graph.SliceAxisSpec, rank)
	for i := 0; i < rank-1; i++ {
		axes[i] = graph.AxisRange()
	}

	axes[rank-1] = graph.AxisRange(0, 1)
	L := graph.Slice(colorsLCH, axes...)
	axes[rank-1] = graph.AxisRange(1, 2)
	C := graph.Slice(colorsLCH, axes...)
	axes[rank-1] = graph.AxisRange(2, 3)
	H := graph.Slice(colorsLCH, axes...)

	// THE MISSING MATH: Without this, every petal renders as Red!
	A := graph.Mul(C, graph.Cos(H))
	B := graph.Mul(C, graph.Sin(H))

	// Lab to XYZ (D50) components
	fy := graph.Div(graph.Add(L, graph.Const(g, float64(16.0))), graph.Const(g, float64(116.0)))
	fx := graph.Add(graph.Div(A, graph.Const(g, float64(500.0))), fy)
	fz := graph.Sub(fy, graph.Div(B, graph.Const(g, float64(200.0))))

	// Helper function for the piecewise CIELAB inverse curve
	invCurve := func(t *graph.Node) *graph.Node {
		threshold := graph.Const(g, float64(0.206893))
		cond := graph.GreaterThan(t, threshold)

		// True branch: t^3
		trueBranch := graph.Pow(t, graph.Const(g, float64(3.0)))

		// False branch: (t - 16/116) / 7.787
		falseBranch := graph.Div(
			graph.Sub(t, graph.Const(g, float64(16.0/116.0))),
			graph.Const(g, float64(7.787)),
		)
		return graph.Where(cond, trueBranch, falseBranch)
	}

	// Apply curve and multiply by D50 Reference White
	X := graph.Mul(invCurve(fx), graph.Const(g, float64(0.96422)))
	Y := graph.Mul(invCurve(fy), graph.Const(g, float64(1.00000)))
	Z := graph.Mul(invCurve(fz), graph.Const(g, float64(0.82521)))

	// Recombine into XYZ tensor of shape [..., 3]
	xyz := graph.Concatenate([]*graph.Node{X, Y, Z}, -1)

	// 3. Chromatic Adaptation (Bradford D50 -> D65)
	bradfordMatrix := graph.Const(g, [][]float64{
		{0.95557, -0.02828, 0.01229},
		{-0.02303, 1.00994, -0.02048},
		{0.06316, 0.02101, 1.32991},
	})

	xyz65 := graph.DotGeneral(
		xyz, []int{-1}, nil,
		bradfordMatrix, []int{0}, nil,
	)

	// 4. XYZ (D65) to Linear sRGB Matrix
	xyzToRgbMatrix := graph.Const(g, [][]float64{
		{3.2406, -0.9689, 0.0557},
		{-1.5372, 1.8758, -0.2040},
		{-0.4986, 0.0415, 1.0570},
	})

	rgbLinear := graph.DotGeneral(
		xyz65, []int{-1}, nil,
		xyzToRgbMatrix, []int{0}, nil,
	)

	// 5. sRGB Gamma Correction (Piecewise)
	thresholdGamma := graph.Const(g, float64(0.0031308))
	condGamma := graph.LessOrEqual(rgbLinear, thresholdGamma)

	// True Branch: 12.92 * c
	trueGamma := graph.Mul(rgbLinear, graph.Const(g, float64(12.92)))

	// False Branch: 1.055 * c^(1/2.4) - 0.055
	// Clamp negative linear values to 0.0 so XLA's Pow doesn't throw NaNs
	safeRgbLinear := graph.Max(rgbLinear, graph.ZerosLike(rgbLinear))
	powTerm := graph.Pow(safeRgbLinear, graph.Const(g, float64(1.0/2.4)))
	falseGamma := graph.Sub(
		graph.Mul(graph.Const(g, float64(1.055)), powTerm),
		graph.Const(g, float64(0.055)),
	)

	// Apply Gamma curve and return the final [..., 3] tensor
	return graph.Where(condGamma, trueGamma, falseGamma)
}
