package geocolor

import (
	"math"

	"github.com/gomlx/gomlx/pkg/core/graph"
)

// ComputeLocalMetric calculates a simplified local metric tensor based on the CIEDE2000 color difference formula.
// It returns a tensor natively representing the edge scaling weights for adjacent voxels in the [L, a, b] grid.
func ComputeLocalMetric(ctx *graph.Graph, coords *graph.Node) *graph.Node {
	fMin := graph.Const(ctx, float64(1e-7))
	fOne := graph.Const(ctx, float64(1.0))

	// Helper for easy constants
	c := func(val float64) *graph.Node { return graph.Const(ctx, val) }

	// Extract channels
	L := graph.SliceAxis(coords, -1, graph.AxisRange(0, 1))
	a := graph.SliceAxis(coords, -1, graph.AxisRange(1, 2))
	b := graph.SliceAxis(coords, -1, graph.AxisRange(2, 3))

	a_sq := graph.Square(a)
	b_sq := graph.Square(b)
	
	// C = sqrt(a^2 + b^2 + epsilon) to prevent divide by zero
	C_sq := graph.Add(a_sq, b_sq)
	C := graph.Sqrt(graph.Add(C_sq, fMin))

	// Compute Hue angle h = atan2(b, a)
	// Atan2 returns radians in [-pi, pi]
	h := graph.Atan2(b, a)

	// Precompute degrees in radians for standard Go math to use as GoMLX constants
	deg2rad := float64(math.Pi / 180.0)
	
	// T = 1 - 0.17 cos(h - 30°) + 0.24 cos(2h) + 0.32 cos(3h + 6°) - 0.20 cos(4h - 63°)
	
	// 1. cos(h - 30°)
	h_minus_30 := graph.Sub(h, c(30.0 * deg2rad))
	cosH_minus_30 := graph.Cos(h_minus_30)

	// 2. cos(2h)
	h2 := graph.Mul(h, c(2.0))
	cos2H := graph.Cos(h2)

	// 3. cos(3h + 6°)
	h3 := graph.Mul(h, c(3.0))
	h3_plus_6 := graph.Add(h3, c(6.0 * deg2rad))
	cos3H_plus_6 := graph.Cos(h3_plus_6)

	// 4. cos(4h - 63°)
	h4 := graph.Mul(h, c(4.0))
	h4_minus_63 := graph.Sub(h4, c(63.0 * deg2rad))
	cos4H_minus_63 := graph.Cos(h4_minus_63)

	// Combine into T
	T := fOne
	T = graph.Sub(T, graph.Mul(c(0.17), cosH_minus_30))
	T = graph.Add(T, graph.Mul(c(0.24), cos2H))
	T = graph.Add(T, graph.Mul(c(0.32), cos3H_plus_6))
	T = graph.Sub(T, graph.Mul(c(0.20), cos4H_minus_63))

	// Compute SL
	// SL = 1 + (0.015 * (L - 50)^2) / sqrt(20 + (L - 50)^2)
	L_minus_50 := graph.Sub(L, c(50.0))
	L_minus_50_sq := graph.Square(L_minus_50)
	SL_den := graph.Sqrt(graph.Add(c(20.0), L_minus_50_sq))
	SL_num := graph.Mul(c(0.015), L_minus_50_sq)
	SL := graph.Add(fOne, graph.Div(SL_num, SL_den))

	// Compute SC
	// SC = 1 + 0.045 * C
	SC := graph.Add(fOne, graph.Mul(c(0.045), C))

	// Compute SH
	// SH = 1 + 0.015 * C * T
	SH := graph.Add(fOne, graph.Mul(c(0.015), graph.Mul(C, T)))

	// Calculate metric edge weights for a diagonal representation of CIEDE2000 Jacobian.
	// wL = 1 / SL
	wL := graph.Div(fOne, SL)

	// Since da and db mix components of dC and C dh, we project the scale weights:
	// g_aa = a^2 / (C^2 SC^2) + b^2 / (C^2 SH^2)
	// g_bb = b^2 / (C^2 SC^2) + a^2 / (C^2 SH^2)
	
	SC_sq := graph.Square(SC)
	SH_sq := graph.Square(SH)
	
	a_sq_div_C_sq := graph.Div(a_sq, graph.Add(C_sq, fMin))
	b_sq_div_C_sq := graph.Div(b_sq, graph.Add(C_sq, fMin))

	g_aa := graph.Add(graph.Div(a_sq_div_C_sq, SC_sq), graph.Div(b_sq_div_C_sq, SH_sq))
	g_bb := graph.Add(graph.Div(b_sq_div_C_sq, SC_sq), graph.Div(a_sq_div_C_sq, SH_sq))

	// Handle neutral origin C = 0 safely: fallback to 1/SC for both. 
	// As C->0, a->0 and b->0, so the limits cleanly resolve to isotropic weights, but division by C+epsilon might be noisy.
	isZero := graph.LessOrEqual(C_sq, fMin)
	g_aa = graph.Where(isZero, graph.Div(fOne, SC_sq), g_aa)
	g_bb = graph.Where(isZero, graph.Div(fOne, SC_sq), g_bb)

	// weights = sqrt(g_ii)
	wa := graph.Sqrt(g_aa)
	wb := graph.Sqrt(g_bb)

	return graph.Concatenate([]*graph.Node{wL, wa, wb}, -1)
}

// ApplyDiminishingReturns implements a non-Riemannian dampening scaling function
// on a tensor of accumulated path distances. It uses the logarithmic dampening
// f(x) = ln(1 + c * x), where `c` is a configurable scaling parameter passed as a *graph.Node.
// This allows the metric space to exhibit sub-additive geodesics over large distances.
func ApplyDiminishingReturns(ctx *graph.Graph, distances *graph.Node, cParam *graph.Node) *graph.Node {
	// f(x) = ln(1 + c * x)
	// We use Log1p(x) which computes ln(1 + x) more accurately for small x.
	scaled := graph.Mul(cParam, distances)
	return graph.Log1p(scaled)
}
