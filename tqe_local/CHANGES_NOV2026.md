# Fixes after the September 2026 audit (applied 10 Nov 2026)

## Code changes

| File | Change | Why |
|---|---|---|
| `ansatze.py` | Every ansatz takes `layer_hook`, called after each layer's entangling block. One generic builder for all topologies; data-encoding variants now exist for every topology (star, all-to-all, heavy-hex included). | Noise was applied once after the whole circuit, but the paper says "per qubit per layer". |
| `ansatze.py` | New `heavy_hex_true`: connected n-qubit subgraph of the real IBM Falcon-27 coupling map (BFS from qubit 0). Old `heavy_hex` kept, labelled LEGACY. | The old "heavy-hex" was a chain plus (i, i+2) chords, not a heavy-hex graph. |
| `metrics.py` | All noisy QNodes pass the noise through `layer_hook`. | Per-layer noise. |
| `metrics.py` | `haar_kl_divergence(..., reference="real")` by default; new `frame_potential_ratio`. | RY + CNOT/CZ circuits produce only real states, so the complex-Haar reference made KL saturate at large n. The frame-potential ratio does not saturate. |
| `metrics.py` | New `gradient_variance_all` (all parameters; costs `z0` and `local`). | The theta_0 / Z_0 diagnostic is not representative for asymmetric topologies (star, linear). |
| `run_gap23.py` | Noise loop fixed. | It iterated over the preset dict as if it held (fn, kwargs) pairs; every noisy cell raised, the error was swallowed, and the noisy Gap-3 sweep wrote **zero rows**. |
| `run_gap23.py` | Flag only: the "RZ-RY-RZ" layer actually applies **RX**-RY-RZ. | Make code and paper agree before reusing those results. |
| `run_fixed_rerun.py` | New runner. Noisy and noiseless cells use identical seeds (0..runs-1) and identical run counts; one Jacobian gives theta_0, z0-all and local-all variances; kernel rank uses 64 data points. | Paired comparison removes the seed-mismatch artefact; 20 data points capped kernel rank at 20. |
| `test_fixes.py` | 7 tests (all pass). | Noise count per layer, per-layer differs from end-only, noiseless unchanged vs old code, states are real, real-Haar sampler, Falcon subgraphs connected with degree <= 3, new metrics run. |

The old theta_0 diagnostic from the new runner reproduces the old `gradient_variance` exactly for the same seed (checked: 0.15699 vs 0.15699).

## Paper text that must change (from the September audit)

1. Noise section (VI-E): rewrite from the new paired, per-layer results.
2. Remove the reported noise percentage differences unless they survive the paired rerun.
3. Kernel rank values 24.7-25.8 are not in the data (old cap was 20). Replace with the 64-point rerun.
4. Heavy-hex kernel rank: now computed (both legacy and true heavy-hex).
5. Haar KL: switch to the real-Haar reference and report the frame-potential ratio.
6. "Four orders of magnitude" headline: replace with the measured spread.
7. State which gradient diagnostic is used (all-parameter, local cost) and keep theta_0 as a supplementary check.
8. Describe heavy-hex as the real Falcon-27 subgraph, or say clearly that the old one was an approximation.
