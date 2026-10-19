# Plugging in your own simulator

The workflow talks to a simulator only through this interface (`agentic_tcad_demo.device.Simulator`):

```python
class Simulator(Protocol):
    def simulate(self, params: Mapping[str, float], active: frozenset[str],
                 cond: Condition, vg: np.ndarray) -> SimResult: ...
```

**Inputs**
- `params`: physical parameter values by name.
- `active`: the set of enabled model components.
- `cond`: the drain bias `vd` [V] and the channel length `length_um` [µm].
- `vg`: the gate-voltage grid [V].

**Output.** Return `SimResult(vg, id, converged, message)`, where `id` is |I<sub>D</sub>| in A on the same
grid. Set `converged=False` instead of raising when the solve fails. You may also provide a
`recovered()` method that returns a simulator with more robust numerical settings. The workflow calls it
once after a failure.

**Configuration**

```yaml
simulator:
  kind: plugin
  entry: "my_private_package.adapter:MySimulator"
  options: {work_dir: "..."}      # passed to the constructor
```

Keep any adapter for a commercial TCAD tool in a separate, private package. Its input decks, vendor
examples, documentation excerpts, license-server settings, and measured data must never be added to this
repository. The synthetic "measurement" generator in `runner.py` must also be replaced by your own,
access-controlled data loader.
