# Decision rules for a planning agent

## Observation fields

| Field | Meaning |
|---|---|
| `J`, `terms` | Objective (paper Eq. 1) and its weighted terms: `log`, `lin`, `vth`, `ion`, `ss` |
| `dominant_term`, `diagnosis` | The largest weighted term and its physical reading |
| `parameters` | Current values (phi_SD, phi_G in eV, mu0 in cm2/Vs, Q_f in cm^-2, N_tail in cm^-3 eV^-1) |
| `groups[]` | Parameter groups with the OFAT-predicted Gauss–Newton gain and whether they were tried this cycle |
| `eligible_components[]` | Components that may be activated now, with the evidence |
| `parameter_updates_stalled` | True after two accepted updates below 1% relative gain. Only an activation or a stop is then legal |

## Legal actions

| Action | Legal when |
|---|---|
| `update_group` + group name | The group is listed, not yet tried this cycle, and updates are not stalled |
| `activate_component` + component | The component is listed in `eligible_components` |
| `stop` (target null) | Always |

## Physics hints (synthetic model)

- **Rigid V_th offset:** move `electrostatics` (phi_G). If phi_G is pinned at a bound and the offset
  persists, `fixed_charge` becomes eligible.
- **On-current deficit with a high S/D work function:** reduce the contact barrier (`contact`) before
  raising mobility (`transport`).
- **Persistent SS mismatch:** no base parameter can fix it. `tail_states` becomes eligible.
- **phi_G and Q_f:** they act only through V_T, so they are not separately identifiable. Avoid drifting
  along that direction for negligible gains.
- **When to stop:** stop when every predicted gain is a small fraction of J and the remaining error
  looks like model-form error.
