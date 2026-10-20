Calibrate the synthetic oxide-semiconductor TFT model against the synthetic measurement.
Call tcad_observe first. Then repeatedly choose one action: tcad_update_group on an untried group with a
positive predicted gain, tcad_activate_component only for a component listed in eligible_components, or
tcad_stop when no action is expected to reduce J or the remaining error looks like model-form error.
Give a one-sentence physical rationale for every action. After tcad_stop, report J, the activations and the
held-out errors from the returned summary.
