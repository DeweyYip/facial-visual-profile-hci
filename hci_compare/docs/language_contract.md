# Language input and wording contract v1

## Scope

The October 30 layer converts current model predictions into basic English descriptions without an LLM. It does not identify a person, diagnose conditions, rate appearance, infer personality or generate personalized advice. It describes model outputs, rather than declaring annotations or predicted traits to be verified facts.

## Structured input

language_input.schema.json defines the JSON shape. language.validate_payload also checks semantic constraints that the schema alone does not fully express: original attribute order; model/evidence identity; finite numeric data; available results versus null missing values; decision == (score >= that model's frozen threshold); geometry eligibility; and agreement recomputed from available model decisions.

| Field | Meaning |
| --- | --- |
| schema_version / policy_version | fvp_language_v1 / template_v1 |
| score_meaning | uncalibrated_model_score |
| wording_policy | Description-only margin 0.05 and maximum six claims |
| input | Final RGB fingerprint, corruption family and severity; no photograph or upload path |
| detection | Detector reason, face count and geometry validity |
| models.r18/r50/m3 | Fixed Seed1 identity, availability and 24 ordered evidence records |
| evidence record | Evidence ID, attribute, score, frozen threshold, original boolean decision |
| agreement | Per-attribute agreement among available decisions; fewer than two available models is insufficient |

Missing/inference-failed models have null scores and decisions. A threshold is retained only when its model loaded. No other model's values are substituted. The three model score vectors are never averaged or merged. Agreement indicates matching classifier decisions, not truth or correctness.

## Wording rules

1. All facial descriptions require exactly one detected face and valid fresh landmarks. This product gate applies even to RGB-only descriptions. Debug score tables can still show RGB outputs on a no-face input, but those outputs do not become a facial narrative.
2. Each tab describes one model. An unavailable model produces an explicit unavailability message, not a fallback to another model.
3. Only positive original decisions may supply attribute claims. Negative labels are omitted; a negative No_Beard decision, for example, does not prove a beard is present.
4. Positive scores within 0.05 of their own frozen threshold are omitted from prose. This margin is a transparent engineering abstention heuristic, not calibrated uncertainty or evidence that other claims are correct. It does not alter any score, threshold, displayed classifier decision or formal experiment result.
5. The template emits at most six claims in the original attribute order. It does not rank raw scores across attributes. Other predictions remain in the 24-attribute table.
6. Multiple wording-eligible hair colors or textures are omitted as ambiguous narrative details. Baldness suppresses overlapping color/texture/bangs descriptions. These are conservative wording choices, not relabeling of the model predictions. Mustache and No_Beard refer to distinct attributes and may coexist.
7. Smile does not become happiness; under-eye bags do not become tiredness or illness; gray hair does not become age; eyeglasses do not become a diagnosis. No age, sex/gender, race/ethnicity, identity, attractiveness, health, personality or lifestyle statement may be added.
8. Corruption notes describe the current selected input. They do not claim that a transform caused a particular prediction error. Model-disagreement notes report only the computed decision count and do not choose a winner.
9. Each emitted phrase has an evidence ID linking to the same model's positive record. The UI evidence table shows the phrase, score and threshold. Notes are deterministic summaries of policy, input metadata or model agreement, not additional facial claims.
10. Input/settings changes, invalid image inputs, model reloads and failures clear the old descriptions, evidence and JSON together with the original result controls. A language-validation failure suppresses prose but preserves otherwise valid image/score outputs.

## Allowed appearance phrases

| Attribute | Template phrase |
| --- | --- |
| Smiling | a smile |
| Mouth_Slightly_Open | a slightly open mouth |
| Eyeglasses | eyeglasses |
| Bangs | bangs |
| Wearing_Hat | a hat |
| Mustache | a mustache |
| No_Beard | no visible beard |
| Black_Hair | black hair |
| Bushy_Eyebrows | bushy eyebrows |
| Arched_Eyebrows | arched eyebrows |
| High_Cheekbones | high cheekbones |
| Bags_Under_Eyes | under-eye bags |
| Big_Lips | full lips |
| Big_Nose | a large nose |
| Double_Chin | a double chin |
| Oval_Face | an oval face shape |
| Receding_Hairline | a receding hairline |
| Blond_Hair | blond hair |
| Brown_Hair | brown hair |
| Gray_Hair | gray hair |
| Bald | baldness |
| Straight_Hair | straight hair |
| Wavy_Hair | wavy hair |
| Sideburns | sideburns |

## Future LLM boundary

No LLM, API key, external generation request or model download is part of this milestone. The schema and contract prepare a later language-layer experiment. Any future LLM must receive only validated structured evidence, keep sources distinct, obey unavailable/face gates and return evidence links for facial assertions. Input data must not be treated as instructions.

The present audit_description check verifies exact deterministic template output and evidence. It is not a general semantic hallucination detector for arbitrary LLM prose. Free-form local/API output needs its own later checks; template tests alone do not establish future LLM faithfulness.

## Development examples and limits

The 60 entries in development_cases.json are synthetic structured fixtures, not 60 photographs or real-user observations. They are development inputs with explicit expected claim sets and statuses. They cover all 24 phrases and policy edge cases. The check is not an accuracy experiment, a human user study, or a held-out language evaluation. The later 20 reserved evaluation cases are not included or consumed here.

| Cases | Coverage |
| --- | --- |
| dev_001–dev_024 | One positive case for each allowed attribute |
| dev_025–dev_036 | Threshold equality, near-boundary scores, per-model thresholds and wording margin |
| dev_037–dev_044 | Detection failures and unavailable models |
| dev_045–dev_050 | Hair ambiguity and compatible attribute combinations |
| dev_051–dev_055 | Model disagreement, one-model availability and six-claim cap |
| dev_056–dev_060 | All five corruption families |
