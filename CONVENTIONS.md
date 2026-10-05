# Writing conventions for the note

Adapted from the conventions of the anode-consumption article. Where they
differ it is because this document is a measurement note, not a theory paper:
almost everything in it is something that was measured, and the reader needs to
know how.

**The reader.** A computational scientist who knows aluminium cells and does not
know this project. Not a specialist in this surrogate, not a newcomer to
electrolysis. Keep the text on the project; do not regress into the physics.

## 1. Claim first

Every section, subsection and paragraph opens with the claim it establishes, in
one sentence, before any setup. A reader who stops after the first sentence
should have the finding. Never build to a result.

## 2. Say how every quantity is measured

A number without its protocol is not a result. Each reported quantity names the
run or dataset it came from, the settings that produced it, and the units. The
protocol itself goes to the appendix; the text carries the claim and the number.

## 3. Measured and assumed never blur

Mark which numbers are measurements, which are estimates, and which are taken
from the literature. A sentence that mixes them without saying so is a defect.

## 4. Every effect is reported against its noise floor

No effect size without the floor it is being judged against, in the same
sentence or the next. "22.9% better, against a 0.5% floor" is a result;
"22.9% better" is not.

## 5. As few symbols as possible

A symbol must earn its place by being used at least three times in separated
places. Otherwise write the words. No symbol carries two meanings anywhere in
the document, including across scalar/vector or roman/italic distinctions.
Index convention: j anodes (24), m modes, i nodes (M), n runs (N).

## 6. No undefined terms

Every term is defined at first use or is not used. This includes terms that are
standard in the surrogate literature but not in the cell literature, and the
reverse.

## 7. Few numbered environments

Around eight in the whole document, each load-bearing: a definition that is
referred to later, a proposition that is proved, a measurement that is cited
elsewhere. Everything else is prose. A numbered environment nobody references
is a defect.

## 8. Negative results are stated as results

A refuted idea keeps its section and says plainly that it was refuted and on
what evidence. Do not delete the attempt and do not soften it.

## 9. Operating regimes are operational, not pathological

`gaussian` is normal operation. `weak`, `single` and `dead` are anode
replacements — scheduled events, not faults. No "fault", "failure" or "stress"
language for them. The campaign's regime mix is a design choice, not an
imbalance.

## 10. References where they are needed

Every claim that is not established in this document cites a source at the
point of use, not in a bibliography sweep at the end.

## 11. Displayed equations are part of the sentence

Punctuated as the sentence requires, introduced by text that says what they do.
No bare `\eqref` as the subject of a clause.

## 12. Numbers carry their conditions

Mesh, tolerance, run length, number of samples. A reported J names the window it
was averaged over and the time it was taken at. A reported error names the split
it was evaluated on.
