# Demo: the same abstract, written for a stranger

## What is there now (188 words, one 8-line sentence)

> This note records, with precise statements and with the provenance of every number,
> the results of Phase 0 of the extension of the magnetohydrodynamic surrogate of
> Chapter 2. Four results are established on the existing campaign of N = 1,594
> simulations, with no new solver runs. (i) A *quadratic functional* of the velocity,
> such as a kinetic energy, is badly predicted by regressing it directly on the anode
> currents and well predicted by evaluating it on a surrogate *field*; the governing
> quantity is the field model's residual, which the functional amplifies, so the
> network suffices for velocity and a linear model does not. ...

## Proposed (190 words, longest sentence 26)

> Simulating the flow inside an aluminium smelting cell takes seven hours. A machine
> -learned surrogate does it in milliseconds, but it predicts a *velocity field*,
> and nobody makes a decision from a velocity field. The quantities engineers act on
> are single numbers computed from it: a kinetic energy, a peak wave height, a
> measure of how much current runs sideways through the metal.
>
> The obvious approach is to predict those numbers directly from the inputs. It fails
> badly — often worse than predicting their average. This paper explains why, and
> shows that the fix is to predict the whole field and then compute the number from
> it exactly.
>
> Whether that helps, and by how much, turns out to be governed by one measurable
> property of the quantity being computed, which we give as a rule: some quantities
> gain nothing from a field model, some need only a crude one, and some need an
> accurate one. We verify the rule on thirty quantities.
>
> Applying it to the case that originally motivated a nonlinear surrogate reverses
> that conclusion: the underlying physics was linear all along, and the nonlinearity
> was an artefact of how the quantity was defined.

## What changed, mechanically

| | before | after |
|---|---|---|
| first sentence about | the document | the problem |
| longest sentence | 8 lines | 26 words |
| terms assumed | 8 | 0 |
| numbers in the abstract | 6 | 0 |
| "why should a stranger care" | absent | sentences 1-3 |

Numbers are not missing by accident. An abstract argues that a question matters and
says what was found; the numbers belong in the section that earns them.
