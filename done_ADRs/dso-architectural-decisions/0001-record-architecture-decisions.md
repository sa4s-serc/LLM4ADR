# ADR-001. Record architectural decisions

Date: 12/08/2019

## Status

Accepted

## Context

The modernisation of the Fix & Go infrastructure requires a number of architectural decisions.

When making decisions, we should record them somewhere for future reference, and to help us remember why we made them.

## Decision

We will use Architecture Decision Records, as described by Michael Nygard in
this article: http://thinkrelevance.com/blog/2011/11/15/documenting-architecture-decisions.

An architecture decision record is a short text file describing a single
decision.

We will keep ADRs in this repository under decisions/adr-NNN-AAAA.md

We should use a lightweight text formatting language like Markdown.

ADRs will be numbered sequentially and monotonically. Numbers will not be
reused.

If a decision is reversed, we will keep the old one around, but mark it as
superseded. (It's still relevant to know that it was the decision, but is no
longer the decision.)

We will use a format with just a few parts, so each document is easy to digest.
The format has just a few parts.

**Title** These documents have names that are short noun phrases. For example,
"ADR 1: Record architectural decisions" or "ADR 9: Use Docker for deployment"

**Status** A decision may be "proposed" if the project stakeholders haven't
agreed with it yet, or "accepted" once it is agreed. If a later ADR changes or
reverses  a decision, it may be marked as "deprecated" or "superseded" with a
reference to its replacement.

**Context** This section describes the forces at play, including technological,
political, social, and project local. These forces are probably in tension, and
should be called out as such. The language in this section is value-neutral. It
is simply describing facts.

**Decision** This section describes our response to these forces. It is stated
in full sentences, with active voice. "We will ..."

**Consequences** This section describes the resulting context, after applying
the decision. All consequences should be listed here, not just the "positive"
ones. A particular decision may have positive, negative, and neutral
consequences, but all of them affect the team and project in the future.

The whole document should be one or two pages long. We will write each ADR as
if it is a conversation with a future person joining the team. This requires
good writing style, with full sentences organised into paragraphs. Bullets are
acceptable only for visual style, not as an excuse for writing sentence
fragments.

## Consequences

One ADR describes one significant decision for the project. It should be
something that has an effect on how the rest of the project will run.

Developers and project stakeholders can see the ADRs, even as the team
composition changes over time.

The motivation behind previous decisions is visible for everyone, present and
future. Nobody is left scratching their heads to understand, "What were they
thinking?" and the time to change old decisions will be clear from changes in
the project's context.

## Credit

Thanks to @daibach and the LAA team at MoJ. This first ADR and the repo structure was copied from https://github.com/ministryofjustice/laa-hosting-architectural-decisions.