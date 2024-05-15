# 5. For release, prioritise consistency over availability

Date: 2021-06-03

## Status

Accepted

## Context

1. We have a fixed deadline
1. We will have our first users week beginning 7 June 2021
1. For our Delius integration, we rely on an intricate set of reference data
1. We are uncertain if there are any issues with the reference data for all possible combinations of offenders
1. We rely on these integrations to set up "container" data in nDelius. If they fail, we need to recover before providers book appointments

We want to ensure when updates fail, we have a way to resume that retries the interaction.

However, **we do not have retry mechanisms in place yet**.

We could add an admin task/script to retry a certain event (we use Spring Application Events that don't
create side effects, so we could retrigger those).

We feel this would create an overhead that would be too much administrative burden together with
the anticipated noise of most users starting the service at the same time.

## Decision

Due to the uncertainties and lack of convenient retry mechanism,
we favour **consistency over availabilty** in the short term.

## Consequences

In other words, we'd rather have the workflow block with an error, fix the problem, roll forward,
ask the user to retry, than the alternative of having to collect and re-trigger by hand.

We will amend the sending the referral workflow to try Delius NSI ("activity") creation synchronously.
The transaction will abort without sending the referral, if the NSI cannot be created.

We assume this and the already sync appointment booking would capture the majority of reference data problems.

We recognise this is only a short term fix until we gain enough confidence.

## We know we can roll this back when

Whichever happens first:

- Our error rate for reference data problems is below 2% over 30 days (after all users are onboarded).
- We put the retry mechanism in place.
