# 4. Store time and actor information in data

Date: 2020-12-01

Note: this is a backfilled decision, exact date unknown 😅

## Status

Accepted

## Context

Interventions data, initially for Commissioned Rehabilitative Services (CRS), will be analysed and used in reporting.

People need to use data available from our service to identify performance concerns:

- when did someone submit a referral
- when did someone book the first supplier assessment appointment
- when did someone submit the first version of the action plan
- etc.

Knowing _who_ did these actions is also valuable to our data users.

We already use [domain events](0002-decouple-with-events.md) and publish messages when these events happen.
However, these events are crucial to the operation of the service. They are valuable to be available for
examination **after** the events happened.

If we maintained state on the referral, it would mutate from draft to sent to waiting for supplier assessment, etc.
Figuring out a state in the past would be difficult without historical data. That data may not be captured often enough.

## Decision

Store all domain events in the data layer as `_at` and `_by` fields on the corresponding tables.

Avoid _persisted_ state fields as it makes accidental data hiding easier (the previous value is lost,
and the logic behind the state change might have changed).

## Consequences

Domain events also have a corresponding pair in the data:

- when did someone receive the referral: `referral.sent_at`, `referral.sent_by`
- when did someone submit the first version of the action plan: `action_plan.submitted_at`, `action_plan.submitted_by`
