# Living Documentation

## Context

A dashboard with more information and deep-links to resources, e.g. setting up SSO with existing identity providers, GuardDuty/Security Hub dashboards.

## Decision

- Create a CloudWatch Dashboard called `superwerker` in the AWS management account. The CW dashboard a) ensures a deep link which can be used to link from the README.md and b) ensures the user is authorized to access the information.
- Display DNS delegation state and setup instructions
- Refresh dashboard with scheduler every minute since this removes the compexity to deal with event-based dashboard generation. Lambda invocations are completely covered by free-tier.

## Consequences

- CW Dashboards don't support auto-reload for text widgets, so browser reload has to be done by the user.
