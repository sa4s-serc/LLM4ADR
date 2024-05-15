# ARC-002: Fog-Cloud Synchronization

## Status
Proposed

## Context

One of the main questions that appear when discussing the new fog service development/integration was how the device contextual representation and data should be synchronized between fog and cloud. Before, when we were using the Meshblu we decided to send all the messages (device registration, schema, and data) to the cloud independently of the message type, data relevance at time or cloud connectivity state. This solution was simple for that moment but was increasing the internal message queues size and obviously isn't optimal mainly when dealing with a constrained environment. Therefore, our goal is to start with a simple but optimal approach in the first stage of the new fog service integration.

## Decisions

* Retry to register and update the device's schema on the cloud five times and if the error persists retry with ten minutes interval.
* Retry to publish only the last device's data.
* Retry to execute only the last commands sent to the device, avoiding to store a lot of pending commands on the message queue (which could lead to inconsistencies).

## Consequences

These decisions are simple and enough for the current development stage. However, the users could have problems to identify if the operations have successfully occurred in an intermittent connection scenario. Because of that, in the next steps, we will need to analyze how to create a synchronization process that meets both user experience and performance requirements.