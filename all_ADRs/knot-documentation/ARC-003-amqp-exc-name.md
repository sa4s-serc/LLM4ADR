# ARC-003: AMQP Exchange name

## Status
Proposed

## Context

As defined by CloudAMQP Blog, "exchanges are message routing agents, defined by the virtual host within RabbitMQ. An exchange is responsible for routing the messages to different queues with the help of header attributes, bindings, and routing keys."
The Exchange name needs to be defined taking into account the architecture defined for the system. Thus, there will be a client that communicates with BabelTower, which has the Connector as one of its services (in this case BabelTower is the Connector client) or the client communicates directly with the Connector.
The goal is to define names that make it clear what this Exchange is used for.

## Decisions

* The communication between two components will be done with two queues, where one component is consumer in one and producer in another and the other component follows the opposite.
* The reference for choosing the queue name will be the client.
* The queue where BabelTower publishes messages will be called 'fogOut'.
* The queue where BabelTower consumes messages will be called 'fogIn'.
* The queue where Connector publishes messages will be called 'connOut'.
* The queue where Connector consumes messages will be called 'connIn'.

## Consequences

These decisions are sufficient to understand the meaning of each queue. Following this recommendation requires a change in the systems involved to use the new nomenclature and now the client should know to which server it is communicating with (e.g. the Connector or the BabelTower).
