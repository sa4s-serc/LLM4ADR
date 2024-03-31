---
title: 0006 Use Redux
---

# Use [Redux](https://redux.js.org) to manage state and [Redux Thunk](https://github.com/gaearon/redux-thunk) middleware to write action creators that return functions

In React, though parent components can pass information to their children components, it's atypical for children components to pass information to parent components. This makes it difficult to handle state that is consistent across multiple components (such as authentication and authorization). Doing so using only React causes a loss in modularity.
Redux is by far the most popular tool used to address the above issue. It also allows for easier testing. Because we'd also like to be able to use thunks (functions that wrap expressions to delay their evaluations) to exert control on when an action is dispatched, we chose redux-thunk middleware once we had settled on using redux.

## Considered Alternatives

* Redux
* MobX

## Decision Outcome

* Chosen Alternative: Redux
* Redux is the most obvious choice to address this issue. There are many supporting tools (such as redux-thunk) already built to address features Redux itself doesn't address. A few of our team members also already had experience with Redux.
* We can also use [Redux dev tools](https://github.com/zalmoxisus/redux-devtools-extension) for debugging. Users of redux devtools must add the extension to their browser of choice (follow instructions in the link above).

## Pros and Cons of the Alternatives

### Redux

* `+` Truss has used in other projects (ProtoWeb App)
* `+` Most commonly used tool to address state management in React.
* `+` Helpful dev tools.
* `-` Haven't explored using other tools thoroughly, so don't really know if this is the 'best' for this specific app.

### MobX

* `+` There are rustlings in the community that this is another good tool.
* `-` Truss hasn't used before.
* `-` No convincing argument that it's better than Redux.
* `-` Redux more prevalent.
