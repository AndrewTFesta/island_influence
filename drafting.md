Words, words, words for Draft 1 abstract

the swarm is split into a multiagent team of leaders that learns to split swarm followers amongst 

Intelligent swarms are useful. The problem with them is that getting them to learn anything is hard

It’s hard because we don’t have a clean learning signal for individual agents

Multiagent learning can clean learning signals for a small team. Swarm shepherding can guide a swarm to achieve a task

Neither one works for both a team as large as a swarm, and can handle achieving several tasks

We solve it with multiagent leader based learning by splitting the swarm into simple swarm followers and a multiagent team of swarm leaders that learn how to split the swarm up to achieve different tasks.

The cool thing is that we can achieve complex swarm coordination with only a small multiagent team actually learning control policies

The contributions are the multiagent system formulation of learning leader-follower swarms, and a shaped fitness technique for these leaders to learn even better swarm coordination

MALBL outperforms standard MAS learning on an objective that requires the swarm to split up to handle various tasks.



Many real world problems would benefit from the large number of agents in a swarm, but would require learning from the swarm, which is hard

Words, words, words for Draft 2 intro

The key insight is that we can achieve complex learning-based coordination in a swarm without actually having each swarm member learn a control policy. Instead, we can have a relatively small subset of specialized learners in the swarm focused on higher level objectives while the rest of the swarm runs simple control policies. This keeps the noisiness in the system feedback low without compromising the capability of the swarm.

We reduce the number of learners in the system without compromising the capability of the swarm. By reducing the number of learners, we simultaneously reduce the noisiness of the system feedback and the computational load imposed by learning.

The cool thing is that we don’t need the entire swarm to be learning in order for us to learn complex coordination. That’s the big aha thing. Oh wait, we don’t need the entire swarm to be learning, we just need a few learners to be learning using some cool multiagent learning tricks and then we can get complex coordination with the WHOLE SWARM

The cool thing is that you don’t need to pick between a small team that can achieve complex coordination or a large team that can achieve simple coordination. You can have a both a large team and a team that can achieve complex coordination.


Words, words, words for Draft 1 Background

These difference objectives can be extended to tightly coupled domains through $D_{++}$, which introduces the concept of "stepping stone" feedback to ....

Multiagent learning inherently suffers from one big problem. There are multiple agents learning in one big environment and there is only one fitness function for the entire system . Distilling this system feedback signal into individual feedback signals for different agents is a large problem in multiagent learning, with many different approaches to actually solving this problem. These approaches are broadly categorized as reward shaping approaches, but apply to both rewards in multiagent reinforcement learning as well as fitnesses in cooperative co-evolutionary algorithms.

Difference objectives have been demonstrated to be quite effective for distilling system feedback into agent-specific feedback \cite{castellini2022difference, aida_2016, agogino2004efficient}. One of the advantages of difference objectives is that they provide a convenient way of disentangling the agent's specific impact on the system away from the system itself. They function 

This work extends potential based reward shaping to the multiagent domain. Potential based reward shaping is all about how you can inject knowledge into the reward function about potentially useful states, or state-actions. You can do this in multiagent systems to speed up learning without affecting the nash equilibria \cite{devlin2011theoretical}

This work extends difference rewards to tackling tightly coupled objectives. When you need many agents to stumble upon the correct joint state-action sequence, you use D++ to give agents stepping stone rewards for making progress towards hitting a coupling requirement even if you don't yet have enough agents. \cite{aida_2016}

This work focuses on difference rewards with policy gradients (multiagent reinforcement learning). this work combines difference rewards with policy gradients. (difference rewards). It also shows that it can learn a reward network to estimate difference rewards (similar to a fitness critic) 

This work focuses on potential based reward shaping with the caveat that the potential value of a state can now change over time. This works for both single agent and multiagent learning potential based reward shaping \cite{devlin2012dynamic}

This work focuses on extending EAs to the multiagent domain with CCEAs and also on how we can use difference evaluation functions to do so. \cite{agogino2004efficient}

This work focuses on multiagent path planning and we can auto generate potential based reward functions for multiagent path planning. \cite{devlin_kudenko_2016}

This work focuses on combining potential based rewards with difference rewards to leverage the benefits of both. (Difference rewards give you a way to assign credit to teammates from the team objectives and potential based rewards give you a way of injecting domain knowledge into your learning signal) \cite{devlin2014potential}

Difference objective functions are one approach 

- difference rewards
- potential based
- fitness critics (used in Golden's work and in Josh's teaming work)



(Reward shaping? Is it really "reward" shaping if it's evolutionary?)

- Talk about credit assignment methods. We should probably talk about D++ in more detail earlier, but we should also talk about difference utility functions/ difference rewards and how they can be used for credit assignment. Talk about what information you actually need to calculate this.

- Maybe reward shaping in general? Might not be super relevant. 

- Talk about D++ with some depth. Talk about how the whole point of D++ is to tackle tight coupling. It works by using counterfactual agents to simulate "what if" scenarios. Ex: What if I had 2 more agents helping me with this objective? What system feedback would I get? Then it uses that to calculate stepping stone feedback signals. Basically, if it takes you and another agent to accomplish the objective, but only you work on it, then you get 1/2 the feedback that you would get if you actually accomplished the objective.

This is really cool but it has two big downsides. The main one is that this becomes computationally expensive for high coupling and lots of agents. If your coupling is 10, and you have 20 agents, then for every agent, all 20 of them, you have to go back through their actions and at each step ask "What if I had 1 more, 2 more, etc up to 9 more agents here? What would the outcome be?" That means you're re running your simulation a whole lot of times to get the results of all of your counterfactual evaluations. That's going to eat up a whole lot of time and computational resources.

The other problem with it fundamentally is that it gives agents positive feedback even when the team didn't necessarily accomplish anything. This can reinforce useless behaviors. For example, say you have 2 agents, 2 pois, and a tight coupling requirement of 2. Ideally, you would have the 2 agents team up, and go to both pois together. Instead, what can happen with D++ is that both agents get split up between the 2 pois. Each agent goes to its own poi, and each agent gets a stepping stone reward for doing so, so they keep doing it. So each agent keeps getting positive feedback, and they either never learn or take a long time to learn that they're actually supposed to go to the POIs together. We want to get the agents to work together, and we don't want to incentivize that kind of splitting up.

Viewed another way, it's like these agents get caught in false local maximas for their feedback signals. They could get much better feedback if they cooperated, but this would involve enough deviation from their current policies that they're going to end up going back to their current policies before they explore enough to get to that better policy that would result in an even higher feedback signal.

- Talk about other approaches to learning in tightly coupled domains. Ie: Fitness critics. See if there's anything else from our lab that tackles this explicitly. Try to find papers from other labs that tackle tight coupling. Maybe make it clear what we mean when we say tight coupling vs what other papers mean when they talk about tight coupling. Maybe also talk about non learning based approaches to solving tightly coupled tasks



Words, words, words for Draft 2 Intro

GAP:
There is no multiagent learning for swarms. How do you do multiagent learning with a swarm? Solve complex objectives with a large swarm - this has not been done. Typically get swarm to a particular point with informative reward signal.

Prior methods:

Reward shaping has been done for small MAS with complex tasks - we have fitness critics, difference rewards, etc. It works for smaller systems
Swarm shepherding has been done for large swarms, but simple tasks. Use a learner to create a leader agent, copy paste that agent, then have it lead to swarm to a location (or sequence of locations?)

Why is this difficult?

in the team feedback signal makes 

When we give individual agents the system feedback signal for their performance, we are not actually giving each agent useful feedback on how that individual agent performed. What we want is to give each agent a feedback signal that captures that agent’s contribution to the team so that eah agent can maximize that individual learning signal which is responsive to that individual agent’s actions. (misalignment of system feedback with agent actions, clean learning signal, sensitive to agents’ actions, indivualized, incentivize agent to contribute to the team (specifically to split up tasks and not to help with tasks that other agents already have covered))

The noisiness in the system feedback signal 

The noisiness in the system feedback makes this signal insensitive to individual agents’ actions such that we cannot distill an agent’s individual contribution to the system’s feedback.

The noisiness from many agents’ interactions make it unclear which agents are actually contributing to the system’s performance, which 

An individual’s contribution to the team is not captured by the system feedback signal, which is what we actually want to incentivize the agents to learn
If agents are able to maximize the overall learning signal, sure the system should be able to perform well. However, the actual sensitivity of the team feedback signal is pretty low and noisy for a single agent, so it’s not clear how a agent policy actually maps to the system feedback signal
Alternatively, if we have some kidn of separation of different agents with different learning signals, then each agent’s learning signal is actually sensitive to that particular agent, making it so that the agent can actually learn now

lack of a clean learning signal makes it so that it is not clear which agents are actually contributing to the system’s performance. 

This is important because…. You can’t learn if your learning signal doesn’t actually incentivize you to complete your objective?

However, learning to coordinate a large number of independent agents is not trivial. The team feedback signal does not capture the performance of individual agents, making it difficult to give agent-specific feedback.


What is the problem?
learning to coordinate a large number of independent agents is challenging because the agents’ interactions create a noisy system feedback signal that is particularly difficult to learn from in environments with sparse and uninformative feedback.

The agents’ interactions together create a noisy system feedback signal that is difficult to learn from, particularly in environments with sparse and uninformative feedback signals.
each agent’s actions contribute to a noisy system feedback signal.

 is challenging because the tightl
Tightly coordinated actions between different agents interfere with each other’s learning signal and capabilities to accomplish tasks?
because the system feedback signal 
 captures the interactions between many agents, and doesn’t provide information about the performance of individual agents.

However, learning to coordinate a large number of agents is challenging because they receive a team feedback signal. This feedback signal doesn’t actually capture how well each agent did - it only captures the overall team performance. This could accidentally reinforce poor behaviors from teammates that aren’t actually doing anything helpful just because other agents did actually contribute to the team performance.

 the system feedback signal 

The problem is that their learning signals interfere with each other

Be broad about the messiness of the problem
Nick said something about how the problem is that they interfere with each other’s learning signals, making it difficul tot learn, so what I do is reduce the complexity of the problem by taking away from someo f that noise through the followers that act predictably
The important thing is that this gets even messier when you have LARGE numbers of agents

Why do I care?
it is too dangerous to send humans, communication is poor, or the system needs to react quickly to solve a problem.


 where the environment is too dangerous for humans, c

Multiagent systems are ideally suited for many domains including space exploration, ocean monitoring, and air traffic control. Cooperative Co-Evolutionary Algorithms (CCEAs) and Multiagent Reinforcement Learning (MARL) techniques offer new solutions to complex multiagent coordination problems.

Needs more:
Why do we want multiagent systems? What is advantageous about MAS?
Didn’t cover what the actual problem is
Need multiagent system because:
Communication fails
Cant send humans
Quick adaptation is critical
Focus on domains where leader follower makes sense
Explain domains in a way that you need many agents to solve them
Make a sell for learning systems with A LOT of agents

Domains 
Ocean Monitoring
	Multiple AUVs working together to collect observations
	Much faster than a single AUV, but multiple will have trouble communicating underwater
	Cant send humans - doesn’t make sense to send a bunch of humans in submarines for ocean monitoring/exploration
	You get broader coverage with multiple AUVs and don’t deal with as many communication problems if you use a multiagent system that doesn’t need constant communication 
	Followers are great for dealing with coupled observations - might need to get mulitple for observing living things, or maybe you want observations from multiple points of view, or maybe you want to cross validate observations between agents (if one agent’s view is obstructed, it doesn’t matter because you just use another agent’s view so it doesn’t matter if each agent doesn’t perform perfectly the whole time) - redundancy

Oil cleanup
	When there’s an oil spill, it covers a HUGE area and requires a FAST response
	One robot cleaning it is going to take too long and ultimately not accomplish much
	A swarm of robots cleaning will go much faster and clean a much broader area, being way more effective at actually cleaning up the oil

Disaster recovery / Search and rescue
	Need to respond quickly - multiple agents (robots?) may need to work together in order to actually get someone out of a dangerous situation - like lifting a heavy object that fell on someone and trapped them
	One agent is not enough - need to act fast, need to adapt, need broad coverage, need teamwork
	The more agents the better because you have more available agents to help with tasks and react to changing needs of changing situations - Swarm would be fantastic

Reasons for multiagent systems
Cant send humans. Too dangerous - send robots instead!
Communication is poor - need to decentralize decision making
Need to cover a large space - much faster with a multiagent system
Need to react quickly - multiple robots can cover an area and react to a problem much faster than a single robot









—--------------------------------------------------------------------------------------------------------------------------
The big difficult thing is figuring out how to assign credit to individual agents for their contribution to the team

We can do that with difference evaluations

The GAP is that when you extend those difference evaluations to tightly coupled domains, you start racking up high computation cost

Our method extends difference evaluations to tightly coupled domains without the additional computational cost

The cool thing about our approach is that we invert the problem to make it easier to solve. Whereas D++ goes in the front door and solves the problem the hard way with a high computational cost, we invert the problem to go in the back door and solve the problem the easy way with low computational cost.

The contribution of this paper is Multiagent Leader Based Learning, a multiagent learning framework for learning to coordinate large numbers of agents in tightly coupled domains.

The key results are that our method solves coordination problems in the tightly coupled rover domain much faster than D++, and can even solve problems that D++ cannot.

—
The global feedback signal does not contain information about the performance of individual agents or their contribution to that feedback signal. We need to distill that global feedback signal into individual agent feedback signals. This is important because we want agents to get rewarded for contributing to the team. If the team performs well overall, but one agent isn’t actually contributing anything to the team, then we don’t want to reward that one agent. We want that agent to learn how to contribute to the team.

Words, words, words
Cooperative Co-Evolutionary Algorithms (CCEAs) and Multiagent Reinforcement Learning (MARL) techniques offer new solutions to many of these challenges. However, to learn a solution for a tightly coupled task, multiple agents must first simultaneously stumble upon a correct sequence of joint-actions that completes that task. This is highly unlikely in a multiagent system where every agent is sampling random actions until the team receives a positive feedback signal for accomplishing the task.



Each agent makes an independent contribution to the team, but only receives feedback on the performance on the team as a whole, not its own contribution

Agents receive feedback on the performance of the team, 

In multiagent learning, the team receives feedback on the performance of the team, not individual agents. 

an agent must distill a global feedback signal G into an 

Cooperative The difficult thing here is that 

offer new solutions to many coordination problems. 

presents new challenges. 

Cooperative Co-Evolutionary Algorithms (CCEAs) and Multiagent Reinforcement Learning (MARL) techniques offer new solutions to many of these challenges. However, to learn a solution for a tightly coupled task, multiple agents must first simultaneously stumble upon a correct sequence of joint-actions that completes that task. This is highly unlikely in a multiagent system where every agent is sampling random actions until the team receives a positive feedback signal for accomplishing the task.

distill feedback into feedback for individual agents.

Many interactions between agents creates a noisy team feedback signal, so that one agent affects the feedback that another agent receives.

The team fitness does not 

The global fitness of a team does not provide information about the contribution of an individual agent to the team. Instead, the actions of one agent can significantly impact the feedback that another agent receives, creating a noisy feedback signal which is difficult to learn from.



 through D++ requires heavy computation. 

D++ is effective for learning to complete multiagent tasks with low coupling requirements, but requires exponentially more computations with each additional agent required to hit the coupling requirement. This is due to the requirement that D++ must compute one counterfactual outcome for every agent required to hit the coupling requirement. 


