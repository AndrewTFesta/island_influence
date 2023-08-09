Survey of Current Approaches to Fitness Shaping in Multiagent Evolutionary Learning
=====

# Index

- [Roadmap](#roadmap)
- [Multiagent Reliance-Based Learning State](#multiagent-reliance-based-learning)
  - [Introduction](#introduction)
  - [Background](#background)
  - [Approach](#approach)
  - [Results](#results)
  - [Conclusion](#conclusion)
  - [Future Work](#future-work)
- [Environments](#environments)
  - [HarvestEnv](#harvestenv)
- [Agents](#agents)
  - [Harvester](#harvester)
  - [Excavator](#excavator)
  - [Obstacle](#obstacle)
  - [PoI](#poi)
- [Rewards](#rewards)
- [CCEA](#ccea)
  - [Mutation](#mutation)
  - [Selection](#selection)

# How to write a survey paper

https://www.researchvoyage.com/how-to-write-better-survey-papers/
https://academia.stackexchange.com/questions/43371/how-to-write-a-survey-paper
https://academia.stackexchange.com/questions/150773/is-it-worth-writing-and-publishing-a-survey-paper-while-a-phd-student
https://academia.stackexchange.com/questions/183553/what-exactly-is-a-survey-article-review-article-in-research
https://academia.stackexchange.com/questions/167708/what-are-benefits-of-writing-a-survey-paper
https://academia.stackexchange.com/questions/195960/potential-plagiarism-of-published-papers-when-writing-survey-paper
https://essaypro.com/blog/how-to-write-a-survey-paper-brief-overview
http://www.cs.ucf.edu/~lboloni/Teaching/EEL6788_2008/slides/SurveyTutorial.pdf
https://gradebees.com/writing-survey-paper/
https://blogs.ubc.ca/cpen542/term-paper/survey-paper/
https://phdservices.org/survey-paper/
https://www.researchgate.net/post/How_to_write_a_survey_paper_in_computer_science_related_topics

# Introduction

Reward shaping is a heuristic for faster learning. Generally, it is a function $F(s,a,s′)$ added to the original reward function $R(s,a,s′)$ of the original MDP. Ng et al. proved that if the shaping function is based on a state-dependent potential, i.e. $F(s,a,s′)=γϕ(s′)−ϕ(s)$, then the optimal policy for the new MDP with $R+F$ as the reward function is still optimal for the original MDP.

To give an example where reward-shaping is immediately useful, you can imagine a scenario where you have an existing heuristic function for a MDP problem and you want to incorporate that somehow into a MDP solver. You can think of this heuristic as being a rough approximation to the optimal value function V
. Reward-shaping then gives us a natural way to incorporate that heuristic into the solver: all we have to do is use that heuristic as the potential function for defining the reward-shaping function. This allows us to distribute the rewards more evenly (and accurately) across the state-space, and potentially allow for better solutions (and quicker convergence) in, say, anytime algorithms.

Aside: I recently co-authored a paper related to this subject with my research group, so I’m very happy to see that someone out there is actually still interested in this sort of thing!

# Background
