[8:53 PM]kili: The way I was thinking of taking it was to frame it relative to a base ccea loop and mfl, where they are both at opposing sides of a spectrum of temporal abstraction. Then mfl struggles with overcoming situations where it's behaviors are insufficient for the task at hand. Asymmetric Island models is a balance between the two where the mainland effectively plans over more abstract behaviors than just ccea, but it also allows for incoporating new policies from the sub-islands. The issue here is that this sharing of information only occurs at the mainland, which delays the learning process as the updated policies only affect a different subisland after first going thorugh the mainland migration.
[8:54 PM]sirius: I'd like to get a better sense of the method. If memory serves me, you have islands where agents of each class learn some class-specific primitive behaviors in presence of other classes and then you migrate policies between islands occasionally. Right?
[8:55 PM]kili: So the inter-island migrations help speed up this information shairng. An additional benefit is that the policies each island sends to the mainland is also trained partially under the influence of the other types of agents in the system, allowing for policies that are better suited to a dependence.
[8:55 PM]kili: Yes, basically.
[8:55 PM]sirius: right. This makes sense
[8:56 PM]kili: It was originally just a constant time based schedule, but that resulted in regular decreases that you can still see, but it was alleviated slightly in the long run by making the schedule decaying.
[8:56 PM]kili: So first migrtations ia after 50, then 75, then 100, etc
[8:57 PM]sirius: I like this characterization of a balance between the two levels of abstraction. Are these results to support this? From the plots on overleaf it seems that island_inf does marginally better than mfl
[8:57 PM]kili: The mainland is actually closer to mfl than a normal ccea loop.
[8:58 PM]kili: Sometimes, yes. The issue mfl faces, specifically in this context, is that it seems too willing to go into hazardous regions, which slow how quickly it can move (the excavators can remove these areas).
[8:59 PM]kili: I think it's an artifact of the behaviors I gave mfl, since it's performance is based on the behaviors you actually give it.
[8:59 PM]kili: I was considering adding another plot with "bad" behaviors, to illustrate that mfl isnt just magic. It requires adequate behaviors to plan over. 
[8:59 PM]sirius: yeah. So what behaviors do you give mfl?
[8:59 PM]sirius: and do the island_inf model have these behaviors as well?
[9:00 PM]kili: So in a sense, mfl is almost a gold standard that I was hoping to meet.
[9:00 PM]kili: No, the island_inf models are all trained from scratch. Mfl is the only model with any preset behaviors.
[9:01 PM]sirius: (also btw: I am okay with both a call and text. I don't want it to seem that I want to be text-only. Either works for me just fine!)
[9:01 PM]sirius: oh I see I see
[9:01 PM]kili: The state is divided into regions around the rover (typical), and each region has a "layer". One layer for each type of object (harvesters, excvators, obstacles, pois). MFL is then given behaviors that either push it towards, or away from, the densest region of a layer.
[9:02 PM]kili: So with 4 regions around the rover, and 4 layers, that's 32 behaviors
[9:02 PM]kili: wait, my math is bad
[9:02 PM]kili: 8
[9:02 PM]sirius: no thats right. 4x4x2
[9:03 PM]kili: no, yeah, 32
[9:03 PM]sirius: well actually. if you have two behaviors per object, that would be 4x2
[9:04 PM]sirius: Does mfl need one behavior per quadrant per object?
[9:04 PM]kili: yes
[9:05 PM]kili: no, yeah, its 8
[9:05 PM]kili: each layer, then it finds the densest regions, it goes in that direction
[9:06 PM]kili: it's two behaviors for each object. one to go towards the densest region, one to go away from the densest region 
[9:07 PM]sirius: so those are pre-trained behaviors given to mfl
[9:07 PM]kili: hard coded, yes
[9:07 PM]sirius: and then what are the rewards used for the island-inf model?
[9:07 PM]kili: They arent networks. Just a computation
[9:08 PM]sirius: oh I see
[9:08 PM]kili: The rewards are based on the type of agent on the subislands. On the mainland, it's the total value of the pois collected.
[9:08 PM]kili: On the harvester subislands, its the value of the pois collected.
[9:08 PM]kili: On the excavator, its the value of the regions removed (essentially hte regions acts like pois for excavators)
[9:09 PM]kili: The harvesters are also penalized for going into a hazardous region.
[9:09 PM]kili: I was planning on adding a penalty for excavators going over pois (like a truck running over a crop), but that did not happen
[9:10 PM]kili: There is no coupling requirement in the tests, but the code supports it. All the pois just have a coupling of one.
[9:11 PM]kili: Interestingly enough, the excavators actually learn more effective policies faster than the harvesters.
[9:11 PM]kili: On the subislands.
[9:12 PM]kili: I think it's because the excavators are making room for the harvesters, early on. So the excavators making progress on their task means that the harvesters are better able to accomplish their task.
[9:13 PM]kili: So it's less that that excavators learn faster (though this is still true), and more that it comes from the harvesters have to almost wait for the excavators to learn something useful before they can starting making effective progress on learning.
[9:13 PM]kili: I saw this most when I scaled up the number of harveseters in the environment, or when I increased the penalty for colliding with a hazard.
[9:14 PM]sirius: neat!
[9:14 PM]sirius: yeah that makes sense
[9:14 PM]kili: In these cases, mfl didnt even seem to learn that much faster than island_inf (though still a bit faster). But just islands was pretty slow, though it did get up to a decent performance in the end.
[9:15 PM]sirius: might want to just deploy a run on a server with some coupling. Not necessarily needed I don't think, but definitely good to have.
[9:16 PM]kili: It's also kind of important that islands and mfl is almost a bastardization of the original implementations. I tried to take the core /skeleton of each design to try and make the comparisons as fair as possible, because a lot of the backbone is shared between them.
[9:16 PM]kili: So i'm comparing less the overall algorithm, and more the architectural design of how learning/information flows between populations. 
[9:17 PM]kili: I want to have some coupling results for aamas (or hopefully the presentation), but for the immediate paper, I dont think so.
[9:19 PM]sirius: I think the concern without coupling is that would learning be a problem if you were to take the rewards you give on the islands and directly using policy gradients to train agents on the main team task?
[9:20 PM]sirius: Since agents can learn to independently learn their class-speicifc behaviors (kinda), would it be a difficult problem to learn for just agents learning using PPO?
[9:21 PM]kili: I'm less concerned with if their is a better approach for this specific problem, and more so about exploring how to fix up some issues present in these approaches
[9:21 PM]kili: there's the easy arguement that policy gradients needs..well, a gradient
[9:22 PM]sirius: I think just adding that comparison where you have the exact same rewards for the classes but no island model (so effectively both classes learn directly on the team task using PPO), would help you to get away with needing to show any coupling requirement
[9:22 PM]kili: and that rl in general is a lot more suited to learning from local rewards, where the gradient is more meaningful (and accurate)
[9:22 PM]kili: eas are more suited to overall beahviors, but mfl doesnt really allow for altering those "behaviors"
[9:23 PM]sirius: right but the rewards on the islands (values of collected pois and removing regions) are dense enough to use gradient methods, right?
[9:23 PM]kili: in truth, the optimization on each island isn't necessarily tied to ccea, and it's likely that a mix of different techinques, each suited to particular learning, would work best
[9:23 PM]kili: like ppo/trpo/maddpg on the subislands, and an mfl-esque ea on the mainland
[9:25 PM]kili: That is true too, though at least currently, the experiments are not set up that way. I'm not actually sure that it would be all that effective.
[9:25 PM]kili: I'd probably want to combine the sub-task rewards in some way, like a weighted sum of the pois and obstacles collected.
[9:26 PM]kili: Otherwise, the learning signal might just be too nebulous or noisy for the excavators to learn something.
[9:30 PM]sirius: well no I was thinking that you could keep those two rewards separate just like before. The excavators will use ppo to learn to maximize the obstacles-collected reward and the harvesters will learn to maximize the pois-collected reward using ppo. Both classes learn together (using their own class-specific reward) directly on the team task
[9:30 PM]kili: Oh, I get what you mean.
[9:31 PM]kili: I would be concerned with the excavators learning something that isn't necessarily beneficial to the team however, like possibly excavating a large region of obstacles that arent actually impeding any pois
[9:31 PM]kili: in the current setup, that wouldnt actually happen because of how obstacles and pois are distributed
[9:31 PM]sirius: I would be very curious to see that, because I think without coupling, those rewards might be dense enough for a gradient method to potentially learn. The exception being that if the policies trained with island-inf learn some kind of interesting inter-agent behaviors, then they would blow ppo out of the water (or however that phrase goes)
[9:32 PM]kili: Honestly, I think the environment just isnt complicated enough for that.
[9:32 PM]sirius: right but that's not a bad thing if your experiments show that PPO fails to learn here but island-inf learns!
[9:33 PM]kili: oh, I see. Yeah. That could be an interesting thing to look at.
[9:33 PM]sirius: I guess then that is the question that a reviewer might ask. Why do I consider using your approach when current methods are able to solve this problem just fine?
[9:34 PM]kili: the frank answer would be I dont care, that's not the point
[9:36 PM]kili: more eloquent would be that current methods, in the realm of eas, either require carefulyl constructed behaviors, shaping (which requires the functional shape of the reward), or might be inefficient to learn due to a mix between the non-stationarity inherent in mas, or bexcause in attempting solving this issue, they overly restrict the flow of learned information.
[9:37 PM]kili: the environment then is not an all encompasing baseline, but a testbed to illustrate key differences between the approaches 
[9:38 PM]kili: the key takeaway isn't "use this architecture with these parameters", but that intermittently learning in the presence of updated learning from others can be beneficial to realizing inter-dependent behaviors, without the need for domain knowledge or designing these behaviors a priori 
[9:43 PM]kili: another benefit here would also be that a lot of the components are rather plug-and-play. The overall architecture is not overly dependent on too many high-level design decisions (such as the optimizers on each island, or the population sizes, or the tasks). It (seems to me) like it would be easier to build out and scale a learning system using a architecture based on this design, but that's not really a hard-supported result. More just musings about the implementation and how you might be able to leverage more computionanal resources more easily.
[9:44 PM]kili: And then you can pick out each component and work on it (somewhat) independently, but you dont lose all the benefit of multiagent learning (vs independent learning).
[9:48 PM]sirius: This is an interesting idea btw that I have been wanting to test. using this island like architecture as a way to aligning several different optimization methods to a single team fitness
[9:48 PM]kili: yeah, I think it's a natural branching off point
[9:49 PM]kili: I've been just thinking of some ways that it might be useful and trying to get notes down as they come up 
[9:49 PM]sirius: yeah I can buy that. But
without the need for domain knowledge or designing these behaviors a priori 
It would be perfect if we could show some complex behaviors learnt using this island-inf that cannot be easily defined a priori.
[9:50 PM]sirius: I agree, but we will def have to back those claims with some results
[9:50 PM]kili: the other variation (that I think weve discussed before) is that an island isnt also just tied to optimizing agents. Even in a pretty vanialla version, you could use it to optimize more types of processes or dynamics, such as an env or even the reward itself.
[9:50 PM]kili: yeah...
[9:51 PM]kili: yeah
[9:51 PM]kili: at least for me, anytime I saw it do something, it was just like ok, that makes sense.
[9:51 PM]kili: and then it didnt seem like it would be that hard to hardcode
[9:52 PM]kili: but I think that's partly because the env is so simple in its dynamics, that there really isn't a whole lot of actually interesting behaviors it could learn
[9:54 PM]kili: but that ofc, is predicated on that my assumption there isn't anything particualrly interesting (behavior-wise) to learn is true. But if it was interesting to learn, I probably wouldnt be able to envision them, and the behaviors I saw may only be a subset or in an entirely different region of the state-action space than where the interesting behaviors are
[9:55 PM]kili: thanks for the chat, btw. Just chatting and the questions helped me at least start articulating some points. 
[9:56 PM]sirius: yeah that is quite tricky. Its hard to tell if the environment is simple enough to not require interesting behaviors or if it potentially can generate interesting behaviors that we just aren't looking for
[9:58 PM]sirius: I think what you have so far is definitely a good place to be at. For aamas, results with coupling, or comparisons to ppo or other evo methods, or using different optimization methods on the islands... any of these will help strengthen your points
[9:59 PM]kili: for other evo methods, what might you be thinking?
[9:59 PM]kili: I was already trying to make the env compatible with gym, so hopefully one of the more tested implementations would work
[10:00 PM]kili: it's like 80% maybe ready to act as a pettingzoo custom env, but I havent actually tested it 
[10:01 PM]sirius: For now, would it be possible to test performance on this environment without introducing the presence of other agent classes on the islands? so keep the rewards on the islands the same, the mainland rewards the same. The only thing that changes is that each island only has its own agent class with no other classes to influence it. If that is possible to do, then you could use that as a gap in the current island model architectures that your method addresses.
[10:02 PM]sirius: If running those tests isn't an option for now, then we might want to rethink how we re-contextualize Kagan's 8 questions to present this work as more of a study than fixing a previous gap
[10:02 PM]sirius: Will need to think about this a bit more
[10:02 PM]kili: I feel like that shouldnt be an issue. For one, I expect the harvester island to just take forever to learn anything, and it would actually be dragging back the learning since it would almos always be lagging behind the mainland
[10:03 PM]kili: there are two deadlines at play too
[10:03 PM]kili: the masters document and the aamas paper
[10:03 PM]kili: we have a little more than an additional month for the aamas deadline 
[10:08 PM]sirius: honestly that would be a great result to show that a gap clearly exists and your method is able to address it! 
[10:09 PM]sirius: yeah focus on the masters document for sure. You have good results to go off on and describe this as more of a study. After the masters deadline, we can brainstorm more and see what we can do for aamas