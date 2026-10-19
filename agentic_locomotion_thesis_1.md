# Agentic Reward and Controller Synthesis for Prompt-Specified Legged Locomotion on Commodity Hardware

**Working thesis document: Preliminaries, Introduction, Literature Review, and Methods**

*Draft prepared October 2026*

---

## Contents

- 0. Preliminaries: A Ground-Up Primer
  - 0.1 How to read this chapter
  - 0.2 Robots as mechanical systems
  - 0.3 Describing position and orientation
  - 0.4 Physics simulation and MuJoCo
  - 0.5 The Gymnasium Ant robot in detail
  - 0.6 Control: from torques to gaits
  - 0.7 Reinforcement learning
  - 0.8 Reward design, shaping, and reward hacking
  - 0.9 The software stack: Gymnasium and Stable-Baselines3
  - 0.10 Large language models and agents
  - 0.11 Experimental methodology concepts
  - 0.12 Glossary
- 1. Introduction
- 2. Literature Review
- 3. Methods
- References
- Appendix A. Prompt templates
- Appendix B. Repository layout and run commands

---

## 0. Preliminaries: A Ground-Up Primer

### 0.1 How to read this chapter

This chapter is written for a student who has some programming experience (ideally Python) and some exposure to calculus and linear algebra, but no prior background in robotics, reinforcement learning, or large language model agents. It builds the vocabulary used throughout the rest of the document in the order in which the ideas depend on one another: first the physical robot, then how it is simulated, then how it is controlled, then how controllers can be learned, then how the learning objective (the reward) is designed, and finally how language-model agents can be put in charge of designing that objective. Each subsection ends with the specific takeaway that matters for this thesis. A glossary at the end of the chapter collects every technical term in one place.

The single most important idea to carry forward is this: in our pipeline, a language-model agent never moves the robot's legs directly. Instead, it writes and edits two kinds of files, a *reward file* and a *controller file*, which together define a learning problem. A standard reinforcement learning algorithm then solves that learning problem in a physics simulator, producing a small neural network that does move the legs. A separate, fixed evaluator measures whether the resulting behavior matches what the user asked for, and the agent uses that measurement to revise its files. Everything in this chapter exists to make that loop understandable.

### 0.2 Robots as mechanical systems

**Rigid bodies and links.** A robot is modeled as a collection of *rigid bodies*, which are idealized solid objects that do not bend or deform. In robotics these bodies are called *links*. A leg might consist of an upper link (a thigh) and a lower link (a shin). Each link has a mass, a center of mass, and an *inertia tensor*, which describes how hard it is to rotate the link about different axes.

**Joints and degrees of freedom.** Links are connected by *joints*, which constrain how one link can move relative to another. The most common joint in legged robots is the *hinge* (or *revolute*) joint, which allows rotation about a single axis, like an elbow. Each hinge contributes one *degree of freedom* (DOF), meaning one independent number is needed to describe its state (its angle). A *free joint* places no constraint at all; a body attached to the world by a free joint can translate in three directions and rotate about three axes, giving six degrees of freedom. The torso of a walking robot is attached to the world by a free joint, because nothing bolts it to the ground.

**Configuration and generalized coordinates.** The complete description of where every part of the robot is at an instant is called its *configuration*, and the list of numbers that encodes it is called the vector of *generalized coordinates*, conventionally written $q$. Its time derivative, $\dot q$, is the vector of *generalized velocities*. For a legged robot, $q$ contains the position and orientation of the torso plus the angle of every joint.

**Actuators and torque.** Joints move because *actuators* (motors) apply *torque*, which is the rotational analogue of force, measured in newton-metres (N·m). A joint with a motor is *actuated*; a degree of freedom with no motor is *unactuated*. A legged robot is *underactuated* in an important sense: the six degrees of freedom of the torso have no motor. The robot can only move its torso by pushing against the ground with its feet. This is the root of why walking is a hard control problem.

**Kinematics versus dynamics.** *Kinematics* is the geometry of motion: given joint angles, where is the foot? *Dynamics* is the physics of motion: given torques and contact forces, how do the joint angles and torso accelerate? The dynamics of a rigid-body robot are summarized by the *equations of motion*,

$$M(q)\,\ddot q + C(q,\dot q) + g(q) = S^\top \tau + J_c(q)^\top f_c ,$$

where $M(q)$ is the mass matrix, $C(q,\dot q)$ collects velocity-dependent (Coriolis and centrifugal) effects, $g(q)$ is gravity, $\tau$ is the vector of motor torques, $S$ selects which coordinates are actuated, and $J_c^\top f_c$ maps contact forces $f_c$ at the feet into the generalized coordinates. You will not need to manipulate this equation in this thesis, because the simulator solves it for us, but it explains three facts that matter later. First, the torso only moves through the contact term, so if a foot is not touching the ground it cannot help propel the body. Second, contacts switch on and off as feet lift and land, which makes walking a *hybrid* system with discrete mode changes, a major reason why classical control of legged robots is difficult. Third, friction limits how much horizontal force a foot can apply before it slips, so a policy that commands large torques may simply skid.

**Takeaway.** A walking robot is an underactuated, contact-rich system. Good walking emerges from timing foot contacts well, which is why both the reward and the controller structure in our pipeline pay explicit attention to foot contacts.

### 0.3 Describing position and orientation

**Frames.** A *coordinate frame* is an origin plus three perpendicular axes. The *world frame* is fixed to the ground. The *body frame* is attached to the robot's torso and moves with it. "Forward velocity" in a reward function can mean velocity along the world's $x$-axis or along the direction the robot is facing; confusing the two is a classic bug, and our metric library defines both explicitly.

**Orientation.** The orientation of the torso can be described by three angles, *roll* (tilting sideways), *pitch* (tilting nose up or down), and *yaw* (turning left or right), but these angles behave badly in some configurations. Simulators therefore store orientation as a *unit quaternion*, a four-number representation $(w, x, y, z)$ with $w^2+x^2+y^2+z^2=1$. You can convert a quaternion into a rotation matrix, from which useful quantities fall out easily. The most useful one for locomotion is the *up-vector*: the direction the torso's local $z$-axis points in the world. If the $z$-component of the up-vector is close to 1, the robot is upright; if it is negative, the robot has flipped onto its back.

**Takeaway.** Many reward terms and evaluation metrics (uprightness, heading, body-frame velocity) are computed from quaternions. Our pipeline supplies the agent with helper functions so it does not have to re-derive quaternion algebra, a common source of errors in language-model-written code.

### 0.4 Physics simulation and MuJoCo

**Time stepping.** A physics simulator advances the world in small increments of time called *timesteps*. At each timestep it computes contact forces, solves the equations of motion, and integrates to get new positions and velocities. Smaller timesteps are more accurate and more stable but cost more computation.

**Control frequency and frame skip.** A controller usually does not choose a new action at every physics timestep. Instead, it chooses an action, the simulator holds that action for several physics steps, and then the controller observes again. The number of physics steps per control step is the *frame skip*. If the physics timestep is 0.01 s and the frame skip is 5, the controller acts every 0.05 s, which is a *control frequency* of 20 Hz.

**MuJoCo.** MuJoCo (Multi-Joint dynamics with Contact) is a physics engine designed for robotics and control research (Todorov, Erez & Tassa, 2012). It is now open source, maintained by Google DeepMind, distributed as a Python package (`pip install mujoco`), and runs efficiently on ordinary CPUs, including Apple Silicon. Robots are described in an XML format called *MJCF*, which lists bodies, joints, geometric shapes used for collision (*geoms*), actuators, and sensors. At runtime MuJoCo separates the *model* (`mjModel`, the fixed description: masses, joint limits, actuator gains) from the *data* (`mjData`, the changing state: `qpos`, `qvel`, contact list, sensor readings). This separation is convenient for us: reward and metric code reads from `mjData` without modifying anything.

**Determinism and seeds.** Given the same initial state and the same actions, MuJoCo produces the same trajectory. Randomness in experiments comes from random initial states, random exploration noise in the learning algorithm, and random network initialization, all of which are controlled by *random seeds*. Recording seeds is essential for reproducibility.

**Simulation versus reality.** A simulator is an approximation. Policies trained only in simulation often fail on real hardware, a problem called the *sim-to-real gap*. This thesis deliberately stays in simulation; we note the gap only to delimit the scope of our claims.

**Takeaway.** MuJoCo gives us a fast, deterministic, CPU-friendly "verifier" for every candidate reward and controller the agent proposes. Cheap, trustworthy evaluation is the property that makes the agentic loop feasible on a laptop.

### 0.5 The Gymnasium Ant robot in detail

The robot studied in this thesis is *Ant*, a simulated quadruped distributed with the Gymnasium library. It originates from the generalized advantage estimation paper of Schulman et al. (2016) and has been a standard continuous-control benchmark for a decade. The current version is `Ant-v5`.

**Morphology.** Ant has a spherical torso with four legs radiating outward. Each leg consists of two links joined to the torso and to each other by hinge joints: a *hip* joint (rotating the leg in the horizontal plane) and an *ankle* joint (raising and lowering the lower link). There are therefore eight actuated hinges in total, plus the free joint of the torso. According to the official documentation, the robot is about 0.75 m tall and weighs roughly 0.91 kg.

**State vector.** MuJoCo's `qpos` for Ant has 15 entries: 3 for torso position $(x,y,z)$, 4 for the torso quaternion, and 8 joint angles. `qvel` has 14 entries: 3 linear velocities, 3 angular velocities, and 8 joint angular velocities. (The quaternion's four numbers correspond to three rotational velocities, which is why `qvel` is one shorter than `qpos`.)

**Observation.** By default the observation given to the learning agent excludes the torso's $x$ and $y$ position (so that the learned behavior does not depend on where in the world the robot happens to be) and contains the remaining 13 position entries, the 14 velocities, and 78 numbers describing external contact forces on each body, for a total of 105 dimensions. The contact-force entries can be switched off with a constructor argument, leaving a 27-dimensional *proprioceptive* observation (information the robot could sense about its own body).

**Action.** The action is an 8-dimensional vector in $[-1,1]^8$, each entry scaled to a torque on one hinge.

**Default reward.** Ant-v5's built-in reward is the sum of a *healthy reward* (a constant 1 per step while the robot is upright within a height range), a *forward reward* (velocity of the torso along world $x$), minus a *control cost* (0.5 times the squared norm of the action) and a *contact cost* (a small penalty on large external contact forces). This reward produces forward running. Our pipeline replaces it with agent-designed rewards.

**Episodes.** The physics timestep is 0.01 s with a frame skip of 5, so each control step lasts 0.05 s (20 Hz). An *episode* lasts at most 1000 control steps (50 simulated seconds). By default it ends early (*terminates*) if the torso height leaves the interval [0.2, 1.0] m, which catches falls but also makes behaviors such as "lie down" impossible unless termination is reconfigured. This is one reason our controller file is allowed to adjust termination settings within safe bounds.

**Takeaway.** Ant is simple enough to train in minutes to hours on a CPU, yet rich enough (eight motors, four contacts, a free-floating torso) to exhibit distinct gaits, turning, lateral motion, posture tasks, and failure modes such as flipping over. It is the ideal testbed for a laptop-scale study.

### 0.6 Control: from torques to gaits

**Open-loop versus closed-loop control.** An *open-loop* controller produces actions without looking at the robot's state, for example a fixed rhythmic pattern of torques. A *closed-loop* (or *feedback*) controller measures the state and adjusts its actions accordingly. Feedback makes control robust to disturbances: if the robot starts to tip, a feedback controller can respond.

**PD control.** The workhorse feedback law in robotics is *proportional-derivative* (PD) control. Given a *target* joint angle $q^\ast$, the motor torque is

$$\tau = k_p\,(q^\ast - q) - k_d\,\dot q ,$$

where $k_p$ (the proportional gain, or *stiffness*) pulls the joint toward the target and $k_d$ (the derivative gain, or *damping*) resists fast motion. With PD control in place, a higher-level controller only needs to output target angles rather than raw torques, which is often easier to learn. The choice between *torque actions* and *position-target actions* is a design decision that the agent in our pipeline is permitted to make in the controller file.

**Gaits.** A *gait* is a repeating pattern of leg motion. For a quadruped, gaits are characterized by the *duty factor* (the fraction of each cycle a foot spends on the ground) and the *phase offsets* between legs. In a *trot*, diagonal legs (front-left with back-right, front-right with back-left) move together. In a *pace*, legs on the same side move together. In a *bound*, the two front legs move together and the two back legs move together. In a *pronk*, all four legs move together, producing a hop. In a four-beat *walk*, each leg is offset by a quarter cycle. A contact pattern can be written as a binary time series per foot (1 = touching the ground, 0 = in the air), an idea exploited by SayTap (Tang et al., 2023) and used in our metric library.

**Central pattern generators.** Animals generate rhythmic locomotion partly through neural circuits in the spinal cord called *central pattern generators* (CPGs) (Ijspeert, 2008). Roboticists imitate them with coupled oscillators. A common model gives each leg $i$ an oscillator with amplitude $r_i$ and phase $\varphi_i$:

$$\dot r_i = a\,(\mu - r_i^2)\,r_i, \qquad \dot\varphi_i = \omega + \sum_j r_j\,c_{ij}\,\sin(\varphi_j - \varphi_i - \Phi_{ij}).$$

The amplitude converges to $\sqrt{\mu}$, the phase advances at frequency $\omega$, and the coupling terms pull the legs into the desired phase offsets $\Phi_{ij}$, which encode the gait. The oscillator outputs are mapped to joint targets, for instance hip target $=\theta_{\text{hip}}^0 + A_{\text{hip}}\cos\varphi_i$ and ankle target $=\theta_{\text{ankle}}^0 + A_{\text{ankle}}\max(0,\sin\varphi_i)$ so that the lower leg lifts only during the swing half of the cycle. A CPG is a compact, interpretable controller with a handful of parameters (frequency, amplitudes, offsets, phase lags). It can be used open-loop, tuned by black-box optimization, or combined with a learned neural network that adds *residual* corrections (Bellegarda & Ijspeert, 2022). CPG parameters are exactly the kind of "controller file" content an agent can reason about in natural language ("increase the frequency, switch to a trot").

**Hierarchical control.** Complex behaviors are usually organized in layers running at different speeds: a fast low-level controller (tens to hundreds of hertz) keeps the robot balanced and tracks commands such as "walk forward at 0.5 m/s, turn left at 0.3 rad/s", while a slow high-level controller (around 1 Hz or slower) decides which command to issue based on goals and sensory input. This layering is how our pipeline lets a language model respond to stimuli without having to control joints in real time.

**Takeaway.** The "controller file" in our pipeline chooses the structure of the low-level controller: raw torques, PD position targets, a CPG with learned residuals, or a pure open-loop CPG. Structure strongly shapes what is easy to learn.

### 0.7 Reinforcement learning

**The agent-environment loop.** *Reinforcement learning* (RL) is a framework in which an *agent* learns by trial and error (Sutton & Barto, 2018). At each time $t$ the agent observes a state $s_t$, chooses an action $a_t$, and the *environment* responds with a new state $s_{t+1}$ and a scalar *reward* $r_t$. (Note the overloaded word: in RL, the "agent" is the learning controller, a small neural network. In the rest of this thesis we call it the *policy* to avoid confusion with *language-model agents*.)

**Markov decision processes.** Formally the problem is a *Markov decision process* (MDP), a tuple $(\mathcal S, \mathcal A, P, R, \gamma)$: states, actions, transition probabilities $P(s_{t+1}\mid s_t,a_t)$, a reward function $R(s_t,a_t,s_{t+1})$, and a *discount factor* $\gamma\in[0,1)$. The *Markov* property means the next state depends only on the current state and action, not on the earlier history.

**Policies and returns.** A *policy* $\pi_\theta(a\mid s)$ is a rule for choosing actions, here a neural network with parameters $\theta$ that outputs a probability distribution (typically a Gaussian) over actions. A *trajectory* is a sequence of states, actions, and rewards. The *return* is the discounted sum of rewards from time $t$ onward, $G_t = \sum_{k\ge 0}\gamma^k r_{t+k}$. The discount makes near-term rewards count more than distant ones; with $\gamma=0.99$, rewards about 100 steps (5 seconds for Ant) in the future still matter substantially. The learning objective is to find parameters that maximize the expected return, $J(\theta)=\mathbb E_{\pi_\theta}[G_0]$.

**Value functions and advantages.** The *state-value function* $V^\pi(s)$ is the expected return starting from state $s$ under policy $\pi$; the *action-value function* $Q^\pi(s,a)$ is the expected return after taking action $a$ in $s$. The *advantage* $A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$ says how much better an action is than the policy's average behavior. Most modern algorithms learn an estimate of $V$ or $Q$ (the *critic*) alongside the policy (the *actor*), a design called *actor-critic*.

**Policy gradients and PPO.** *Policy gradient* methods adjust $\theta$ in the direction that makes high-advantage actions more likely. *Proximal Policy Optimization* (PPO) (Schulman et al., 2017) is the most widely used variant. It collects a batch of experience, then optimizes the clipped objective

$$L^{\text{CLIP}}(\theta)=\mathbb E_t\Big[\min\big(\rho_t(\theta)\hat A_t,\ \operatorname{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)\hat A_t\big)\Big],\qquad \rho_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)},$$

which improves the policy while preventing any single update from changing it too drastically. Advantages $\hat A_t$ are usually computed with *generalized advantage estimation* (GAE) (Schulman et al., 2016). PPO is *on-policy*: it learns only from data collected by the current policy and discards data after each update. It is robust and parallelizes well across multiple simulator copies.

**SAC.** *Soft Actor-Critic* (SAC) (Haarnoja et al., 2018) is *off-policy*: it stores past experience in a *replay buffer* and reuses it many times, which makes it more *sample-efficient* (it needs fewer simulator steps) but more computationally expensive per step. SAC maximizes return plus an *entropy* bonus that keeps the policy exploratory: $J(\pi)=\sum_t\mathbb E\big[r_t+\alpha\,\mathcal H(\pi(\cdot\mid s_t))\big]$. On a CPU, the trade-off between PPO's cheap-but-many steps and SAC's expensive-but-few steps must be measured, not assumed, which is why our methods begin with a calibration phase.

**Termination and truncation.** An episode can end because something meaningful happened (the robot fell: *termination*) or because a time limit was reached (*truncation*). The distinction matters: after termination there is no future reward, but after truncation the future would have continued. Gymnasium reports the two separately.

**Normalization.** Neural networks learn best when inputs have roughly zero mean and unit variance. Running *observation normalization*, and often *reward normalization*, is standard practice for MuJoCo tasks.

**Variance across seeds.** Two RL runs that differ only in random seed can produce very different results (Henderson et al., 2018). Any claim about whether one reward function is better than another must therefore be supported by multiple seeds and appropriate statistics (Section 0.11).

**Takeaway.** In our pipeline the RL algorithm is a fixed, trusted component, a "compiler" that turns a reward file and a controller file into a trained policy. The language-model agent designs the inputs to this compiler; it does not modify the compiler itself except through a small whitelist of hyperparameters.

### 0.8 Reward design, shaping, and reward hacking

**The reward design problem.** The reward function is the only way to tell an RL algorithm what we want. Singh, Lewis & Barto (2010) formalized the *reward design problem*: the designer has some true measure of success (a *fitness function*), but optimizing that measure directly may be difficult, so the designer chooses a *training reward* such that a policy trained on it scores well on fitness. The reward is a means to an end; fitness is the end. This distinction is the conceptual backbone of our pipeline, and also of Eureka (Ma et al., 2024).

**Sparse versus dense rewards.** A *sparse* reward is nonzero only when the goal is achieved (e.g., +1 when the robot reaches a target). Sparse rewards are faithful but hard to learn from, because random exploration rarely stumbles onto success. A *dense* reward gives feedback at every step (e.g., reward proportional to forward velocity). Dense rewards learn faster but can encode the wrong thing.

**Reward components.** Locomotion rewards are almost always sums of weighted components: a task term (track a velocity), regularizers (penalize energy, jerky actions, torso tilt), and survival terms (a bonus for staying upright). The *relative scales* of components matter as much as their form. If an upright bonus of 1 per step dwarfs the velocity term, the policy may learn to stand still.

**Reward shaping.** Adding terms to guide learning is called *shaping*. Ng, Harada & Russell (1999) proved that shaping terms of the special form $F(s,s')=\gamma\Phi(s')-\Phi(s)$, for any function $\Phi$ of state, never change which policy is optimal. Most practical shaping is not of this form and can therefore change the optimal behavior, for better or worse.

**Reward hacking.** When a learned policy maximizes the reward in a way the designer did not intend, it is called *reward hacking* or *specification gaming* (Amodei et al., 2016; Skalse et al., 2022). Classic locomotion examples include a robot that moves forward by flipping onto its back and kicking, or one that earns an "alive" bonus by refusing to move. Reward hacking is the central risk of automated reward design, because an agent that is rewarded for raising the training reward may unwittingly write rewards that are easy to exploit.

**Takeaway.** Our pipeline separates the *training reward* (written and edited by the agent) from the *evaluation fitness* (compiled from the user's specification, approved by a human, and frozen before any search begins). The agent can see fitness scores but cannot edit the evaluator. This separation is our primary defense against reward hacking.

### 0.9 The software stack: Gymnasium and Stable-Baselines3

**Gymnasium.** Gymnasium (Towers et al., 2024), the maintained successor to OpenAI Gym (Brockman et al., 2016), defines a standard interface for RL environments. An environment is created with `gymnasium.make("Ant-v5", **kwargs)`. Calling `env.reset(seed=...)` returns an initial observation and an `info` dictionary; calling `env.step(action)` returns a 5-tuple `(observation, reward, terminated, truncated, info)`.

**Wrappers.** A *wrapper* is an object that sits around an environment and modifies its behavior without changing its code. A *reward wrapper* replaces the reward; an *action wrapper* transforms actions (for example, converting position targets into PD torques); an *observation wrapper* adds or removes observation entries. Our entire pipeline is implemented as wrappers around the unmodified `Ant-v5`, which keeps the simulator itself identical across all experiments.

**Vectorized environments.** Running several copies of an environment in parallel, either in one process (`SyncVectorEnv`/`DummyVecEnv`) or in separate processes (`AsyncVectorEnv`/`SubprocVecEnv`), increases throughput on multi-core CPUs.

**Stable-Baselines3.** Stable-Baselines3 (SB3) (Raffin et al., 2021) is a PyTorch library of reliable RL algorithm implementations, including PPO and SAC. Training a policy is a few lines: construct the algorithm with an environment and hyperparameters, call `.learn(total_timesteps=...)`, then `.save(...)`. SB3's maintainers note that small multilayer-perceptron policies of the kind used here typically train efficiently on CPU.

**Takeaway.** Gymnasium, MuJoCo, and SB3 are mature, CPU-friendly, open-source tools, which is what makes a laptop-scale study accessible to others.

### 0.10 Large language models and agents

**Large language models.** A *large language model* (LLM) is a neural network trained on large amounts of text (and, increasingly, code and images) to predict the next piece of text, called a *token*. Modern LLMs can write and debug code, read documentation, interpret plots and images, and follow detailed instructions. The text given to the model is the *prompt*; the maximum amount of text it can consider at once is its *context window*. Generation is stochastic: a *temperature* parameter controls how varied the outputs are, so the same prompt can yield different reward functions on different calls. That variability is a feature for search (it produces diverse candidates) and a nuisance for reproducibility (it must be logged and replicated across runs).

**Tool use.** LLMs offered through modern APIs can call *tools*: the developer describes functions (name, purpose, input schema), the model emits a structured request to call one, the developer's code executes it and returns the result, and the model continues. Tools let a model read files, run validation checks, launch training jobs, or query logs.

**Agents.** An *LLM agent* is a program that runs an LLM in a loop: observe the current situation, reason, take an action (often a tool call), observe the result, and repeat until a goal is met. The ReAct pattern (Yao et al., 2023) interleaves reasoning and acting; *reflection* methods such as Reflexion (Shinn et al., 2023) have the model write a critique of its own previous attempt and use that critique to do better. A *multi-agent* system assigns different roles (designer, critic, supervisor) to different LLM instances.

**LLMs as optimizers.** When an LLM repeatedly proposes candidates, receives scores, and proposes improved candidates, it acts as the *mutation operator* of an evolutionary search. This is the core mechanism of Eureka (Ma et al., 2024) and of our pipeline.

**Failure modes.** LLMs can *hallucinate*, for example by referencing an observation field that does not exist or misreading which index of a vector holds a quantity. They can also over-correct in response to feedback. Our pipeline therefore validates every candidate statically and with a short smoke test before spending compute on it, and executes generated code in a sandbox.

**Takeaway.** The "agentic AI" in this thesis is an LLM in a tool-using loop whose job is engineering: write reward and controller files, read training diagnostics, revise. It operates on a timescale of minutes, never inside the 20 Hz control loop. A second, separate LLM role (the supervisor in Section 3.9) makes slow, high-level decisions from sensor summaries.

### 0.11 Experimental methodology concepts

**Seeds and replication.** Because both RL training and LLM generation are stochastic, every condition is repeated with several independent seeds. There are two distinct sources of randomness to replicate: the *LLM run* (which reward/controller files get written) and the *RL seed* (how training on a fixed file turns out).

**Robust aggregate statistics.** With few runs, means are fragile. Agarwal et al. (2021) recommend the *interquartile mean* (IQM, the mean of the middle 50% of scores) with *bootstrap confidence intervals*, obtained by resampling runs with replacement many times and reading off percentiles.

**Baselines and ablations.** A *baseline* is a comparison method (e.g., the default Ant reward, or a human-written reward). An *ablation* removes one component of the proposed method (e.g., the visual feedback) to measure its contribution.

**Budget matching.** Iterative methods use more total compute than one-shot methods. A fair comparison gives baselines the same total number of simulator steps, so that improvements cannot be attributed merely to longer training (Wang et al., 2026).

### 0.12 Glossary

| Term | Meaning in this thesis |
|---|---|
| Action | The 8-number command sent to Ant's motors (or to the PD/CPG layer) at each control step |
| Agent (LLM agent) | A language model operating in a tool-using loop to design reward and controller files |
| Controller file | `controller.yaml`, the agent-editable specification of action-space structure (torque, PD, CPG), gains, termination settings, and whitelisted training hyperparameters |
| CPG | Central pattern generator: coupled oscillators producing rhythmic joint targets |
| Duty factor | Fraction of a gait cycle a foot is in contact with the ground |
| Episode | One simulated trial, up to 1000 control steps (50 s) for Ant |
| Evaluator / fitness | Frozen code, compiled from the user's specification, that scores a trained policy; never edited by the agent |
| Frame skip | Number of physics timesteps per control step (5 for Ant-v5) |
| IQM | Interquartile mean, a robust average across runs |
| MJCF | MuJoCo's XML robot description format |
| MuJoCo | The physics engine used for all simulation |
| PD control | Feedback law turning target joint angles into torques |
| Policy | The small neural network trained by RL that maps observations to actions |
| PPO / SAC | The on-policy and off-policy RL algorithms used |
| Reward file | `reward.py`, agent-editable code computing the training reward and its named components |
| Reward hacking | A policy exploiting the training reward in unintended ways |
| Reward reflection | A textual summary of how each reward component evolved during training, fed back to the agent |
| Seed | A number fixing the random choices in a run |
| Spec (BehaviorSpec) | The user's structured prompt describing the desired behavior |
| Successive halving | A budget-allocation scheme that trains many candidates briefly, keeps the best fraction, and trains survivors longer |
| Supervisor | The LLM role that issues high-level commands from sensor summaries at about 1 Hz |


---

## 1. Introduction

### 1.1 Motivation

Reinforcement learning has produced remarkable legged locomotion controllers, from quadrupeds that traverse rough terrain to humanoids that recover from pushes. Behind nearly every such result lies a reward function that a human expert designed through extended trial and error. Surveys of practitioners confirm what anyone who has trained a walking robot knows: reward design is overwhelmingly done by hand, is slow, and frequently yields rewards that are suboptimal or that produce unintended behavior (Booth et al., 2023). The reward is not the only hand-tuned artifact. The structure of the low-level controller (whether the policy outputs torques or joint targets, what the PD gains are, whether a rhythmic pattern generator provides a prior over gaits, when an episode should terminate) is chosen by the same expert, through the same cycle of training, watching videos, and editing files.

This cycle has an unusual property that makes it a good candidate for automation by language-model agents: it is a search problem with a cheap, trustworthy verifier. Every candidate reward and controller can be compiled into a policy by a standard RL algorithm and then scored by a physics simulator. Recent work has shown that large language models can write reward code that rivals expert human rewards when embedded in such a loop (Ma et al., 2024; Xie et al., 2024). However, the most prominent demonstrations rely on GPU-accelerated simulation running thousands of environments in parallel on data-center hardware. Eureka's experiments were run on an eight-A100 workstation (Ma et al., 2024), and a 2026 agentic successor reports hundreds of GPU-hours per task for MuJoCo-based humanoid tasks (Lee et al., 2026). This places the approach out of reach for many students, educators, and researchers, and it leaves open the question of whether the benefits survive when compute is scarce.

At the same time, the one recent study that deliberately restricted itself to CPU training reported that its diagnostic refinement method helped substantially on sparse, structured tasks but did not robustly improve dense-reward locomotion, where randomness in RL training rather than reward quality dominated outcome variance (Wang et al., 2026). That finding is both a warning and an opportunity. It suggests that a laptop-scale pipeline for locomotion must be designed around the statistics of RL variance, must allocate its limited budget carefully, and may benefit from editing more than the reward alone, for instance by giving the agent control over action-space structure and gait priors that make learning easier and less noisy.

### 1.2 Problem statement

Given a user's structured description of a desired locomotion behavior for a simple simulated legged robot, automatically produce a trained controller that exhibits that behavior, using only a consumer laptop without a discrete GPU, by letting language-model agents iteratively author and revise the training reward and the low-level controller configuration, with progress judged by an independent, user-approved evaluator. Additionally, enable the resulting skills to be sequenced in response to environmental stimuli by a language-model supervisor operating over summarized sensory input.

### 1.3 Thesis statement

*Agentic language-model systems can automatically tune reward functions and controller configuration files to achieve user-specified locomotion behaviors for a simple legged robot, using a structured behavior specification as input and running entirely on a CPU-only laptop, provided that (a) evaluation fitness is separated from the training reward and frozen before search, (b) compute is allocated adaptively across candidates, and (c) the agents receive structured diagnostic feedback about how each candidate trained and behaved.*

### 1.4 Research questions and hypotheses

**RQ1 (Feasibility).** On a MacBook Pro without a discrete GPU, can the pipeline produce Ant policies that satisfy prompt-specified behaviors, as measured by frozen evaluators, within an overnight compute budget (at most 8 wall-clock hours per behavior)? *H1:* For at least 7 of the 10 behaviors in our task suite (Section 3.10), the pipeline's final policies meet the specification's success threshold in a majority of independent pipeline runs.

**RQ2 (What to edit).** Does granting the agent authority over the controller file, in addition to the reward file, improve outcomes compared with reward-only editing under an identical compute budget? *H2:* Joint reward-and-controller editing yields higher evaluator fitness than reward-only editing, with the largest gains on gait-specific behaviors (e.g., trot, pronk) where a CPG prior is relevant.

**RQ3 (What feedback to give).** Does adding visual evidence (rendered keyframes and foot-contact diagrams) to numeric reward reflection improve alignment between the learned behavior and the specification? *H3:* Multimodal reflection reduces the rate of evaluator-detected constraint violations relative to numeric-only reflection.

**RQ4 (Efficiency and rigor).** How do LLM-generation variance and RL-training variance compare in this setting, and does adaptive budget allocation (successive halving) outperform uniform allocation at matched compute? *H4:* Successive halving produces higher final fitness than uniform allocation at the same total simulator steps.

**RQ5 (Stimulus response).** Can an LLM supervisor, reading summarized sensor data at about 1 Hz and issuing commands to a command-conditioned low-level policy produced by the pipeline, respond appropriately to stimuli (approach a beacon, avoid obstacles, flee a moving threat)? *H5:* The LLM supervisor matches or exceeds a hand-written finite-state baseline on task success in novel stimulus configurations, at the cost of higher latency.

### 1.5 Scope

The thesis is *in scope* for simulated locomotion of Gymnasium's Ant-v5 quadruped in MuJoCo, with optional replication on one additional simple morphology (Gymnasium's Hopper or HalfCheetah) to test generality. It covers reward files, controller configuration files (action-space structure, PD gains, CPG parameters, termination settings, and a whitelist of training hyperparameters), automated evaluation, and a high-level stimulus-response layer driven by simulated sensors. It is *out of scope* for physical hardware, sim-to-real transfer, morphology design (which the co-design literature addresses), vision-based low-level control, rough-terrain locomotion requiring heightmaps, and fine-tuning of language models. The pipeline must run end-to-end on a single Apple Silicon MacBook Pro; any cloud usage is limited to calls to a hosted language model API.

### 1.6 Contributions

This thesis aims to contribute (1) an open-source, CPU-only pipeline in which LLM agents co-design reward and controller files for prompt-specified locomotion; (2) a structured behavior-specification format and a library of verified metric primitives from which frozen evaluators are compiled, with a human approval gate; (3) an empirical study, with budget-matched baselines and separate replication of LLM and RL randomness, of what the agent should edit and what feedback it should receive; (4) a demonstration that the skills produced by the pipeline can be orchestrated by an LLM supervisor responding to simulated stimuli; and (5) a reproducibility package (pinned environment, logged prompts and transcripts, hashed artifacts) designed so that others can replicate the results on their own laptops.

### 1.7 Document roadmap

Chapter 0 provides the background needed to read the rest of this document. Chapter 2 reviews prior work on language models as reward designers, as controllers and controller designers, and as high-level decision makers for legged robots, and identifies the gaps this thesis addresses. Chapter 3 specifies the methodology in enough detail to reproduce it: hardware and software, the behavior specification format, the editable artifacts, the agent roles and tools, the search and budget-allocation procedure, the evaluation protocol, the stimulus-response experiments, and the statistical analysis plan.


---

## 2. Literature Review

### 2.1 Scope and method of the review

This chapter surveys research at the intersection of three areas: automated reward design for reinforcement learning, controller design for legged locomotion, and the use of large language models (LLMs) as engineering agents and as high-level decision makers for robots. The review was conducted as a structured narrative review rather than a formal systematic review. Sources were identified through searches of arXiv and the proceedings of ICLR, NeurIPS, ICML, CoRL, RSS, ICRA, and IROS, together with forward and backward citation chasing from the most central papers (Eureka, Text2Reward, Language to Rewards, and SayTap). Search terms combined "large language model" or "LLM agent" with "reward design", "reward function generation", "reward shaping", "locomotion", "quadruped", "Ant", "MuJoCo", "controller", "gait", "curriculum", "high-level planning", and "vision-language". The window covers foundational work from 1999 onward and LLM-era work through early autumn 2026. Inclusion favored work that (a) uses an LLM to produce or revise an artifact that shapes how a locomotion or continuous-control policy is learned, (b) evaluates on simulated legged robots or standard MuJoCo benchmarks, or (c) uses an LLM to make high-level decisions for a legged robot from sensory information. Because this is a fast-moving field, several 2025 and 2026 works are preprints that had not completed peer review at the time of writing; they are identified as such.

The chapter proceeds from foundations to frontier. Section 2.2 reviews the reward design problem and pre-LLM approaches to automating it. Section 2.3 reviews learning-based and oscillator-based legged locomotion, which defines what a "controller file" can contain. Section 2.4, the core of the review, examines LLMs as reward designers, organized by how much freedom the LLM is given and what feedback it receives. Section 2.5 examines LLMs as controllers and as designers of controller structure. Section 2.6 examines LLMs as high-level decision makers that respond to sensory input. Section 2.7 synthesizes these strands, presents a comparative table, and identifies the gaps that motivate this thesis.

### 2.2 Foundations: the reward design problem before language models

#### 2.2.1 Formalizing reward design

The idea that rewards are a designed artifact rather than a given property of the world was formalized by Singh, Lewis & Barto (2010), who framed reward design as the problem of choosing a reward function, from some space of candidates, such that an agent trained on it maximizes a separate fitness criterion. Eureka later adopted this formulation directly, restating it as a *reward design problem* in which a learning algorithm maps a reward to a policy and a fitness function scores the policy, possibly only through queries (Ma et al., 2024). The formulation is important for this thesis because it makes explicit that the designer's goal is not a high training reward but a high fitness, and that the two can diverge.

Ng, Harada & Russell (1999) established the theory of reward shaping, proving that potential-based shaping terms leave the optimal policy unchanged. Most shaping in practice, including nearly all LLM-generated shaping, is not potential-based; Wang et al. (2026) argue that well-designed one-time milestone bonuses can be viewed as rough approximations of potential-based shaping over an augmented state, while continuous per-step bonuses are not and can alter optimal behavior.

#### 2.2.2 Learning rewards from data

Two families of methods learn rewards rather than writing them. Inverse reinforcement learning infers a reward from expert demonstrations (Abbeel & Ng, 2004; Ziebart et al., 2008), and preference-based learning fits a reward model to human comparisons of trajectory pairs (Christiano et al., 2017). Both reduce hand engineering but require data (demonstrations or many labels) and generally produce opaque neural reward models that are hard to inspect or edit, a limitation emphasized by both Eureka and Text2Reward in motivating code-based alternatives (Ma et al., 2024; Xie et al., 2024).

#### 2.2.3 Evolutionary and AutoRL reward search

Before LLMs, automated reward search was explored with evolutionary methods. Niekum, Barto & Spector (2010) applied genetic programming to reward functions; Faust, Francis & Mehta (2019) evolved reward parameters as part of AutoRL, and Chiang et al. (2019) applied AutoRL to navigation. As the Eureka authors note, these approaches were largely limited to searching parameters within hand-provided reward templates rather than generating new reward structure (Ma et al., 2024). This limitation is precisely what code-writing LLMs relax.

#### 2.2.4 Reward misdesign and specification gaming

A substantial literature documents how rewards go wrong. Booth et al. (2023) surveyed practitioners and found that the overwhelming majority design rewards by manual trial and error and believe their rewards are suboptimal, and they showed that trial-and-error design can overfit rewards to a particular algorithm and hyperparameter setting. Knox et al. (2023) catalogued reward misdesign in autonomous driving. Amodei et al. (2016) identified reward hacking as a core safety problem; Skalse et al. (2022) gave it a formal definition; and Pan, Bhatia & Steinhardt (2022) showed that more capable optimizers exploit misspecified rewards more aggressively. For automated reward design, this literature implies that a search process optimizing training reward, or optimizing a fitness function that is itself an imperfect proxy, will find and amplify specification flaws. This motivates our separation of an agent-editable training reward from a frozen, human-approved evaluator.

### 2.3 Foundations: learning and structuring locomotion controllers

#### 2.3.1 Benchmarks and algorithms

The MuJoCo physics engine (Todorov et al., 2012) and the Gym interface (Brockman et al., 2016), now maintained as Gymnasium (Towers et al., 2024), established the standard continuous-control benchmarks, including Ant, which was introduced in the generalized advantage estimation work of Schulman et al. (2016). PPO (Schulman et al., 2017) and SAC (Haarnoja et al., 2018) became the dominant on-policy and off-policy algorithms for these tasks, with reliable implementations in Stable-Baselines3 (Raffin et al., 2021). Henderson et al. (2018) and Agarwal et al. (2021) showed that results on these benchmarks are highly sensitive to random seeds, implementation details, and evaluation protocol, and recommended robust aggregate statistics with confidence intervals.

#### 2.3.2 Legged locomotion through reinforcement learning

Learned locomotion controllers transferred to real quadrupeds in work such as Hwangbo et al. (2019), which trained agile skills for ANYmal in simulation, and Kumar et al. (2021), whose Rapid Motor Adaptation learned to adjust online to changing terrain and payloads. Rudin et al. (2022) showed that massively parallel GPU simulation could train quadruped walking in minutes. This last result is a double-edged precedent for our work: GPU parallelism made reward iteration fast enough that LLM-driven reward evolution became practical (Ma et al., 2024), but it also set an expectation of hardware that many researchers lack.

#### 2.3.3 Rhythmic priors and controller structure

A parallel tradition structures locomotion controllers around central pattern generators (Ijspeert, 2008). CPGs produce coordinated rhythmic motion from a few interpretable parameters (frequency, amplitude, phase offsets), and gaits correspond to particular phase-offset patterns. Bellegarda & Ijspeert (2022) combined CPGs with deep RL, letting a learned policy modulate oscillator parameters to produce robust quadruped locomotion. Raffin et al. (2022) used a CPG built from four coupled oscillators with phase-dependent swing and stance frequencies for an elastically actuated quadruped, and tuned key gait parameters with black-box optimization directly on hardware. Huang et al. (2020) showed that a single shared modular policy, with message passing between limb modules, could control many morphologies, demonstrating that decentralized controller structure can be learned. For this thesis, these works establish that the *structure* of a controller (action space, oscillator prior, gains) is a design variable with large effects on learnability, and that it is described by a small number of parameters that are natural to express in a configuration file.

### 2.4 Language models as reward designers

This section is organized by the degree of freedom given to the LLM, from filling parameters of a fixed reward template, to writing free-form reward code once, to iteratively evolving reward code with automated feedback, to agentic systems with richer diagnostics and multiple roles.

#### 2.4.1 Templated rewards and language as a proxy reward

An early line of work asked LLMs to *be* the reward. Kwon et al. (2023) prompted an LLM to judge whether an agent's behavior matched a user's described objective, using its judgments as a proxy reward signal. This works for tasks where behavior can be summarized in text, but it produces scalar judgments rather than inspectable reward structure and is expensive to query at every step.

Language to Rewards (L2R) (Yu et al., 2023) introduced the idea that reward functions are a better interface between language and low-level control than direct actions, because rewards are semantically rich while low-level actions are hardware-specific and scarce in LLM training data. L2R used a *Reward Translator*, a two-stage LLM pipeline that first fills a natural-language motion-description template and then converts it into calls to a predefined set of reward primitives with numeric parameters. The resulting reward was optimized in real time by MuJoCo MPC, a model-predictive controller, rather than by RL. Across 17 tasks on a simulated quadruped and a dexterous manipulator, the system allowed users to iteratively correct behavior in natural language and observe results immediately; it also demonstrated tasks such as making a quadruped stand up or perform a moonwalk-like motion. The strength of L2R is interactivity and grounding; its limitation, as later work emphasized, is expressivity, because the LLM can only combine hand-designed primitives. When Eureka adapted L2R as a baseline, giving it primitives derived from the human reward terms, L2R was competitive on low-dimensional tasks but fell well behind on high-dimensional ones, which the Eureka authors attributed to the restricted reward space (Ma et al., 2024).

For our thesis, L2R is relevant in two ways. It is the clearest precedent for an LLM producing *quadruped* behavior from language, and its template is a reminder that the space an agent can edit should be large enough to express the needed behavior but constrained enough to be validated.

#### 2.4.2 Free-form reward code: Text2Reward and self-refinement

Text2Reward (Xie et al., 2024) generalized from templates to free-form, *dense, shaped* reward code. Its prompts provide a compact, Python-class-style abstraction of the environment (attributes, types, and helper methods), background knowledge in the form of useful library functions, and optionally a retrieved example pairing a similar instruction with verified reward code. Generated code is executed to catch syntax and runtime errors, which are fed back for correction; the authors report that this step reduced code error rates from roughly 10% to near zero. After training, users can watch rollout videos and give natural-language feedback that is incorporated into a revised reward.

Text2Reward is the most direct precedent for the specific robot and simulator in this thesis. Its locomotion experiments used Gym MuJoCo's Hopper and Ant, with policies trained using Stable-Baselines3 implementations of PPO and SAC, and evaluated six novel behaviors that have no expert-written reward to compare against: for Ant, moving forward, lying down, and waving the front-right leg while staying in place. Success was judged by the authors watching 100 rollouts per task, with zero-shot generated rewards reaching success rates of 94%, 98%, and 95% respectively for these Ant behaviors. On manipulation, generated rewards matched or exceeded expert-written ones on 13 of 17 tasks. A particularly instructive case for our design is "lie down": the instruction was ambiguous about whether the Ant should end up belly-down or belly-up, and a single round of user feedback specifying the torso orientation resolved it. This example demonstrates two needs that our pipeline addresses explicitly: a structured specification that removes common ambiguities before search begins, and an evaluator that can verify properties such as orientation automatically rather than relying on human video review.

Two limitations of Text2Reward motivate further work. First, it is primarily single-shot generation followed by optional *human* feedback, not an autonomous improvement loop. Second, success on novel locomotion tasks was assessed by human judgment of videos, which does not scale and is not reproducible by others. Its RL training also ran on cloud instances with an A10G GPU, although the MuJoCo locomotion tasks themselves do not require GPU simulation.

Song et al. (2023) proposed a self-refinement framework in which an LLM writes an initial reward from a natural-language description, the reward is used to train a policy, performance is assessed, and the results are fed back for the LLM to refine its reward. Evaluated on continuous control tasks across three robotic systems, LLM-designed rewards rivaled or surpassed manually designed ones. This work established automated (rather than human) feedback as a viable driver of reward improvement, a step toward the evolutionary approach that followed.

#### 2.4.3 Evolutionary reward search: Eureka and its descendants

Eureka (Ma et al., 2024) is the central reference point for this thesis. It rests on three design choices. First, *environment as context*: rather than a hand-written abstraction, the LLM receives the raw environment source code (with any existing reward removed), automatically trimmed to the parts that define observations and state variables. Second, *evolutionary search*: in each iteration the LLM samples a batch of independent reward candidates, each is used to train a policy, and the best becomes the parent for the next batch of in-context mutations. Third, *reward reflection*: rewards must return their individual components, and the feedback to the LLM includes the trajectory of each component and of the fitness score at intermediate training checkpoints, so that the LLM can see which terms were being optimized, which were stagnant, and which dominated in magnitude.

The scale of Eureka's evaluation is substantial: 29 environments spanning 10 robot morphologies, implemented in Isaac Gym, with GPT-4 as the reward writer. Eureka's rewards outperformed expert human rewards on 83% of tasks with an average normalized improvement of 52%, and, combined with a curriculum, enabled a simulated Shadow Hand to perform pen-spinning. Several details are directly relevant to our design. The default search used 5 iterations with 16 samples per iteration and 5 independent restarts per environment; the authors found that 16 samples were enough to guarantee at least one executable reward in the first iteration. The history passed to the LLM was deliberately *Markovian*, keeping only the latest best reward and its reflection, to control context length. Removing reward reflection and passing only the fitness score reduced average normalized performance by 28.6%, with larger degradation on higher-dimensional tasks. Isaac Gym's Ant was among the tasks, with the fitness defined as forward progress per step. Eureka also ran on the MuJoCo Humanoid from Gym with only a formatting change (NumPy instead of PyTorch tensors), producing a reward that achieved higher forward velocity than the official human reward, which suggests the method transfers to the CPU-based MuJoCo stack we use. Finally, Eureka showed that textual human feedback could replace automated reflection: a humanoid trained with human reflections ran somewhat slower but was preferred by most users in a blind comparison for its more stable gait. All experiments ran on a single eight-A100 workstation, with each Eureka run taking under a day; the authors note that four V100s would suffice. That is far more than a laptop, and Eureka's evolutionary design implicitly depends on being able to train 16 policies in parallel quickly.

Several extensions build on Eureka's loop. DrEureka (Ma et al., 2024b) extended LLM-driven design from rewards to *domain randomization* configurations, using the trained policy's behavior under perturbed physics to set safe randomization bounds and then asking the LLM to generate randomization distributions; this enabled sim-to-real transfer for quadruped locomotion and a novel task of a quadruped walking atop a yoga ball. DrEureka is significant for this thesis because it is a precedent for an LLM editing *simulation and training configuration*, not only reward code. CurricuLLM (Ryu et al., 2025) used an LLM to decompose a target task into a sequence of language-described subtasks, translate each into executable reward and goal-distribution code, and evaluate the resulting trajectories to select among candidates; it was tested on manipulation, AntMaze navigation, and humanoid locomotion, with a policy deployed on the Berkeley Humanoid. AURA (2025) pushed the curriculum idea toward an engineering system, compiling LLM-generated curricula into executable training configurations with static schema validation before execution and retrieving prior runs from a vector database, and deployed resulting policies zero-shot on a custom humanoid. The schema-validation step in AURA is a direct precedent for the validated `controller.yaml` used in our methods.

#### 2.4.4 Preference-based and human-in-the-loop variants

A related line uses LLMs to supply preferences rather than reward code. LAPP (Jian et al., 2025) has an LLM label preferences between trajectories using numerical state-action summaries and a behavior instruction; a transformer reward predictor trained on these labels then guides RL. On quadruped locomotion and dexterous manipulation, LAPP achieved precise control of gait patterns and rhythmic timing and enabled quadruped backflips that standard LLM-generated or hand-crafted rewards could not achieve. Language-Guided Preference Learning (2025) used an LLM to generate candidate quadruped reward functions in context and then had humans rank the resulting behaviors, combining the coverage of language with the precision of preferences. MAPL (2026, preprint) decomposed natural-language locomotion criteria into multiple objectives and learned from LLM preference rankings over each. REvolve (Hazra et al., 2025) evolved LLM-generated rewards using human feedback as the fitness signal. These works point to a recurring theme: when the desired behavior is stylistic or hard to state as a formula, some form of judgment over behavior (human, LLM, or vision-language model) is needed in the loop.

#### 2.4.5 Agentic, diagnostic, and low-compute variants (2025-2026)

The most recent work reframes LLM reward design as an agentic engineering process with richer diagnostics, and begins to confront cost.

STRIDE (2025, preprint) automated reward design, deep RL training, and feedback optimization for humanoid locomotion, targeting sprint-level performance. LEARN-Opt (Cardenoso & Caarls, 2025, preprint) proposed a fully autonomous, model-agnostic framework that generates, executes, and evaluates reward candidates from textual descriptions of a system and its objective, without requiring preliminary metrics or environment source code; its central empirical observation is that automated reward design is a high-variance problem in which the typical candidate fails, so that multiple runs are needed to find good rewards. This finding supports population-based search over single-shot generation.

The study closest to our compute setting is Wang et al. (2026, preprint), "When LLM Reward Design Fails." Its authors deliberately used CPU training and few LLM calls per run (one to four), with Claude Haiku as the reward generator and PPO as the learner, evaluating primarily on sparse MiniGrid tasks and treating MuJoCo reaching and locomotion (HalfCheetah, Hopper) as boundary tests. They found that one-shot generation was unreliable, identified two dominant failure modes, *reward flooding* (continuous per-step bonuses that swamp the task signal) and *semantic/API misunderstanding* (misusing environment fields), and showed that diagnostic-driven refinement guided by this failure taxonomy improved sparse tasks dramatically, whereas re-prompting with metrics alone did much worse. On dense locomotion, however, their success-based diagnostics misfired (there is no binary success signal, so high return with zero "success" triggered false reward-hacking warnings), and even a corrected return-trend diagnostic did not produce robust gains; on Hopper, simply training the default reward for the same total episode budget beat the refinement loop. A crossed variance decomposition suggested that in continuous control, RL-training stochasticity and interaction effects outweighed variation due to which reward the LLM wrote. They also found that best-of-three selection was a strong baseline. These results shape our methodology in four concrete ways: diagnostics must be designed for dense, continuous locomotion rather than borrowed from sparse tasks; every comparison must be budget-matched; LLM and RL randomness must be replicated separately; and selection among multiple candidates should be treated as a baseline, not assumed to be inferior to refinement.

The Reward Design Agent (RDA) of Lee et al. (2026, preprint) argued that Eureka's purely numeric reflection often yields policies that succeed on the benchmark's success metric while violating the intent of the instruction, for example a humanoid that throws a package to the goal instead of pushing it. RDA uses a vision-language model to decompose an instruction into subtasks, generate subtask-structured reward candidates, visually analyze rollout frames to score each subtask and explain failures, revise the subtask list, and then revise the reward. On 12 ManiSkill manipulation tasks and 4 HumanoidBench whole-body tasks, RDA matched or exceeded Eureka on success while substantially improving instruction alignment as judged by a separate vision-language evaluator, and performance improved as the number of candidates per iteration increased from one to eight. The compute cost was high: the authors report roughly 240 to 360 GPU-hours per HumanoidBench task, noting that MuJoCo-based HumanoidBench lacks GPU acceleration and is therefore slow. RDA's evidence that visual trajectory analysis improves alignment motivates our RQ3, while its cost motivates our attempt to obtain similar benefits from a few rendered keyframes and contact diagrams rather than dense video analysis.

Two further 2026 systems frame the whole RL workflow as agentic engineering. Prompt-to-Policy (Chung et al., 2026), described by Lee et al. (2026), automates the reward-engineering loop with role-specialized agents and VLM-based trajectory judging. HARBOR (2026, preprint) treats robot RL automation as a *harness-engineering* problem, automating the workflow from environment setup to simulated policy training given a simulator codebase and task specification, with reusable actions and executable validations. Both reinforce a design principle we adopt: agents should operate through narrow, validated tool interfaces rather than with unrestricted access to the codebase.

#### 2.4.6 Joint optimization of reward with other design variables

A distinct line co-optimizes reward with robot *morphology*. RoboMoRe (Fang et al., 2025, preprint) uses an LLM to jointly propose morphology parameters and reward functions, first exploring diverse morphology-reward pairs with a diversity-reflection mechanism and then refining top candidates by alternating LLM-guided reward and morphology updates; it reports outperforming human-engineered designs across eight tasks, including standard MuJoCo Ant and three customized Ant variants. Debate2Create (2026, ICML) assigns a design agent and a control agent to a structured debate, with LLM judges providing multi-objective feedback and every candidate trained and ranked in simulation (using Brax). Across five MuJoCo locomotion benchmarks it reached up to 3.2 times the default Ant score, iterative debate outperformed compute-matched zero-shot generation by roughly 18 to 35 percent, and the learned rewards transferred to the default morphology in four of five tasks. Its authors stress that the LLM agents have a bounded role, proposing hypotheses that only matter once they survive the same training and evaluation protocol. Although morphology design is out of scope for this thesis, these works are the closest precedents for an LLM jointly editing *reward and a second artifact that changes the learning problem*, which is what our controller file does. They also show that multi-agent role separation can improve over a single agent at matched compute.

### 2.5 Language models as controllers and controller designers

#### 2.5.1 LLMs inside the control loop

Wang et al. (2024) asked whether an LLM can act directly as a low-level feedback controller. In "Prompt a Robot to Walk", GPT-4 received a description prompt and a history of recent observations and actions and output normalized target joint positions, which a PD controller tracked; the prompt was updated with new observations after each inference. Experiments across several robots, terrains, and simulators showed that the LLM could produce walking without task-specific fine-tuning. The work is an important existence proof, but it also illustrates the limits of putting an LLM in the loop: control must run at the LLM's inference rate, and every step costs an API call. A 2025 study on sensory-motor control with LLMs via iterative policy refinement similarly had an LLM write and refine control policies using data gathered while interacting with Gymnasium classic-control tasks and MuJoCo's inverted pendulum. These works support our design decision to keep LLMs out of the 20 Hz loop and use them instead to design the controller.

#### 2.5.2 Designing the interface between language and low-level control

SayTap (Tang et al., 2023) is the most relevant work on *controller interface design* for quadrupeds. Rather than generating joint commands or rewards, the LLM converts a user's command into a desired *foot contact pattern*, a binary matrix specifying which feet are on the ground at each time. A low-level RL controller is trained, with a reward concerned only with contact timing, to realize arbitrary contact patterns sampled from a feasible distribution; at test time the LLM supplies the pattern. With a prompt containing just three in-context examples of common gaits, the LLM translated diverse commands, including indirect ones, into appropriate contact patterns, and the controller transferred to real hardware. Compared with alternative interface designs, SayTap's approach achieved over 50% success in predicting correct contact patterns and solved 10 more tasks out of 30. SayTap shows that the right intermediate representation makes language-to-locomotion tractable, and it motivates our use of contact-pattern metrics both in evaluators and as optional CPG phase targets in controller files.

Code as Policies (Liang et al., 2023) showed that LLMs can write executable robot policy code that composes perception and control primitives, an approach later used by many systems. VLM-PC (Chen et al., 2024) represents a quadruped's built-in locomotion controller as a set of *skills* defined by a few parameters (forward and lateral velocity, gait type, body height, yaw rate, duration) and has a vision-language model reason over images and interaction history to choose the next skill, in a receding-horizon fashion. This parameterized-skill interface is closely related to the command interface we use between the supervisor and the low-level policy.

#### 2.5.3 Automated tuning of controller parameters

Outside the LLM literature, controller parameters such as CPG frequencies, amplitudes, and swing/stance durations have long been tuned by black-box optimization (Raffin et al., 2022) or adapted online by RL (Bellegarda & Ijspeert, 2022). We found very little work in which an LLM agent edits the *structure and parameters of the low-level controller* (action space, PD gains, oscillator prior, termination) jointly with the reward for a locomotion task. DrEureka's LLM-generated domain randomization (Ma et al., 2024b), AURA's schema-validated curricula, and the morphology co-design systems are the nearest neighbors, but none treat controller structure as an editable artifact. This is the gap that RQ2 addresses.

### 2.6 Language models for high-level decision making from sensory information

#### 2.6.1 General-purpose robotic planning with LLMs

A large body of work uses LLMs for high-level decisions while leaving low-level control to learned or classical skills. SayCan (Ahn et al., 2022) combined an LLM's estimate of which skill is useful for an instruction with learned value functions estimating which skill is feasible in the current state, grounding language in robot affordances. Inner Monologue (Huang et al., 2022) closed the loop by feeding textual feedback from perception and success detectors back to the LLM planner, enabling replanning when actions failed. Code as Policies (Liang et al., 2023) generated policy code that queried perception APIs. PaLM-E (Driess et al., 2023) embedded continuous sensor observations directly into a multimodal language model, and RT-2 (Brohan et al., 2023) trained vision-language-action models that output robot actions as tokens. VoxPoser (Huang et al., 2023) used LLMs and VLMs to compose 3D value maps for manipulation planning, and LM-Nav (Shah et al., 2022) combined language, vision, and navigation models for instruction-following navigation. Vemprala et al. (2024) documented design principles for prompting ChatGPT to control robots through high-level function libraries. The consistent architectural lesson is a hierarchy in which the language model operates slowly over abstract skills or commands, and fast learned or classical controllers handle dynamics.

#### 2.6.2 Legged robots specifically

For legged robots the hierarchy is particularly clear, because balance must be maintained at high frequency. SayTap places the LLM above a contact-pattern controller (Tang et al., 2023). VLM-PC places a VLM above parameterized locomotion skills (Chen et al., 2024). NaVILA (Cheng et al., 2025) uses a vision-language-action model to produce mid-level actions expressed in language (such as moving forward a certain distance), which a separately trained RL locomotion policy executes in real time, enabling vision-and-language navigation for quadrupeds and humanoids. LocoVLM (Nahrendra et al., 2026, preprint) uses a pre-trained VLM to retrieve motion descriptors from a database based on instructions or robot-centric images and conditions a style-aware locomotion controller on them, adapting locomotion in real time. A 2025 system for adaptive interactive navigation of quadrupeds used a VLM and object detector to describe the scene, an LLM to build and revise a tree of primitive skills, and an RL skill library in which a velocity-tracking walking policy executes commands from higher-level navigation and pushing skills. A 2025 study on quadruped movement-plan generation grounded LLM-generated plans into ROS navigation commands using fused LiDAR, IMU, and odometry data and reported success above 90% across several indoor scenarios.

Across these systems, three interface choices recur: (a) *velocity or command interfaces*, where the high level issues target velocities, headings, or gait choices to a command-conditioned policy; (b) *skill-selection interfaces*, where the high level chooses among named skills; and (c) *intermediate representations*, such as contact patterns or language waypoints. Sensory information reaches the language model either as images (for VLMs) or as structured text summaries of perception (object lists, distances, success flags), as in Inner Monologue. Latency is handled by running the language model asynchronously at low frequency and relying on the low-level policy for stability.

### 2.7 Synthesis and research gaps

Table 2.1 summarizes the most relevant works along the dimensions that matter for this thesis.

**Table 2.1.** Comparison of representative LLM-in-the-loop locomotion and reward-design systems.

| Work | Robot / simulator | What the LLM produces | Feedback to LLM | Learner | Reported compute |
|---|---|---|---|---|---|
| L2R (Yu et al., 2023) | Quadruped, manipulator / MuJoCo | Parameters of templated reward terms | Human corrections in language | MuJoCo MPC | Real-time MPC |
| Text2Reward (Xie et al., 2024) | Hopper, Ant (Gym MuJoCo); manipulators | Free-form dense reward code | Execution errors; optional human feedback | SB3 PPO/SAC | Cloud A10G instances |
| Self-refined LLM (Song et al., 2023) | Three robotic systems | Reward code | Automated performance summary | Deep RL | Not central |
| Eureka (Ma et al., 2024) | 10 morphologies incl. Ant / Isaac Gym; MuJoCo Humanoid | Free-form reward code, evolved | Reward-component reflection | PPO | 8×A100 workstation |
| DrEureka (Ma et al., 2024b) | Quadruped, hand / Isaac Gym; real robot | Reward and domain-randomization configs | Policy behavior under perturbation | PPO | GPU |
| CurricuLLM (Ryu et al., 2025) | Fetch, AntMaze, humanoid | Subtask curriculum, reward and goal code | LLM evaluation of trajectories | RL | GPU |
| LAPP (Jian et al., 2025) | Quadrupeds, hand | Trajectory preference labels | Not applicable (labels train reward model) | RL | GPU |
| Wang et al. (2026) | MiniGrid; MuJoCo reach, HalfCheetah, Hopper | Reward shaping code | Failure-taxonomy diagnostics | PPO | CPU, 1–4 LLM calls |
| RDA (Lee et al., 2026) | ManiSkill; HumanoidBench (MuJoCo) | Subtasks and reward code | VLM visual trajectory analysis | SAC | ~240–360 GPU-h per humanoid task |
| RoboMoRe / D2C (2025–26) | Ant and MuJoCo locomotion | Morphology and reward | Simulation scores; LLM judges (D2C) | RL (Brax for D2C) | GPU |
| Prompt a Robot to Walk (Wang et al., 2024) | Several legged robots / simulators | Joint position targets each step | Observation history | None (LLM is policy) | API calls per step |
| SayTap (Tang et al., 2023) | Quadruped / sim and real | Foot contact patterns | None at runtime | RL controller pretrained | GPU for controller |
| VLM-PC, NaVILA, LocoVLM | Legged robots | Skill parameters, language waypoints, motion descriptors | Images and history | Pretrained locomotion policies | GPU / onboard |

From this synthesis, five gaps emerge.

**G1. Compute accessibility.** The strongest results in LLM-driven reward design rely on GPU-parallel simulation and data-center hardware. The only study we found that restricted itself to CPU training reported weak gains for dense locomotion (Wang et al., 2026). There is no demonstrated, carefully evaluated pipeline that produces prompt-specified locomotion behaviors on a laptop, and no study of how to allocate a laptop's limited simulation budget across LLM-proposed candidates.

**G2. Controller structure as an editable artifact.** Prior work lets LLMs edit rewards, curricula, domain randomization, or morphology, and separately shows that controller structure (action space, CPG priors, interfaces such as contact patterns) strongly affects learnability. We found no work in which an LLM agent jointly edits reward code and a validated low-level controller configuration for locomotion, and no measurement of whether doing so helps under fixed compute.

**G3. Specification and evaluation of prompt-defined behaviors.** Benchmarks used in Eureka-style work supply a ground-truth fitness function. When the behavior is defined by a user's prompt (e.g., "trot at 0.8 m/s with low energy"), someone must define fitness. Text2Reward relied on human video review; RDA used a VLM judge. Neither offers a reproducible, structured path from a user's specification to a verified evaluator that the reward-writing agent cannot influence, and RDA's results show that evaluation choices determine whether misaligned behaviors are caught.

**G4. Feedback design for dense locomotion at low cost.** Numeric reward reflection is cheap but can miss misalignment (Lee et al., 2026); dense visual analysis is effective but costly; and diagnostics designed for sparse tasks misfire on locomotion (Wang et al., 2026). The value of lightweight, locomotion-specific diagnostics (gait contact diagrams, a few keyframes, constraint-violation reports) is unmeasured.

**G5. From designed skills to stimulus response.** Reward-design work ends with a trained policy for one behavior; high-level LLM decision-making work assumes a pre-existing library of locomotion skills. No work we found connects the two, using an agentic design loop to produce a command-conditioned locomotion skill and then using an LLM supervisor to deploy it in response to simulated stimuli, all on modest hardware.

### 2.8 Positioning of this thesis

This thesis addresses G1 through G5 with a single pipeline evaluated on a deliberately simple robot. It inherits Eureka's evolutionary search with component-level reflection (Ma et al., 2024), Text2Reward's compact environment abstraction and execution-based validation (Xie et al., 2024), RDA's insight that behavioral evidence beyond scalar scores improves alignment (Lee et al., 2026), and the methodological cautions of Wang et al. (2026) regarding budget matching, variance, and diagnostic design. Its novel elements are the joint editing of reward and controller files under a validated schema, a structured specification compiled into a frozen evaluator from verified metric primitives, adaptive budget allocation suited to a laptop, and the extension from designed skills to an LLM supervisor that responds to stimuli.


---

## 3. Methods

### 3.1 Overview

The method is a closed loop with four kinds of participants: a deterministic *orchestrator* written in ordinary Python, several *LLM agents* with distinct roles, a fixed *RL trainer* (Stable-Baselines3 on MuJoCo), and a frozen *evaluator*. The loop is organized into four phases, of which Phase 0 is run once per machine and Phases 1 to 3 are run per behavior.

```
                       ┌───────────────────────────────────────────────┐
  BehaviorSpec (YAML)  │ PHASE 1: SPECIFY                              │
  written by user ───► │  Spec Interpreter agent ─► evaluator config   │
                       │  ─► human approval ─► evaluator FROZEN (hash) │
                       └───────────────────────┬───────────────────────┘
                                               ▼
  ┌──────────────────────────────── PHASE 2: DESIGN-SEARCH LOOP ─────────────────────────────────┐
  │                                                                                               │
  │  Designer agents (×4, parallel) ── propose ──► reward.py + controller.yaml  (K per generation)│
  │        ▲                                            │                                         │
  │        │                                   validate (static + smoke test)                     │
  │   diagnoses                                         ▼                                         │
  │        │                         Orchestrator: successive-halving training on CPU             │
  │  Analyst agent ◄── diagnostics ◄── (PPO/SB3 + MuJoCo Ant-v5 wrappers, J parallel jobs)        │
  │  (numbers, flags,                                   │                                         │
  │   gait plot, frames)                                ▼                                         │
  │                                  Frozen evaluator ─► fitness, constraint report ─► archive    │
  └───────────────────────────────────────────────────────────────────────────────┬──────────────┘
                                                                                  ▼
                                 Final training (5 RL seeds) of best candidate ─► policy + report
                                                                                  │
  PHASE 3 (stimulus layer): command-conditioned policy (task T9) ◄────────────────┘
        ▲ commands @ ~1 Hz                       ▲ reflexes @ 20 Hz
  Supervisor agent ◄── sensor summaries ◄── AntStimulusArena (rangefinders, beacon, threat)
```

Phase 0 calibrates the machine and the RL trainer. Phase 1 turns the user's specification into a frozen evaluator. Phase 2 is the agentic search over reward and controller files. Phase 3 uses a command-following skill produced by Phase 2 inside a stimulus-response experiment with an LLM supervisor.

Agentic AI enters at exactly four points, each with a narrow tool interface: (1) the *Spec Interpreter* maps the user's specification onto verified metric primitives and asks clarifying questions; (2) *Designer* agents write and revise `reward.py` and `controller.yaml`; (3) the *Analyst* agent turns training and evaluation diagnostics into a structured diagnosis; and (4) the *Supervisor* agent issues high-level commands from sensor summaries. No agent can modify the simulator, the RL algorithm's code, the evaluator, the orchestrator, or the compute budget.

### 3.2 Hardware and software environment

#### 3.2.1 Reference hardware

The reference machine is an Apple Silicon MacBook Pro with at least 10 CPU cores and 16 GB of unified memory; an M3 Pro, M4 Pro, or newer chip with 18 GB or more is recommended. No discrete GPU is used, and the Apple GPU is not used for training (Section 3.2.3 explains why). The laptop must be on mains power, and long runs are wrapped in `caffeinate -i` to prevent sleep. Every run logs the chip model, core counts, memory, macOS version, and power state, and samples CPU frequency every 60 s so that thermal throttling can be detected and reported.

#### 3.2.2 Software stack

All software is open source except the hosted LLM API. Versions below are the latest stable releases at the time of writing; the exact versions used are frozen in a `uv.lock` lockfile after Phase 0 and must not change during the study.

| Component | Package | Version policy | Role |
|---|---|---|---|
| Language | Python | 3.12.x | Runtime |
| Environment manager | `uv` | latest | Reproducible lockfile |
| Physics | `mujoco` | 3.x (3.7.0 or later; the most recent release found while drafting); pin after Phase 0 | Simulation |
| Environments | `gymnasium[mujoco]` | 1.x with `Ant-v5` (1.3.0 is supported by SB3 2.9); pin after Phase 0 | Env API, Ant-v5 |
| RL | `stable-baselines3` | ≥ 2.9 (supports Gymnasium up to 2.0, requires PyTorch ≥ 2.8) | PPO, SAC, VecNormalize |
| Deep learning | `torch` | ≥ 2.8, CPU build | Policy networks |
| Black-box optimization | `cma` (pycma) | latest 3.x | CMA-ES for CPG baselines |
| Statistics | `rliable`, `scipy`, `numpy` | latest | IQM, bootstrap CIs, tests |
| Validation | `pydantic` ≥ 2, `pyyaml` | latest | Spec and controller schemas |
| LLM access | `anthropic` Python SDK | latest | Messages API with tool use |
| Media | `imageio`, `matplotlib` | latest | Keyframes, gait diagrams |
| Process control | `psutil` | latest | Timeouts, memory guard |

The default LLM for Designer, Analyst, and Spec Interpreter roles is a current mid-tier Claude model accessed through the Anthropic Messages API with tool use (at the time of writing, `claude-sonnet-5-5`), with a smaller model (`claude-haiku-5-5`) and a larger model (`claude-opus-5-5`) used in the model-capacity ablation. The pipeline is model-agnostic: all prompts and tools are defined independently of the provider, and an optional replication with an open-weight model is planned to test model dependence. Exact model identifiers, request parameters, and timestamps are logged with every call, because hosted models can change over time. API documentation: https://docs.claude.com/en/api/overview.

#### 3.2.3 CPU execution policy

All RL training runs on CPU (`device="cpu"`). Small MLP policies on MuJoCo tasks are typically limited by simulation rather than by neural-network computation, so GPU acceleration offers little benefit, and Apple's Metal backend for PyTorch adds overhead for small batches. To avoid thread oversubscription when several training jobs run concurrently, every training process sets `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and `torch.set_num_threads(1)`, and uses `SubprocVecEnv` with the default `spawn` start method on macOS. GPU-parallel simulators (MuJoCo's MJX or MuJoCo Warp, Isaac Gym) are explicitly out of scope, since the point of the study is accessibility.

### 3.3 Simulator and robot configuration

#### 3.3.1 Base environment

All experiments use the unmodified Gymnasium `Ant-v5` environment, constructed with `include_cfrc_ext_in_observation=False` (yielding the 27-dimensional proprioceptive observation) and `exclude_current_positions_from_observation=True`. The base environment's own reward is discarded by the reward wrapper but its `info` dictionary is preserved for logging. To allow the PD and CPG controllers to act at physics resolution, the base environment is created with `frame_skip=1` (physics step 0.01 s), and a controller wrapper holds each policy action for `action_repeat=5` physics steps, recomputing PD torques at every physics step. The effective control frequency therefore remains 20 Hz and the episode length remains 1000 policy steps (50 s), matching the standard configuration. Termination settings (`terminate_when_unhealthy`, `healthy_z_range`) are passed through from `controller.yaml` within the bounds listed in Section 3.6.2.

#### 3.3.2 Wrapper stack

From inside to outside, each environment instance is composed of the following wrappers, all implemented in the project's `lib/` package and never edited by agents:

1. **`StateViewWrapper`** constructs, after every physics step, a read-only `StateView` object exposing named physical quantities (Section 3.3.3), and accumulates per-step data needed by evaluator metrics (contacts, torques, positions).
2. **`ControllerWrapper`** implements the action mode selected in `controller.yaml` (torque, PD target, CPG with residual, or open-loop CPG; Section 3.6.2), including action repeat, PD computation, oscillator integration, optional low-pass filtering, and clipping to the actuator range $[-1,1]$.
3. **`CommandWrapper`** (only for command-conditioned tasks) samples velocity commands at episode start and every 5 s, and appends them to the observation.
4. **`ObservationWrapper`** appends optional features declared in `controller.yaml` (command, CPG phase as $(\sin\varphi_i,\cos\varphi_i)$ for each leg, previous action).
5. **`RewardWrapper`** calls the agent's `compute_reward` in the sandbox (Section 3.6.1), returns the total reward to the learner, and logs each named component.
6. **`TimeLimit`** truncates at 1000 policy steps.

The vectorized environment is wrapped in SB3's `VecNormalize` (observation and reward normalization, observation clipping at 10). Normalization statistics are saved with every checkpoint so that resumed training and evaluation use the same statistics.

#### 3.3.3 The StateView API

Following Text2Reward's finding that a compact, Pythonic abstraction of the environment helps LLMs write correct reward code (Xie et al., 2024), and Wang et al.'s (2026) finding that misunderstanding environment interfaces is a dominant failure mode, agents never see raw observation indices. They write rewards against a documented `StateView` with explicitly named and unit-annotated fields, including: torso position (m), torso quaternion, up-vector $z$-component (`up_z`), roll/pitch/yaw (rad), linear velocity in world and body frames (`lin_vel_world`, `lin_vel_body`, m/s), angular velocity in body frame (rad/s), joint positions and velocities as dictionaries keyed by joint name (`hip_fl`, `ankle_fl`, ..., `hip_br`, `ankle_br`), applied actuator torques, the previous policy action (`prev_action`), foot contact booleans and normal forces for each of the four feet, foot positions, a torso-ground contact flag, the current command (if any), CPG phases (if any), the simulation time, and the control timestep. Foot and torso contact are computed by scanning MuJoCo's contact list for contacts between the floor geom and the geoms belonging to each lower leg or the torso; geom identifiers are resolved by name from `mjModel` at startup, and the startup routine asserts that every expected name exists, failing loudly if a future Gymnasium release renames them. A helper library (`lib.reward_helpers`) provides tolerance kernels (Gaussian, linear, sigmoid), quaternion utilities, and safe normalization functions, each documented with its signature and an example.

#### 3.3.4 Joint conventions and calibration

Ant's hinges have asymmetric ranges, and the sign of "lifting the foot" differs between legs. To avoid sign errors in PD and CPG controllers, the `ControllerWrapper` reads each joint's range from `mjModel.jnt_range` at startup and works in *normalized joint coordinates*, where $-1$ and $+1$ correspond to the ends of each joint's range and 0 to its midpoint. In Phase 0, a calibration script determines for each ankle which normalized direction raises the foot (by commanding each direction and measuring foot height) and for each hip which direction moves the foot toward the robot's front, and stores the result in `calibration.json`. CPG outputs and PD targets are expressed in these calibrated coordinates, so that "lift" and "forward" mean the same thing for every leg.

### 3.4 The behavior specification format

#### 3.4.1 Design rationale

The user's prompt follows a fixed YAML schema, the *BehaviorSpec*. A structured format serves three purposes. It forces the user to state the quantities that free-text instructions often leave ambiguous (the Ant "lie down" orientation ambiguity reported by Xie et al. (2024) is the canonical example). It provides the information needed to compile a verified evaluator. And it bounds the search budget. Free text is still allowed in a `description` field, which the Spec Interpreter reads, but every scored property must ultimately be expressed in terms of the metric primitives in Section 3.5.1.

#### 3.4.2 Schema

```yaml
# BehaviorSpec v1
task_id: string                     # unique identifier, e.g. "T5_trot"
robot: ant_v5                       # only value supported in this thesis
description: >                      # free text; read by the Spec Interpreter
  Trot forward at a moderate speed with diagonal legs moving together,
  keeping the body level and using little energy. Do not fall or flip.
objectives:                         # scored, weighted; at least one required
  - metric: mean_velocity           # name from the metric primitive library
    args: {frame: world, axis: x}
    target: 0.8                     # in the metric's units (m/s)
    tolerance: 0.15                 # scale of the scoring kernel
    weight: 1.0
  - metric: gait_match
    args: {gait: trot}
    target: 1.0
    tolerance: 0.2
    weight: 0.7
  - metric: cost_of_transport
    direction: minimize             # one-sided: lower is better
    limit: auto                     # "auto" = 0.8 x Phase-0 baseline CoT
    tolerance: 0.5
    weight: 0.3
constraints:                        # hard pass/fail gates
  - metric: flip_rate
    max: 0.0
  - metric: termination_rate
    max: 0.1
  - metric: tilt_rms
    max_deg: 12
  - metric: lateral_drift
    max: 0.15                       # m of lateral drift per m of progress
success:
  objectives_within_tolerance_fraction: 0.8   # fraction of eval episodes
  constraints: all
evaluation:
  episodes: 10
  episode_seconds: 30
  warmup_seconds: 2                 # excluded from metrics
  initial_state_noise: default
  commands: null                    # or a command schedule for T9
budget:
  wall_clock_hours: 8
  max_env_steps: 30_000_000
  max_llm_calls: 300
editable: [reward, controller]      # set per experimental condition
stimuli: null                       # only used in Phase 3 mission specs
notes_for_designers: >              # optional user hints, passed verbatim
  A trot usually needs a stride frequency of roughly 1-3 Hz.
```

The schema is defined with `pydantic`, which rejects unknown metric names, missing units, and invalid ranges before any LLM is involved.

### 3.5 Evaluator compilation

#### 3.5.1 Metric primitive library

The evaluator is assembled only from verified *metric primitives*, each implemented once in `lib/metrics.py`, unit-tested against hand-constructed trajectories, and documented with units and definition. Agents can read the documentation but cannot add or modify primitives. The initial library is:

| Primitive | Definition (computed after warm-up over each evaluation episode) |
|---|---|
| `mean_velocity(frame, axis)` | Mean torso linear velocity along an axis of the world or body frame |
| `mean_yaw_rate` | Mean torso angular velocity about world $z$ |
| `velocity_tracking_error` | RMS error between commanded and achieved $(v_x, v_y, \omega_z)$, body frame, each normalized by its command range |
| `displacement(axis)` / `path_length` | Net displacement along an axis / total planar path length |
| `heading_drift` | Absolute change in yaw over the episode |
| `lateral_drift` | Absolute displacement perpendicular to the intended direction per metre of progress |
| `torso_height_mean`, `torso_height_std` | Statistics of torso $z$ |
| `tilt_rms`, `tilt_max` | RMS and maximum of $\sqrt{\text{roll}^2+\text{pitch}^2}$ |
| `upright_fraction` | Fraction of time with up-vector $z > 0.5$ |
| `flip_rate` | Fraction of episodes in which up-vector $z < 0$ at any time |
| `termination_rate` | Fraction of episodes ending by termination before the time limit |
| `torso_ground_contact_fraction` | Fraction of time the torso touches the floor |
| `cost_of_transport` | $\sum_t\sum_j \lvert\tau_j\dot q_j\rvert\Delta t \,/\, (m g d)$ with $d$ the planar path length |
| `action_rate` | Mean $\lVert a_t-a_{t-1}\rVert^2$ (smoothness) |
| `duty_factor(foot)` | Fraction of time a foot is in contact |
| `stride_frequency` | Touchdowns per second averaged over feet |
| `phase_offsets` | Circular mean of each foot's touchdown phase relative to front-left |
| `gait_match(gait)` | $1-\bar\delta/\pi$, where $\bar\delta$ is the mean circular distance between measured and target phase offsets (trot, pace, bound, pronk, walk) |
| `flight_fraction` | Fraction of time with no foot in contact |
| `foot_height(foot)` statistics | Mean/max height of a named foot; fraction of time above a threshold |
| `oscillation_frequency(foot, axis)` | Dominant frequency of a foot's motion (FFT) |
| `stationarity` | Net planar displacement (for in-place tasks) |
| `pose_error(target)` | Error in torso height and up-vector relative to a target pose, held for a duration |
| `joint_limit_saturation` | Fraction of time any joint is within 2% of a limit |

#### 3.5.2 Scoring

Each objective $m$ with measured value $x_m$ is converted to a score $s_m\in[0,1]$. Two-sided objectives use a Gaussian kernel, $s_m=\exp(-((x_m-\text{target}_m)/\text{tol}_m)^2)$. One-sided objectives give $s_m=1$ when the limit is satisfied and $s_m=\exp(-((x_m-\text{limit}_m)/\text{tol}_m)^2)$ otherwise. Constraints $c_j$ yield a soft gate $g_j=\operatorname{clip}(1-v_j/\text{tol}_j,0,1)$ where $v_j\ge 0$ is the amount of violation (with $\text{tol}_j$ set to 25% of the constraint bound by default), and a hard pass/fail flag. The episode fitness is

$$F = \Big(\prod_j g_j\Big)\cdot\frac{\sum_m w_m s_m}{\sum_m w_m},$$

and the candidate's fitness is the mean of $F$ over evaluation episodes. Using a product of soft gates means that constraint violations sharply reduce fitness, while the smooth form still ranks near-misses above gross failures, which helps the search. *Success* is a binary judgment applied to the final policy: all hard constraints pass in every evaluation episode, and each objective is within tolerance ($s_m\ge e^{-1}$, i.e., within one tolerance of target) in at least the specified fraction of episodes.

#### 3.5.3 The Spec Interpreter and the human approval gate

The Spec Interpreter agent receives the BehaviorSpec and the metric library documentation. Its task is to check the specification for ambiguities and gaps relative to the free-text description, propose additional objectives or constraints the user likely intends (for example, an orientation constraint for "lie down", or a `stationarity` constraint for "wave a leg in place"), and pose at most three clarifying questions to the user through an `ask_user` tool. It outputs a proposed evaluator configuration, which `pydantic` validates. The user (in our experiments, the researcher, following a written protocol) reviews the proposal, may edit it, and approves it. The approved evaluator configuration is serialized, its SHA-256 hash is recorded, and it is frozen; any later change requires restarting the task. The agent's proposed additions and the human's edits are logged as data for the analysis of RQ1.

For the task suite (Section 3.10), the evaluator configurations are drafted, piloted on a small number of hand-written policies in a development phase, and frozen before any main experiment, which functions as a lightweight preregistration of success criteria.


### 3.6 The editable artifacts

Each candidate is a pair of files, `reward.py` and `controller.yaml`, plus a short natural-language rationale written by the Designer. Depending on the experimental condition, one of the two files may be frozen at a default.

#### 3.6.1 The reward file

**Interface.** The reward file must define exactly one function:

```python
def compute_reward(s: StateView, a: np.ndarray, s_next: StateView,
                   ctx: dict) -> tuple[float, dict[str, float]]:
    """Return (total_reward, components). `ctx` persists within an episode
    and is reset at episode start; use it only for one-time bonuses."""
```

**Rules.** The function may import only `math`, `numpy as np`, and `lib.reward_helpers`. It may not perform I/O, access attributes beginning with an underscore, or call `eval`, `exec`, `open`, `getattr`, or `__import__`. Every additive term must appear by name in the returned `components` dictionary, and every scale or temperature must be a named module-level constant, following Eureka's practice of exposing components and temperatures so that reflection can refer to them (Ma et al., 2024). The file may be at most 200 lines.

**Validation.** Before any training, a candidate passes through four checks: (1) an abstract-syntax-tree (AST) whitelist enforcing the rules above; (2) execution on 2,000 policy steps with zero actions and 2,000 with random actions in a sandboxed subprocess with a 60 s timeout, which must produce only finite values; (3) a magnitude check that flags any component whose mean absolute value is more than 50 times another's (likely domination) or whose per-step value exceeds 100; and (4) a *reward-flooding probe*, which compares the per-episode return of a zero-action policy and a random-action policy to the return of a fixed, hand-written reference gait for the task (stored in `lib/reference_gaits/`) and flags rewards under which standing still earns more than 80% of the reference gait's return. Checks 1 and 2 are blocking; checks 3 and 4 produce warnings that are returned to the Designer, which may revise before submitting. Each Designer may make at most three validation attempts per candidate.

**Sandboxing.** Generated code runs only inside environment worker subprocesses, which have no API credentials, are killed if they exceed time or memory limits (monitored with `psutil`), and are given read-only access to the project directory.

#### 3.6.2 The controller file

The controller file selects and parameterizes the low-level control structure and a small whitelist of environment and training settings. It is validated against a `pydantic` schema; out-of-range values are rejected with an explanatory message.

```yaml
# controller.yaml v1
action_mode: cpg_residual      # torque | pd_target | cpg_residual | cpg_open_loop
pd:                            # used by pd_target and cpg_* modes
  kp: 8.0                      # [1, 40], normalized torque per normalized joint unit
  kd: 0.3                      # [0, 3]
  action_scale: 0.6            # [0.1, 1.0], max target deviation (normalized units)
cpg:                           # used by cpg_* modes
  gait: trot                   # trot | pace | bound | pronk | walk | custom
  phase_offsets: null          # custom only: [FL, FR, BL, BR] in radians
  frequency_hz: 1.5            # [0.3, 4.0]
  hip_amplitude: 0.5           # [0, 1] normalized
  ankle_amplitude: 0.6         # [0, 1] normalized
  hip_offset: 0.0              # [-0.5, 0.5]
  ankle_offset: 0.0            # [-0.5, 0.5]
  duty_bias: 0.0               # [-0.3, 0.3], stance/swing speed asymmetry
  coupling: 1.0                # [0, 5]
  convergence_rate: 10.0       # [1, 50]
  residual_scale: 0.3          # cpg_residual only, [0, 1]
  policy_modulates: [frequency, amplitude]   # subset of {frequency, amplitude, offset}
observation:
  include_command: false
  include_cpg_phase: true
  include_prev_action: true
action_filter:
  lowpass_alpha: 0.0           # [0, 0.8]; a_t <- alpha*a_{t-1} + (1-alpha)*a_t
env:
  terminate_when_unhealthy: true
  healthy_z_range: [0.2, 1.0]  # each bound within [0.05, 1.5]
  reset_noise_scale: 0.1       # [0.0, 0.2]
training:                      # only editable in conditions that allow it
  learning_rate: 3.0e-4        # [1e-4, 1e-3]
  ent_coef: 0.0                # [0, 0.01]
  gamma: 0.99                  # [0.95, 0.995]
  net_arch: [256, 256]         # one of [[64,64],[128,128],[256,256],[256,256,256]]
```

The four action modes are implemented as follows. In **`torque`** mode the policy's 8-dimensional output is sent directly to the actuators (the standard Ant interface). In **`pd_target`** mode the policy outputs normalized joint targets $q^\ast = q_{\text{mid}} + \text{action\_scale}\cdot a$ in calibrated coordinates, and the wrapper applies $u=\operatorname{clip}(k_p(q^\ast-q)-k_d\dot q,-1,1)$ at every physics step. In **`cpg_residual`** mode a four-oscillator CPG (Section 0.6), with phase offsets determined by `gait`, produces nominal joint targets; the policy outputs residual target corrections (scaled by `residual_scale`) plus bounded modulations of the oscillator parameters listed in `policy_modulates`, and PD tracks the sum, following the CPG-plus-learning approach of Bellegarda & Ijspeert (2022). In **`cpg_open_loop`** mode there is no neural policy: the CPG parameters themselves are the controller, optimized with CMA-ES (Hansen, 2016) against the training reward (population 8, at most 40 generations, two 20-second episodes per evaluation), which provides a very cheap controller-only search and a warm start for `cpg_residual`.

Because termination settings are editable, a Designer can disable unhealthy-termination for posture tasks such as lying down, which is impossible under Ant-v5's default height range. Termination settings never affect the evaluator, which always measures the full evaluation episode and records falls and flips through its own metrics.

### 3.7 Agent architecture

#### 3.7.1 Roles

**Orchestrator (not an LLM).** A deterministic Python process implements Algorithm 1 (Section 3.8.2). It owns the budget, the job queue, random seeds, the candidate archive, and all file writes. Keeping orchestration deterministic makes runs reproducible up to LLM sampling and isolates the LLM's influence to the content of candidates and diagnoses.

**Spec Interpreter (LLM, once per task).** Described in Section 3.5.3.

**Designers (LLM, four instances per generation).** Each Designer receives the same task context but a different *strategy instruction*, to promote diversity among candidates, a property both Eureka and RDA found important (Ma et al., 2024; Lee et al., 2026):

| Designer | Strategy instruction (abridged) |
|---|---|
| D1 Refiner | Make the smallest change that addresses the highest-priority diagnosed failure of the current best candidate. |
| D2 Structural | Prefer changes to `controller.yaml` (action mode, gait prior, gains, termination) over reward changes; change the reward only if necessary. |
| D3 Explorer | Propose a substantially different hypothesis from the elite candidates, e.g., a new reward structure or a different action mode. |
| D4 Simplifier | Produce the simplest reward with the fewest components that could plausibly satisfy the specification. |

In conditions where only the reward is editable, D2's instruction is replaced by "Re-weight and re-scale reward components based on the reflection." Each Designer produces two candidates per generation, for $K=8$ candidates per generation.

**Analyst (LLM, once per evaluated candidate that reaches the second rung).** The Analyst receives the diagnostics package (Section 3.9) for one candidate and returns a structured JSON diagnosis containing a list of failure modes from the taxonomy, the evidence for each (citing specific numbers, frames, or contact-diagram features), the likely cause in the reward or controller file, and up to three concrete suggested edits. Diagnoses of all promoted candidates are stored in the archive.

**Supervisor (LLM, Phase 3 only).** Described in Section 3.11.

#### 3.7.2 Tools

All agents interact with the system only through tools exposed via the API's tool-use mechanism. Tool implementations validate inputs and return structured JSON.

| Tool | Available to | Function |
|---|---|---|
| `get_spec()` | All | Returns the BehaviorSpec and the frozen evaluator's human-readable description (metrics, targets, tolerances) |
| `get_env_api()` | Designers, Analyst | Returns StateView documentation, helper-library documentation, controller schema with bounds, and the calibration summary |
| `get_archive(k)` | Designers | Returns the top-$k$ elite candidates (full files, fitness, metric breakdown, diagnosis) and one-paragraph summaries of the previous generation's failures |
| `validate(reward_py, controller_yaml)` | Designers | Runs the checks in Section 3.6.1 and schema validation; returns errors, warnings, and component magnitude statistics from the smoke test |
| `submit(reward_py, controller_yaml, rationale)` | Designers | Enqueues a validated candidate; returns a candidate ID |
| `get_diagnostics(candidate_id)` | Analyst | Returns the numeric diagnostics, automated flags, and images (gait diagram, keyframe grid) |
| `list_metric_primitives()` / `propose_evaluator(config)` / `ask_user(question)` | Spec Interpreter | Evaluator construction and clarification |
| `set_command(...)` / `hold()` / `note(text)` | Supervisor | Phase 3 actions (Section 3.11.4) |

Designers cannot launch training or consume compute directly; `submit` only enqueues a candidate for the orchestrator. This prevents an agent from altering the budget and makes compute accounting exact.

#### 3.7.3 Context management

Each Designer call contains: a fixed system prompt (Appendix A.1) with role, rules, and the strategy instruction; the output of `get_spec` and `get_env_api` (about 4,000 to 6,000 tokens); and the output of `get_archive(3)`, which holds the three best candidates in full with their diagnoses plus failure summaries from the previous generation. This is a deliberate middle ground between Eureka's strictly Markovian history, which keeps only the latest best reward (Ma et al., 2024), and an unbounded transcript. The Analyst sees only one candidate at a time. Designer sampling temperature is 1.0 and Analyst temperature is 0.2; all other sampling parameters use provider defaults and are logged.

### 3.8 Search and compute allocation

#### 3.8.1 Phase 0: machine and trainer calibration

Phase 0 is run once per machine before any experiment and has four steps.

*Step 1, throughput benchmark.* A benchmark script trains PPO on Ant-v5 with the default reward and default controller for 200,000 steps under each combination of environments per job $n_{\text{env}}\in\{2,4,8\}$ and concurrent jobs $J\in\{1,2,3,4\}$, recording aggregate environment steps per second (SPS). The configuration with the highest aggregate SPS whose per-job SPS is at least 60% of the single-job value is selected. The same benchmark is run for SAC with $J\in\{1,2,3,4\}$ and one environment per job.

*Step 2, trainer validation.* Using the selected configuration, PPO with the default Ant-v5 reward is trained for 3 million steps with 3 seeds. The default PPO hyperparameters (Table 3.1) are accepted if the mean evaluation forward velocity exceeds 1.0 m/s for all seeds. If not, a small grid over learning rate $\{1\times10^{-4},3\times10^{-4}\}$ and `n_steps` $\{512, 1024, 2048\}$ is run once, the best setting is adopted, and hyperparameters are then frozen for the whole study (except in conditions that permit editing the `training` block).

*Step 3, joint calibration.* The joint-direction calibration of Section 3.3.4 is run and saved.

*Step 4, reference measurements.* Reference values needed by evaluators are measured (for example, the cost of transport of the default-reward policy, used as the baseline for `limit: auto`), and hand-written reference gaits for the reward-flooding probe are verified.

*Step 5, rung calibration.* Successive halving is only useful if short training runs rank candidates similarly to long ones. Using 12 pilot candidates (hand-written and one-shot LLM rewards for tasks T1 and T5, covering all four action modes), each is trained to 3 million steps with 2 seeds, and the Spearman rank correlation $\rho$ between evaluator fitness at $\{10^5, 2\times10^5, 3\times10^5, 10^6\}$ steps and at $3\times10^6$ steps is computed. The first rung is set to the shortest length with $\rho\ge0.6$, and the later rungs to roughly three and ten times that length. The default rungs below assume $10^5$; if calibration selects a longer first rung, the budget arithmetic is recomputed with the same formula and the number of generations is reduced, if necessary, to stay within the 8-hour limit.

**Table 3.1.** Default PPO hyperparameters (starting point validated in Phase 0).

| Hyperparameter | Value |
|---|---|
| Environments per job | from Phase 0 (expected 4) |
| `n_steps` per environment | 1024 |
| Batch size | 256 |
| Epochs per update | 10 |
| Learning rate | $3\times10^{-4}$, constant |
| Discount $\gamma$ / GAE $\lambda$ | 0.99 / 0.95 |
| Clip range | 0.2 |
| Entropy coefficient | 0.0 |
| Value coefficient / max grad norm | 0.5 / 0.5 |
| Network | separate actor and critic MLPs, [256, 256], tanh |
| Normalization | `VecNormalize` on observations and rewards, clip 10 |

SAC (buffer $10^6$, batch 256, learning rate $3\times10^{-4}$, $\tau=0.005$, automatic entropy tuning, [256, 256] networks, 10,000 warm-up steps) is used only in the final-training ablation (Section 3.12), since its per-step cost on CPU is expected to make it a poor choice for many short screening runs; Phase 0 measurements determine whether that expectation holds.

#### 3.8.2 Phase 2 algorithm

The search runs for $G=4$ generations, each with $K=8$ candidates, and allocates training steps by *successive halving* (Jamieson & Talwalkar, 2016; Li et al., 2018): all candidates receive a short training run, the best fraction is promoted and trained longer by resuming from its checkpoint, and so on. Promotion is based on the frozen evaluator's fitness, never on training reward.

```
Algorithm 1: Agentic reward-and-controller search (per task, per pipeline run)
Input: frozen evaluator E, BehaviorSpec S, budget B, rungs R = [(8, 1e5), (4, 3e5), (2, 1e6)]
Archive A ← ∅;  Parent P ← default candidate (reward template from S, controller default)
for g = 1 .. G:
    C ← ∅
    for each Designer d in {D1..D4} in parallel:
        C ← C ∪ d.propose(S, env_api, A.top(3), A.last_gen_failures)   # 2 validated candidates each
    survivors ← C
    for (n_keep, steps) in R:                                          # successive halving
        survivors ← top n_keep of survivors by fitness (first rung: all K)
        for c in survivors (J concurrent jobs):
            continue training c to `steps` total (seed s_c); evaluate with E (5 episodes)
        for c in survivors at rung ≥ 2: c.diagnosis ← Analyst(diagnostics(c))
    best ← argmax fitness over survivors at final rung
    train best with a second RL seed to 1e6 steps; fitness(best) ← mean over both seeds
    A.add(all candidates with fitness, metrics, diagnoses, rung reached)
    if success(best) on both seeds and g ≥ 2: break                    # early stop
    stop if budget B (env steps, wall clock, or LLM calls) exhausted
Final: take top-2 archive candidates; train each with 5 new RL seeds to 3e6 steps;
       evaluate each seed with E (10 episodes); report the better candidate (selected on
       seeds 1–2) and its metrics on seeds 3–5 (held-out seeds, to avoid selection bias).
```

The default parent for generation 1 is the *spec-derived template reward* (Section 3.12, baseline C4, with weights at their initial values) paired with the default controller (`pd_target` mode, default gains), so that the first reflection has a meaningful reference point. Rung lengths $R$ are those selected in Phase 0, Step 5.

**Budget arithmetic.** Per generation, successive halving consumes $8\times10^5 + 4\times2\times10^5 + 2\times7\times10^5 + 1\times10^6 = 4.0$ million steps. Four generations consume 16 million steps, and final training consumes $2\times5\times3\times10^6=30$ million steps, for a worst-case total of about 46 million steps. If Phase 0 measures an aggregate throughput of $S$ steps per second, the wall-clock time is about $46\times10^6/S$ seconds plus LLM latency (typically a few minutes per generation). For illustration, at $S=3{,}000$ this is about 4.3 hours, and at $S=1{,}600$ it is about 8 hours. If Phase 0 finds $S<1{,}600$, the final-training stage is reduced to the single best candidate (15 million steps), which keeps the total under 8 hours at $S\geq 1{,}100$. The budget actually used is logged and reported for every run, and all baselines are matched to it (Section 3.12).

**Early stopping and failure handling.** A training job is killed early if its episode length falls below 5% of maximum for 100,000 consecutive steps (the robot is falling immediately) or if non-finite values appear; the candidate receives fitness 0 and the failure is reported to the Analyst. If all eight candidates in a generation fail validation or training, the generation is repeated once with the failure messages included; a second total failure ends the run.

### 3.9 Diagnostics and reflection

Wang et al. (2026) showed that diagnostics designed for sparse tasks misfire on dense locomotion, and Lee et al. (2026) showed that numeric reflection alone misses behavioral misalignment. Our diagnostics package is therefore designed specifically for locomotion and combines numbers, automated flags, and a small number of images. For each candidate reaching the second rung, the orchestrator computes:

1. **Reward reflection (numeric).** For each named reward component and for the total, the mean per-episode value at ten evenly spaced training checkpoints, plus its share of total absolute reward, in the format pioneered by Eureka (Ma et al., 2024).
2. **Evaluator report.** Every objective and constraint value, its score, pass/fail, and the overall fitness, at the latest checkpoint and at each earlier rung.
3. **Training-fitness alignment.** The Spearman rank correlation between training return and evaluator fitness across checkpoints.
4. **Gait diagram.** A PNG raster plot of the four feet's contact states over 5 s of an evaluation episode, with measured duty factors and phase offsets annotated, plus the target pattern if the spec names a gait.
5. **Keyframes.** A PNG grid of six frames from a tracking camera at fixed times in one evaluation episode, plus one side-view frame.
6. **Automated flags.** Computed by code, not by the LLM, using the locomotion failure taxonomy in Table 3.2.

**Table 3.2.** Automated locomotion failure flags.

| Flag | Rule | Typical cause |
|---|---|---|
| F1 Component domination | One component accounts for > 80% of total absolute reward | Mis-scaled weights |
| F2 Survival exploitation | Episode length near maximum but primary objective score < 0.2 | Alive/upright bonus outweighs task term (reward flooding) |
| F3 Early termination | Mean episode length < 50% of maximum | Reward encourages unstable motion; termination too strict |
| F4 Flip or belly contact | Flip rate > 0 or torso-ground contact > 5% of time | Missing orientation term; exploit by rolling |
| F5 Gait mismatch | `gait_match` < 0.6 when a gait is specified | No contact-timing incentive; wrong CPG offsets |
| F6 Direction error | Velocity magnitude adequate but heading or lateral drift constraint violated | World- vs body-frame confusion |
| F7 Chattering | `action_rate` above Phase-0 baseline × 3 | No smoothness term; gains too high |
| F8 No learning | Training return slope over last 50% of steps not significantly positive and fitness < 0.3 | Reward too sparse or too flat |
| F9 Reward-fitness divergence | Spearman correlation between training return and fitness < 0 | Reward hacking: the reward is being optimized but the specification is not |

In the numeric-only ablation (Section 3.12), the Analyst receives only items 1 and 2, which approximates Eureka-style reflection. In the no-images ablation it receives items 1 to 3 and 6. The full method provides all six.

### 3.10 Task suite

Ten behaviors, spanning velocity tracking, gait style, efficiency, posture, single-limb motion, command following, and a dynamic stretch goal, test the pipeline. Thresholds are calibrated during a development phase using hand-written controllers and are then frozen.

**Table 3.3.** Task suite (abridged; full BehaviorSpecs in the repository).

| ID | Behavior | Key objectives | Key constraints |
|---|---|---|---|
| T1 | Walk forward at 1.0 m/s | `mean_velocity(world,x)` = 1.0 ± 0.2 | no flips, lateral drift ≤ 0.15, termination ≤ 10% |
| T2 | Walk backward at 0.5 m/s | `mean_velocity(world,x)` = −0.5 ± 0.15 | as T1 |
| T3 | Turn in place counter-clockwise | `mean_yaw_rate` = 0.6 ± 0.15 rad/s | stationarity ≤ 0.5 m per 20 s; no flips |
| T4 | Crab-walk left | `mean_velocity(world,y)` = 0.4 ± 0.1 | heading drift ≤ 15°; no flips |
| T5 | Trot forward | velocity 0.8 ± 0.15; `gait_match(trot)` ≥ 0.8 | duty factor 0.4–0.7 per foot; tilt RMS ≤ 12° |
| T6 | Energy-efficient walk | velocity ≥ 0.6 m/s; CoT ≤ 0.8 × Phase-0 baseline | no flips; termination ≤ 10% |
| T7 | Lie down belly-down and stay | `pose_error` (height ≤ 0.3 m, up-z ≥ 0.8) within 3 s, held | stationarity ≤ 0.3 m; no flips |
| T8 | Wave front-right leg in place | FR foot above 0.15 m ≥ 50% of time; FR oscillation 0.5–2 Hz | other three feet in contact ≥ 90%; stationarity ≤ 0.3 m |
| T9 | Follow velocity commands | `velocity_tracking_error` ≤ 0.25 over commands $v_x\in[-0.5,1.0]$, $v_y\in[-0.4,0.4]$, $\omega_z\in[-0.8,0.8]$ resampled every 5 s | no flips; termination ≤ 5% |
| T10 | Pronk forward (stretch) | `gait_match(pronk)` ≥ 0.7; `flight_fraction` ≥ 0.1; velocity ≥ 0.3 | no flips; tilt max ≤ 30° |

Tasks T7 and T8 deliberately mirror Text2Reward's Ant "lie down" and "wave leg" tasks (Xie et al., 2024), allowing a qualitative comparison while replacing human video judgment with frozen evaluators. Task T9 produces the command-conditioned skill used in Phase 3.


### 3.11 Phase 3: stimulus response with an LLM supervisor

#### 3.11.1 Architecture

The stimulus-response experiments use a three-layer hierarchy, following the pattern that recurs across SayCan, Inner Monologue, VLM-PC, and NaVILA (Section 2.6): a fast learned controller, a fast hand-written reflex layer, and a slow language-model supervisor.

The **low-level controller** is the final T9 command-following policy produced by Phase 2, which runs at 20 Hz and tracks commands $(v_x, v_y, \omega_z)$. The **reflex layer** is a small, hand-written, deterministic rule set evaluated at 20 Hz that overrides the current command in two safety situations: if any front rangefinder reports an obstacle closer than 0.35 m, forward velocity is clamped to zero and the robot yaws away from the nearer side; if torso tilt exceeds 35°, the command is set to zero until tilt recovers. Every reflex activation is reported to the supervisor in its next summary. The **supervisor** is an LLM agent that receives a natural-language mission and, at 1 Hz of simulated time, a compact sensor summary, and responds by calling tools that set the command. Separating reflexes from deliberation ensures that collision avoidance and fall prevention do not depend on LLM latency, while still letting the LLM make every strategic decision.

#### 3.11.2 The stimulus arena

`AntStimulusArena` is a wrapper around Ant-v5 that adds stimuli to a flat 20 m × 20 m region. Obstacles (boxes 0.3 to 1.0 m wide) are inserted into the MJCF model at episode creation using MuJoCo's model-editing interface (`mujoco.MjSpec`) or, equivalently, by templated XML; a *beacon* is a non-physical point target; a *threat* is a kinematic sphere driven by a MuJoCo mocap body that, once triggered, moves toward the robot at 0.3 m/s; and *rest zones* are circular regions where the mission may require the robot to stop. Layouts are generated from a seed by a procedural generator with guaranteed solvability (a collision-free path exists, checked on an occupancy grid).

#### 3.11.3 Sensors and summaries

The simulated sensor suite consists of nine rangefinder rays spanning the front 180° at torso height (computed with MuJoCo ray casting, maximum range 4 m), an IMU (orientation and angular velocity), the four foot-contact flags, the beacon's bearing and intensity (falling off with the inverse square of distance, standing in for a light or sound source), the threat's bearing and distance when within 5 m, the robot's estimated velocity, and cumulative energy used. A deterministic *summarizer* converts these to a short JSON object once per simulated second, for example:

```json
{"t": 12.0, "speed_mps": 0.62, "tilt_deg": 4,
 "nearest_obstacle": {"bearing_deg": -20, "dist_m": 0.8},
 "free_directions_deg": [-90, -60, 30, 60],
 "beacon": {"bearing_deg": 35, "intensity": 0.41},
 "threat": null, "reflex_events": [], "last_command": [0.6, 0.0, 0.2]}
```

Summaries rather than raw arrays are used because they are compact, interpretable, and match how Inner Monologue supplied perception to its planner (Huang et al., 2022). An optional multimodal variant adds a rendered top-down image every 5 s.

#### 3.11.4 Supervisor tools and timing

The supervisor has three tools: `set_command(vx, vy, yaw_rate, duration_s, reason)`, which is clipped to the T9 command ranges and held until the duration expires or a new command arrives; `hold()`, which keeps the current command; and `note(text)`, which writes to a short scratchpad that is returned with subsequent summaries (giving the supervisor memory without an unbounded transcript). The context window holds the mission, the last 10 summaries, and the scratchpad.

Two timing modes are evaluated. In *synchronous* mode the simulation pauses while the supervisor responds, which measures decision quality independent of latency. In *asynchronous* mode the simulation runs at real-time factor 1 while the supervisor computes, and commands take effect when they arrive, which measures the practical cost of LLM latency.

#### 3.11.5 Missions, baselines, and metrics

Three missions of increasing difficulty are defined in natural language: (M1) *seek*, reach the beacon and stop within 0.5 m of it; (M2) *seek and avoid*, reach the beacon in a field of 6 to 12 obstacles; and (M3) *seek, avoid, and flee*, as M2 but a threat is triggered at a random time, and the robot must keep at least 2 m from it, retreating until it is at least 4 m away before resuming. For each mission, 10 layouts form a development set used to refine the supervisor's prompt, and 50 different seeded layouts form the held-out test set, used once.

The supervisor is compared with three baselines using the same low-level policy and reflex layer: (i) a hand-written finite-state machine using a potential-field rule (attraction to the beacon, repulsion from obstacles and the threat); (ii) random commands; and (iii) an oracle planner with full map knowledge that computes a shortest path on an occupancy grid and issues tracking commands, as an upper bound. Metrics are mission success, time to completion, number of obstacle collisions (any non-foot contact with an obstacle), minimum distance to the threat, falls, number of supervisor calls, tokens consumed, and mean decision latency.

### 3.12 Experimental design

#### 3.12.1 Conditions

The following conditions are run for each task in the suite. All conditions use the same frozen evaluator, the same RL hyperparameters (unless the condition allows editing the `training` block), and a total budget of environment steps matched to the median steps actually consumed by the full method (O3) on that task.

| ID | Condition | Edits | Search | Feedback |
|---|---|---|---|---|
| C0 | Default Ant-v5 reward (T1 only; T2/T4 with the forward term re-oriented) | none | none; one long run | none |
| C1 | Human expert | reward + controller | researcher iterates freely within the matched step budget and a 3-hour time box | own judgment |
| C2 | One-shot LLM | reward + controller | 1 candidate, trained with full budget | none |
| C3 | Best-of-K selection | reward + controller | 8 one-shot candidates, successive halving, no reflection | none |
| C4 | Template + CMA-ES (non-LLM AutoRL) | weights of spec-derived template reward + CPG parameters | CMA-ES over ≤ 12 continuous parameters with successive-halving evaluation | fitness only |
| C5 | Eureka-style | reward only | 4 generations × 8, best-parent, uniform allocation | numeric reflection only |
| O1 | Ours, reward only | reward | Algorithm 1 | full diagnostics |
| O2 | Ours, controller only | controller (reward fixed to spec-derived template) | Algorithm 1 | full diagnostics |
| O3 | **Ours, full** | reward + controller | Algorithm 1 | full diagnostics |

The *spec-derived template reward* used by C4, O2, and as the default parent is generated by code (not an LLM) from the BehaviorSpec: for each objective it adds the corresponding per-step quantity passed through the same scoring kernel used by the evaluator, plus a constant upright bonus of 0.5 and a control cost of $0.05\lVert a\rVert^2$, with all weights equal to the objective weights. It represents a careful, automatic but non-agentic baseline.

Ablations of O3 are run on four representative tasks (T1, T5, T7, T9): **A1** without images (numeric reflection, alignment statistic, and flags); **A2** numeric reflection only; **A3** uniform allocation instead of successive halving at matched steps; **A4** four identical Designers (strategy diversity removed); **A5** smaller and larger LLMs; and **A6** SAC instead of PPO for final training (matched wall-clock time rather than matched steps, since SAC's cost per step differs).

#### 3.12.2 Replication

Two sources of randomness are replicated separately. Each condition is run as **5 independent pipeline runs** per task, each with a different LLM sampling sequence (the API does not guarantee determinism, so independence is achieved by separate runs, all logged). The final policy of each run is evaluated across **3 held-out RL seeds × 10 episodes** (Algorithm 1, final step). In addition, for tasks T1 and T5, a **crossed variance study** takes the five final candidates of O3, trains each with the same five RL seeds (25 training runs), and estimates the variance attributable to candidate identity, RL seed, and residual with a two-way random-effects ANOVA, following Wang et al. (2026).

#### 3.12.3 Human rating study

Because frozen evaluators may miss qualities users care about (for instance, naturalness of a gait), a small blinded rating study complements them. For tasks T5, T7, and T8, rollout videos from the final policies of C1, C2, C5, and O3 (one per pipeline run, randomly ordered, condition labels hidden) are rated by three raters on a 1–5 scale for compliance with the free-text description. Inter-rater agreement is reported as Krippendorff's alpha, and rater scores are correlated with evaluator fitness to estimate evaluator validity.

#### 3.12.4 Compute estimate for the whole study

At an assumed aggregate throughput of 3,000 steps per second, O3 requires roughly 4 to 5 hours per pipeline run. Ten tasks × 5 runs gives about 225 machine-hours for O3. Baselines are cheaper per run (they have no Analyst calls and C0/C2 have no search) but are matched in steps, so they cost roughly the same simulation time; restricting C4, C5, O1, and O2 to six tasks and ablations to four tasks gives a total of roughly 1,000 to 1,300 machine-hours. On one laptop running overnight and on weekends this is approximately 10 to 14 weeks; the study is designed so that several students' laptops can share the workload, because every run is an independent job defined by a configuration file and a seed. The total LLM cost is logged and reported in tokens and in currency at the prices in effect at the time of running.

### 3.13 Statistical analysis plan

The primary outcome for RQ1 is the per-run success indicator, aggregated per task as a success fraction with a Wilson 95% confidence interval, and summarized across tasks. The primary outcome for RQ2 to RQ4 is evaluator fitness of final policies on held-out RL seeds. Following Agarwal et al. (2021), per-task fitness is aggregated across runs with the interquartile mean (IQM) and 95% stratified bootstrap confidence intervals (10,000 resamples), computed with `rliable`, and between-condition comparisons report the probability of improvement with its bootstrap interval. Pre-specified hypothesis tests (H2: O3 vs O1; H3: O3 vs A2; H4: O3 vs A3; plus O3 vs C3 and O3 vs C5) use Wilcoxon signed-rank tests on per-task median fitness across the task suite, with Holm–Bonferroni correction across this family. Effect sizes are reported as the median paired difference with bootstrap intervals. For RQ5, mission success rates are compared between the supervisor and the finite-state baseline with a two-proportion test on the held-out layouts, and continuous metrics with Mann–Whitney tests. All analysis code is written and tested on pilot data before the main experiments, and any deviation from this plan is reported as exploratory.

In addition to final-performance statistics, the analysis reports compute-efficiency curves (best fitness so far versus cumulative environment steps and versus cumulative LLM tokens), the frequency of each automated failure flag per generation, and how often the Analyst's suggested edits were adopted by Designers and whether adoption was followed by fitness improvement.

### 3.14 Reproducibility, logging, and safety

**Configuration and hashing.** Every run is fully defined by a run configuration file (task, condition, pipeline-run index, base seed) together with the locked environment. Every candidate's `reward.py` and `controller.yaml`, every evaluator configuration, and every trained checkpoint (with its `VecNormalize` statistics) are stored under content hashes, so that any reported number can be traced to the exact files that produced it.

**LLM logging.** Every LLM request and response, including system prompts, tool calls and results, model identifier, sampling parameters, token counts, latency, and timestamp, is appended to a JSONL transcript per run. Because hosted model behavior may change over time, the transcripts make the study auditable even if exact replication of the LLM outputs is impossible; a *replay mode* re-runs the training and evaluation of all logged candidates without calling the LLM, which makes everything downstream of the LLM exactly reproducible.

**Seeds.** RL seeds are derived deterministically from the base seed, the candidate hash, and the rung, and are logged. Evaluation episodes use a separate, fixed seed list shared across all conditions.

**Release.** The code, BehaviorSpecs, frozen evaluators, prompts, transcripts, final policies, and analysis notebooks will be released under an open-source license, with a one-command entry point (Appendix B).

**Safety and data handling.** Generated code is executed only in sandboxed worker processes as described in Section 3.6.1. Only simulation data, generated code, and the researcher's specifications are sent to the LLM API; no personal data is involved. API keys are stored in the environment of the orchestrator process only.

### 3.15 Threats to validity and limitations

*Internal validity.* LLM outputs are stochastic and providers may update models, which is mitigated by multiple independent pipeline runs, full transcripts, replay mode, and pinned model identifiers. RL variance can mask differences between conditions, which is addressed by held-out seeds, crossed variance estimation, and robust statistics. Selection of the final candidate uses seeds separate from those used to report its performance, avoiding optimistic bias.

*Construct validity.* Evaluator fitness is only as good as the specification and metric primitives. The human approval gate, the pilot calibration of thresholds, and the human rating study estimate how well evaluator fitness tracks what users actually want.

*External validity.* Ant is a simplified, idealized robot, and results may not transfer to robots with realistic actuators, sensing, or terrain; no sim-to-real claims are made. The default Ant reward is widely published, so LLMs may have memorized it, which could inflate performance on T1; tasks T3 to T10 have no canonical published reward and are less susceptible. If time permits, tasks T1, T2, and a posture task will be replicated on Gymnasium's Hopper to test transfer of the method to another morphology.

*Hardware variability.* Throughput varies across laptops and with thermal conditions. Budgets are therefore defined in environment steps rather than wall-clock time for all comparisons, and wall-clock is reported descriptively together with the machine profile.

### 3.16 Work plan

| Phase | Duration | Deliverable |
|---|---|---|
| Infrastructure and Phase 0 calibration | 3 weeks | Wrappers, StateView, metric library with unit tests, sandbox, calibrated trainer, rung calibration |
| Development | 3 weeks | Prompt development on T1 and T5 (development seeds only), frozen evaluators for T1–T10 |
| Main Phase 2 experiments | 10–14 weeks (machine time, partly parallel with writing) | O3 and baselines on all tasks, ablations on four tasks, crossed variance study |
| Phase 3 experiments | 4 weeks | Arena, supervisor, baselines, held-out mission evaluation |
| Analysis and writing | 6 weeks | Results and discussion chapters, public release |


---

## References

Preprints that had not completed peer review at the time of writing are marked *(preprint)*.

- Abbeel, P., & Ng, A. Y. (2004). Apprenticeship learning via inverse reinforcement learning. *Proceedings of ICML*.
- Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A., & Bellemare, M. G. (2021). Deep reinforcement learning at the edge of the statistical precipice. *NeurIPS 34*.
- Ahn, M., Brohan, A., Brown, N., et al. (2022). Do as I can, not as I say: Grounding language in robotic affordances. arXiv:2204.01691.
- Amodei, D., Olah, C., Steinhardt, J., Christiano, P., Schulman, J., & Mané, D. (2016). Concrete problems in AI safety. arXiv:1606.06565.
- AURA authors. (2025). AURA: Agentic upskilling via [full title to verify]: LLM-designed staged curricula for humanoid locomotion. arXiv:2506.02507 *(preprint)*.
- Bellegarda, G., & Ijspeert, A. J. (2022). CPG-RL: Learning central pattern generators for quadruped locomotion. *IEEE Robotics and Automation Letters*, 7(4).
- Booth, S., Knox, W. B., Shah, J., Niekum, S., Stone, P., & Allievi, A. (2023). The perils of trial-and-error reward design: Misdesign through overfitting and invalid task specifications. *Proceedings of AAAI*, 37, 5920–5929.
- Brockman, G., Cheung, V., Pettersson, L., et al. (2016). OpenAI Gym. arXiv:1606.01540.
- Brohan, A., Brown, N., Carbajal, J., et al. (2023). RT-2: Vision-language-action models transfer web knowledge to robotic control. arXiv:2307.15818.
- Cardenoso, F., & Caarls, W. (2025). Leveraging LLMs for reward function design in reinforcement learning control tasks (LEARN-Opt). arXiv:2511.19355 *(preprint)*.
- Chen, A. S., et al. (2024). Commonsense reasoning for legged robot adaptation with vision-language models (VLM-PC). arXiv:2407.02666.
- Cheng, A.-C., Ji, Y., et al. (2025). NaVILA: Legged robot vision-language-action model for navigation. *Robotics: Science and Systems*.
- Chiang, H.-T. L., Faust, A., Fiser, M., & Francis, A. (2019). Learning navigation behaviors end-to-end with AutoRL. *IEEE Robotics and Automation Letters*, 4(2), 2007–2014.
- Christiano, P. F., Leike, J., Brown, T., Martic, M., Legg, S., & Amodei, D. (2017). Deep reinforcement learning from human preferences. *NeurIPS 30*.
- Chung, W., Ha, T., Kwak, Y., Kwon, T., Lee, J.-G., Lee, K., & Lee, S. (2026). Prompt-to-Policy: Agentic engineering for reinforcement learning. https://github.com/krafton-ai/Prompt2Policy *(software/preprint)*.
- Debate2Create authors. (2026). Debate2Create: Robot co-design via multi-agent LLM debate. *Proceedings of ICML*, PMLR 306. arXiv:2510.25850.
- Driess, D., Xia, F., Sajjadi, M. S. M., et al. (2023). PaLM-E: An embodied multimodal language model. *Proceedings of ICML*.
- Fang, J., et al. (2025). RoboMoRe: LLM-based robot co-design via joint optimization of morphology and reward. arXiv:2506.00276 *(preprint)*.
- Faust, A., Francis, A., & Mehta, D. (2019). Evolving rewards to automate reinforcement learning. arXiv:1905.07628.
- Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. *Proceedings of ICML*.
- Hansen, N. (2016). The CMA evolution strategy: A tutorial. arXiv:1604.00772.
- HARBOR authors. (2026). HARBOR: A harness framework for agentic robot reinforcement learning. arXiv:2606.08610 *(preprint)*.
- Hazra, R., et al. (2025). REvolve: Reward evolution with large language models using human feedback. *ICLR*.
- Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., & Meger, D. (2018). Deep reinforcement learning that matters. *Proceedings of AAAI*, 32.
- Huang, W., Mordatch, I., & Pathak, D. (2020). One policy to control them all: Shared modular policies for agent-agnostic control. *Proceedings of ICML*.
- Huang, W., Xia, F., Xiao, T., et al. (2022). Inner Monologue: Embodied reasoning through planning with language models. *CoRL*. arXiv:2207.05608.
- Huang, W., Wang, C., Zhang, R., Li, Y., Wu, J., & Fei-Fei, L. (2023). VoxPoser: Composable 3D value maps for robotic manipulation with language models. *CoRL*. arXiv:2307.05973.
- Hwangbo, J., Lee, J., Dosovitskiy, A., Bellicoso, D., Tsounis, V., Koltun, V., & Hutter, M. (2019). Learning agile and dynamic motor skills for legged robots. *Science Robotics*, 4(26), eaau5872.
- Ijspeert, A. J. (2008). Central pattern generators for locomotion control in animals and robots: A review. *Neural Networks*, 21(4), 642–653.
- Jamieson, K., & Talwalkar, A. (2016). Non-stochastic best arm identification and hyperparameter optimization. *Proceedings of AISTATS*.
- Jian, P., et al. (2025). LAPP: Large language model feedback for preference-driven reinforcement learning. *Transactions on Machine Learning Research*. arXiv:2504.15472.
- Knox, W. B., Allievi, A., Banzhaf, H., Schmitt, F., & Stone, P. (2023). Reward (mis)design for autonomous driving. *Artificial Intelligence*, 316, 103829.
- Kumar, A., Fu, Z., Pathak, D., & Malik, J. (2021). RMA: Rapid motor adaptation for legged robots. *Robotics: Science and Systems*.
- Kwon, M., Xie, S. M., Bullard, K., & Sadigh, D. (2023). Reward design with language models. *ICLR*.
- Language-Guided Preference Learning authors. (2025). Efficiently generating expressive quadruped behaviors via language-guided preference learning. arXiv:2502.03717 *(preprint)*.
- Lee, H., Subramanian, A., Abbatematteo, B., Veerabadran, V., Matias, P., Ridgeway, K., & Kamra, N. (2026). Reward Design Agent for reinforcement learning (RDA). arXiv:2606.01672 *(preprint)*.
- Li, H., Yang, X., Wang, Z., et al. (2024). Auto MC-Reward: Automated dense reward design with large language models for Minecraft. *CVPR*, 16426–16435.
- Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., & Talwalkar, A. (2018). Hyperband: A novel bandit-based approach to hyperparameter optimization. *Journal of Machine Learning Research*, 18(185).
- Liang, J., Huang, W., Xia, F., Xu, P., Hausman, K., Ichter, B., Florence, P., & Zeng, A. (2023). Code as Policies: Language model programs for embodied control. *ICRA*, 9493–9500.
- Ma, Y. J., Liang, W., Wang, G., Huang, D.-A., Bastani, O., Jayaraman, D., Zhu, Y., Fan, L., & Anandkumar, A. (2024). Eureka: Human-level reward design via coding large language models. *ICLR*. arXiv:2310.12931.
- Ma, Y. J., Liang, W., Wang, H.-J., Zhu, Y., Fan, L., Bastani, O., & Jayaraman, D. (2024b). DrEureka: Language model guided sim-to-real transfer. *Robotics: Science and Systems*. arXiv:2406.01967.
- MAPL authors. (2026). MAPL: Multi-objective preference learning for robot locomotion. arXiv:2606.25398 *(preprint)*.
- Nahrendra, I., Lee, S., Lee, D., & Myung, H. (2026). LocoVLM: Grounding vision and language for adapting versatile legged locomotion policies. arXiv:2602.10399 *(preprint)*.
- Ng, A. Y., Harada, D., & Russell, S. (1999). Policy invariance under reward transformations: Theory and application to reward shaping. *Proceedings of ICML*, 278–287.
- Niekum, S., Barto, A. G., & Spector, L. (2010). Genetic programming for reward function search. *IEEE Transactions on Autonomous Mental Development*, 2(2), 83–90.
- Pan, A., Bhatia, K., & Steinhardt, J. (2022). The effects of reward misspecification: Mapping and mitigating misaligned models. *ICLR*.
- Quadruped navigation authors. (2025). Adaptive interactive navigation of quadruped robots using large language models. arXiv:2503.22942 *(preprint)*.
- Quadruped planning authors. (2025). Quadruped-legged robot movement plan generation using large language model. arXiv:2512.21293 *(preprint)*.
- Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., & Dormann, N. (2021). Stable-Baselines3: Reliable reinforcement learning implementations. *Journal of Machine Learning Research*, 22(268), 1–8.
- Raffin, A., Seidel, D., Kober, J., Albu-Schäffer, A., Silvério, J., & Stulp, F. (2022). Learning to exploit elastic actuators for quadruped locomotion. arXiv:2209.07171.
- Rudin, N., Hoeller, D., Reist, P., & Hutter, M. (2022). Learning to walk in minutes using massively parallel deep reinforcement learning. *CoRL*.
- Ryu, K., Liao, Q., Li, Z., Delgosha, P., Sreenath, K., & Mehr, N. (2025). CurricuLLM: Automatic task curricula design for learning complex robot skills using large language models. *ICRA*. arXiv:2409.18382.
- Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016). High-dimensional continuous control using generalized advantage estimation. *ICLR*. arXiv:1506.02438.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. arXiv:1707.06347.
- Shah, D., Osiński, B., Ichter, B., & Levine, S. (2022). LM-Nav: Robotic navigation with large pre-trained models of language, vision, and action. *CoRL*.
- Shinn, N., Cassano, F., Labash, B., Gopinath, A., Narasimhan, K., & Yao, S. (2023). Reflexion: Language agents with verbal reinforcement learning. *NeurIPS*. arXiv:2303.11366.
- Singh, S., Lewis, R. L., & Barto, A. G. (2010). Where do rewards come from? *Proceedings of the International Symposium on AI-Inspired Biology*, 111–116.
- Skalse, J., Howe, N. H. R., Krasheninnikov, D., & Krueger, D. (2022). Defining and characterizing reward hacking. *NeurIPS 35*.
- Song, J., Zhou, Z., Liu, J., Fang, C., Shu, Z., & Ma, L. (2023). Self-refined large language model as automated reward function designer for deep reinforcement learning in robotics. arXiv:2309.06687.
- STRIDE authors. (2025). STRIDE: Automating reward design, deep reinforcement learning training and feedback optimization in humanoid robotics locomotion. arXiv:2502.04692 *(preprint)*.
- Sun, S., Liu, R., Lyu, J., Yang, J.-W., Zhang, L., & Li, X. (2025). A large language model-driven reward design framework via dynamic feedback for reinforcement learning (CARD). *Knowledge-Based Systems*, 326, 114065.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Tang, Y., Yu, W., Tan, J., Zen, H., Faust, A., & Harada, T. (2023). SayTap: Language to quadrupedal locomotion. *CoRL*, PMLR 229. arXiv:2306.07580.
- Todorov, E., Erez, T., & Tassa, Y. (2012). MuJoCo: A physics engine for model-based control. *IEEE/RSJ IROS*, 5026–5033.
- Towers, M., Kwiatkowski, A., Terry, J., et al. (2024). Gymnasium: A standard interface for reinforcement learning environments. arXiv:2407.17032.
- Vemprala, S., Bonatti, R., Bucker, A., & Kapoor, A. (2024). ChatGPT for robotics: Design principles and model abilities. *IEEE Access*.
- Wang, G., Xie, Y., Jiang, Y., et al. (2023). Voyager: An open-ended embodied agent with large language models. arXiv:2305.16291.
- Wang, Y., Tang, Y., Liu, B., Liu, X., & Shang, D. (2026). When LLM reward design fails: Diagnostic-driven refinement for sparse structured RL. arXiv:2605.28918 *(preprint)*.
- Wang, Y.-J., Zhang, B., Chen, J., & Sreenath, K. (2024). Prompt a robot to walk with large language models. *IEEE Conference on Decision and Control (CDC)*. arXiv:2309.09969.
- Xie, T., Zhao, S., Wu, C. H., Liu, Y., Luo, Q., Zhong, V., Yang, Y., & Yu, T. (2024). Text2Reward: Reward shaping with language models for reinforcement learning. *ICLR*. arXiv:2309.11489.
- Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023). ReAct: Synergizing reasoning and acting in language models. *ICLR*.
- Yu, W., Gileadi, N., Fu, C., et al. (2023). Language to rewards for robotic skill synthesis. *CoRL*, PMLR 229, 374–404. arXiv:2306.08647.
- Ziebart, B. D., Maas, A. L., Bagnell, J. A., & Dey, A. K. (2008). Maximum entropy inverse reinforcement learning. *Proceedings of AAAI*.

Software documentation consulted: Gymnasium Ant-v5 documentation (https://gymnasium.farama.org/environments/mujoco/ant/); MuJoCo changelog (https://mujoco.readthedocs.io/); Stable-Baselines3 changelog (https://stable-baselines3.readthedocs.io/); Anthropic API documentation (https://docs.claude.com/en/api/overview).

*Note for the student:* several author lists above are abbreviated with "et al." or attributed to "authors" where the full list could not be verified during drafting. Before submission, complete every entry from the official arXiv or proceedings page and convert to your department's citation style.

---

## Appendix A. Prompt templates

These templates are starting points; the final versions are frozen after the development phase and released with the transcripts. Text in braces is filled by the orchestrator.

### A.1 Designer system prompt

```
You are a reward and controller engineer for a simulated quadruped (Gymnasium Ant-v5,
MuJoCo). Your goal is to produce files that, when used to train a policy with a fixed
PPO trainer, yield behavior that scores highly on a FROZEN evaluator you cannot change.

You write two files:
  1. reward.py  — defines compute_reward(s, a, s_next, ctx) -> (total, components)
  2. controller.yaml — selects action mode and parameters within the schema bounds

Rules:
- Use only fields documented in the StateView API. Never index raw observation arrays.
- Every additive reward term must appear by name in `components`; every scale or
  temperature must be a named constant at module level.
- Allowed imports: math, numpy as np, lib.reward_helpers. No I/O.
- Avoid large per-step bonuses for merely surviving or standing; they let the policy
  satisfy the reward without performing the task.
- Prefer rewards whose components each move toward the evaluator's objectives;
  the evaluator description is provided so you know what success means.
- Call validate() before submit(). Fix all errors; consider all warnings.
- Submit exactly {n_candidates} candidates. For each, write a 2–4 sentence rationale
  stating the hypothesis being tested.

Your strategy for this generation: {strategy_instruction}
```

### A.2 Designer user message

```
TASK SPECIFICATION AND EVALUATOR:
{get_spec output}

ENVIRONMENT API, HELPERS, CONTROLLER SCHEMA, CALIBRATION:
{get_env_api output}

ARCHIVE (top 3 candidates with files, fitness breakdown, and Analyst diagnoses):
{get_archive output}

PREVIOUS GENERATION FAILURE SUMMARIES:
{failure summaries}

Budget status: generation {g} of {G}; remaining environment steps {remaining}.
Produce your candidates now.
```

### A.3 Analyst system prompt

```
You are a locomotion diagnostics analyst. You receive, for ONE trained candidate:
reward-component trajectories over training, the frozen evaluator's report,
the training-return vs. fitness correlation, automated failure flags, a foot-contact
gait diagram, and keyframe images.

Produce JSON with fields:
  "failure_modes": list of flags from {F1..F9} you confirm, or "none",
  "evidence": for each mode, cite specific numbers, frames, or gait-diagram features,
  "likely_cause": the reward term(s) or controller setting(s) responsible,
  "suggested_edits": up to three concrete, minimal edits (name the constant or field
                     and the direction/size of change),
  "keep": aspects of the candidate that are working and should be preserved.
Do not speculate beyond the evidence. If images and numbers disagree, say so.
```

### A.4 Spec Interpreter system prompt

```
You convert a user's BehaviorSpec into an evaluator built ONLY from the provided metric
primitives. Check whether the free-text description implies properties not yet covered
by objectives or constraints (orientation, stationarity, which leg, direction frame,
duration of holding a pose). Propose additions with a one-line justification each.
If something is ambiguous and would change the evaluator, ask the user (max 3 questions)
using ask_user. Output the evaluator configuration via propose_evaluator.
```

### A.5 Supervisor system prompt (Phase 3)

```
You command a quadruped robot through a velocity interface. A fast controller keeps it
balanced; a reflex layer prevents collisions and falls and will report when it acts.
Mission: {mission_text}
Every simulated second you receive a JSON sensor summary. Respond with exactly one tool
call: set_command(vx, vy, yaw_rate, duration_s, reason), hold(), or note(text) followed
by one of the others. Limits: vx in [-0.5, 1.0] m/s, vy in [-0.4, 0.4] m/s,
yaw_rate in [-0.8, 0.8] rad/s. Positive yaw_rate turns left. Bearings are degrees,
positive to the robot's left. Keep reasons under 20 words.
```

---

## Appendix B. Repository layout and run commands

```
agentic-locomotion/
├── pyproject.toml, uv.lock          # pinned environment
├── configs/
│   ├── machine/                     # Phase-0 outputs: throughput, rungs, calibration.json
│   ├── tasks/T1.yaml … T10.yaml     # BehaviorSpecs
│   ├── evaluators/T1.json …         # frozen evaluator configs (+ SHA-256)
│   └── conditions/C0.yaml … O3.yaml, A1.yaml … A6.yaml
├── lib/
│   ├── state_view.py                # StateView and contact detection
│   ├── wrappers.py                  # Controller, Command, Observation, Reward wrappers
│   ├── cpg.py                       # Hopf-oscillator CPG
│   ├── metrics.py                   # metric primitives (+ tests/)
│   ├── evaluator.py                 # scoring and success judgment
│   ├── reward_helpers.py            # kernels and quaternion utilities for agents
│   ├── sandbox.py                   # AST whitelist, subprocess execution, limits
│   ├── reference_gaits/             # hand-written gaits for the flooding probe
│   └── arena.py                     # AntStimulusArena, sensors, summarizer, reflexes
├── agents/
│   ├── prompts/                     # Appendix A templates (frozen)
│   ├── tools.py                     # tool schemas and implementations
│   ├── designer.py, analyst.py, spec_interpreter.py, supervisor.py
│   └── llm_client.py                # provider wrapper + JSONL transcript logging
├── orchestrator/
│   ├── search.py                    # Algorithm 1, successive halving, budget accounting
│   ├── train.py                     # SB3 training/resume jobs
│   └── replay.py                    # retrain/evaluate logged candidates without LLM calls
├── analysis/                        # rliable statistics, figures, variance ANOVA
└── runs/                            # content-addressed artifacts and transcripts
```

Typical commands:

```bash
uv sync                                            # create the pinned environment
uv run python -m orchestrator.phase0 --all         # benchmark, validate trainer, calibrate
uv run python -m agents.spec_interpreter configs/tasks/T5.yaml   # interactive approval
caffeinate -i uv run python -m orchestrator.search --task T5 --condition O3 --run 1
uv run python -m orchestrator.replay runs/T5/O3/run1          # reproduce downstream of LLM
uv run python -m analysis.report --tasks all --conditions all
```

A minimal example of a valid reward file for task T1, of the kind a Designer might produce in generation 1:

```python
import math
import numpy as np
from lib.reward_helpers import gaussian

TARGET_VX = 1.0        # m/s, from the spec
VEL_TEMP = 0.25        # kernel width for velocity tracking
UPRIGHT_W = 0.2
LATERAL_W = 0.5
CTRL_W = 0.02
SMOOTH_W = 0.01

def compute_reward(s, a, s_next, ctx):
    vel = gaussian(s_next.lin_vel_world[0] - TARGET_VX, VEL_TEMP)   # in [0, 1]
    upright = UPRIGHT_W * max(0.0, s_next.up_z)
    lateral = -LATERAL_W * abs(s_next.lin_vel_world[1])
    ctrl = -CTRL_W * float(np.sum(np.square(a)))
    smooth = -SMOOTH_W * float(np.sum(np.square(a - s.prev_action)))
    comps = {"velocity": vel, "upright": upright, "lateral": lateral,
             "control": ctrl, "smoothness": smooth}
    return float(sum(comps.values())), comps
```

Note that the upright bonus is deliberately small relative to the velocity term, which avoids the survival-exploitation failure (flag F2) that a large constant "alive" bonus can cause.
